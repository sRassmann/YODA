import sys, os

from generative.networks.nets import DiffusionModelUNet
from omegaconf import DictConfig, OmegaConf

# append cwd ("[...]/YODA") to sys.path for custom imports
sys.path.append(os.path.realpath(os.getcwd()))

from lib.utils.monai_helper import parse_dtype, scheduler_factory
from lib.utils.etc import (
    print0,
    remove_module_in_state_dict,
    get_ema_checkpoint,
)
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import time
import argparse
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from lib.utils.visualization import save_grid, create_val_batch
from lib.datasets import create_loaders
from lib.utils.run_utils import (
    add_arguments_from_config,
    create_run_dirs,
    link_arguments,
    merge_cli_config,
    setup_ddp,
)
from generative.metrics import SSIMMetric
from ema_pytorch import EMA

from lib.custom_nets.concat_inferer import ThickSliceInferer

# slight performance boost for persistent caching
torch.multiprocessing.set_sharing_strategy("file_system")
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8096, rlimit[1]))


def main(run_name, config, resume_ckpt=None):
    rank, world_size = setup_ddp()

    run_path, sample_path, ckpt_path, tb_path = create_run_dirs(run_name, config)
    train_loader, val_loader = create_loaders(**config["data"])
    target_sequence = config["data"]["target_sequence"]
    guidance_sequence = config["data"]["guidance_sequences"]

    if world_size > 1:
        assert (
            config["trainer"]["num_val_samples"] >= world_size
        ), "num_examples must be >= world_size"

    val_cache_path = config.trainer.get("val_cache_path", "../data/cache/RS/val_slices")
    if config.data.slice_thickness > 1:
        val_cache_path += f"_{config.data.slice_thickness}"
    const_val_batch = create_val_batch(
        num_examples=config["trainer"]["num_val_samples"],
        val_loader=val_loader,
        relevant_sequences=config["data"]["guidance_sequences"]
        + [config["data"]["target_sequence"]],
        val_cache_path=val_cache_path,
        sample_path=sample_path,
    )

    if rank == 0:
        writer = SummaryWriter(log_dir=tb_path)

    device = torch.device(f"cuda:{rank}")
    dtype = parse_dtype(config["trainer"]["dtype"])
    model = DiffusionModelUNet(**config["model"]["unet"])
    start_epoch = 0
    if resume_ckpt is not None:
        print0(f"Resuming training from {resume_ckpt}")
        model.load_state_dict(
            remove_module_in_state_dict(
                get_ema_checkpoint(
                    torch.load(
                        resume_ckpt,
                        map_location=device,  # avoid loading on GPU 0 in all DDP ranks
                    )
                )
            ),
        )
        if "epoch" in resume_ckpt:
            start_epoch = (
                int(os.path.basename(resume_ckpt).split(".")[0].split("_")[1]) + 1
            )
            print0(f"Starting from epoch {start_epoch}.")
    model = model.to(device)
    ema_beta = config["trainer"].get("ema_beta", 0)
    if ema_beta > 0:
        ema = EMA(
            model,
            beta=ema_beta,
            update_after_step=1000,
            update_every=1,
        )
    if world_size > 1:
        model = DDP(
            model, device_ids=[rank], output_device=rank, find_unused_parameters=True
        )
    # model = torch.compile(model)  # should be after DDP for efficient bucketing
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.trainer.lr)
    loss_fn = (
        torch.nn.MSELoss(reduction="none")
        if config["model"]["loss"] == "l2"
        else torch.nn.L1Loss(reduction="none")
    )
    scaler = GradScaler()

    scheduler = scheduler_factory(config["model"]["train_noise_sched"])
    inferer = ThickSliceInferer(
        scheduler, config.data.slice_thickness, config.model.thick_target
    )

    total_start = time.time()
    for epoch in range(start_epoch, config["trainer"]["n_epochs"]):
        model.train()
        epoch_loss = 0
        if rank == 0 and config["trainer"]["show_progress_bar"]:
            progress_bar = tqdm(
                enumerate(train_loader), total=len(train_loader), ncols=110
            )
            progress_bar.set_description(f"Epoch {epoch}")
        else:
            progress_bar = enumerate(train_loader)

        optimizer.zero_grad(set_to_none=True)
        for step, batch in progress_bar:
            images = batch[target_sequence]  # target
            if config.data.slice_thickness > 1 and not config.model.thick_target:
                # dropout adjacent slices to simulate border effects and force to model to learn the generation fully by itself
                tar_ind = config.data.slice_thickness // 2
                if torch.rand(1) < config.model.adj_slice_dropout_p:
                    images[:, 0:tar_ind] = -1
                if torch.rand(1) < config.model.adj_slice_dropout_p:
                    images[:, tar_ind + 1 :] = -1

            guidance = torch.concat(
                [batch[seq] for seq in guidance_sequence], dim=1
            ).to(device)

            images = torch.Tensor(images).to(device)
            guidance = torch.Tensor(guidance).to(device)
            if config.trainer.get("weighting", "") == "brain":
                weight = torch.Tensor(batch["mask"] > 0) * 0.99 + 0.01
                weight = weight.to(dtype).to(device)
            else:
                weight = 1

            with autocast(enabled=True, dtype=dtype):
                loss = shared_step(
                    images,
                    guidance,
                    weight,
                    model,
                    loss_fn,
                    inferer,
                    config,
                )

            scaler.scale(loss).backward()

            if (step + 1) % config.trainer.gradient_accumulation_steps == 0:
                scaler.step(optimizer)  # Optimizer step with scaled gradients
                scaler.update()  # Update scaler
                optimizer.zero_grad()
            if ema_beta > 0:
                ema.update()

            epoch_loss += loss.item()

            if rank == 0 and config["trainer"]["show_progress_bar"]:
                progress_bar.set_postfix(
                    {
                        "loss": epoch_loss / (step + 1),
                    }
                )

        if rank == 0:
            writer.add_scalar("Loss/train", epoch_loss / len(train_loader), epoch)

        if (epoch + 1) % config["trainer"]["val_interval"] == 0:
            eval_model = ema if ema_beta > 0 else model
            eval_model.eval()
            with torch.no_grad():
                val_loss = 0  # Initialize validation loss

                for val_step, val_batch in enumerate(val_loader):
                    images = val_batch[target_sequence].to(device)  # target
                    guidance = torch.concat(
                        [val_batch[seq] for seq in guidance_sequence], dim=1
                    ).to(device)
                    images = torch.Tensor(images).to(device)
                    guidance = torch.Tensor(guidance).to(device)

                    if config.data.get("weighting", "") == "brain":
                        weight = torch.Tensor(val_batch["brain_mask"] > 0) * 0.99 + 0.01
                        weight = weight.to(dtype).to(device)
                    else:
                        weight = 1

                    with autocast(enabled=True, dtype=dtype):
                        loss = shared_step(
                            images,
                            guidance,
                            weight,
                            eval_model,
                            loss_fn,
                            inferer,
                            config,
                        )
                    val_loss += loss.item()  # Accumulate the validation loss

                # Calculate the average validation loss
                avg_val_loss = val_loss / len(val_loader)
                if rank == 0:
                    writer.add_scalar("Loss/val", avg_val_loss, epoch)

                noise = torch.randn(
                    *const_val_batch[target_sequence].shape,
                    device=device,
                )
                val_guidance = torch.concat(
                    [const_val_batch[seq] for seq in guidance_sequence], dim=1
                ).to(device)

                if ema_beta == 0 and world_size > 1:
                    eval_model = eval_model.module

                t = inferer.scheduler.num_train_timesteps - 1
                image_reg = -eval_model(
                    torch.cat([noise, val_guidance], dim=1),
                    torch.Tensor([t] * noise.shape[0]).long().to(device),
                )
                image = inferer.sample(
                    noise,
                    eval_model,
                    scheduler,
                    guidance_sequences=val_guidance,
                    adjacent_guidance=const_val_batch[target_sequence]
                    if config.model.adj_slice_dropout_p == 0
                    else None,
                    verbose=config["trainer"]["show_progress_bar"],
                    dtype=dtype,
                )

                # calculate psnr and SSIM
                gt = const_val_batch[target_sequence].to(device)
                if config.data.slice_thickness > 1 and not config.model.thick_target:
                    gt = gt[
                        :,
                        config.data.slice_thickness
                        // 2 : config.data.slice_thickness
                        // 2
                        + 1,
                    ]

                psnr = 10 * torch.log10(
                    1 / torch.mean((image - gt) ** 2, dim=[1, 2, 3])
                )
                psnr_reg = 10 * torch.log10(
                    1 / torch.mean((image_reg - gt) ** 2, dim=[1, 2, 3])
                )
                spatial_dims = 2  # 3 if config.data.slice_thickness > 1 and config.data.thick_target else 2
                ssim = SSIMMetric(spatial_dims)(image, gt)
                ssim_reg = SSIMMetric(spatial_dims)(image_reg, gt)

            if world_size > 1:
                images = [torch.zeros_like(image) for _ in range(world_size)]
                dist.all_gather(images, image.contiguous())

                psnrs = [torch.zeros_like(psnr) for _ in range(world_size)]
                dist.all_gather(psnrs, psnr)

                ssims = [torch.zeros_like(ssim) for _ in range(world_size)]
                dist.all_gather(ssims, ssim)

                psnrs_reg = [torch.zeros_like(psnr) for _ in range(world_size)]
                dist.all_gather(psnrs_reg, psnr_reg)

                ssims_reg = [torch.zeros_like(ssim) for _ in range(world_size)]
                dist.all_gather(ssims_reg, ssim_reg)

                if rank == 0:
                    image = torch.cat(images, dim=0)
                    psnr = torch.cat(psnrs, dim=0)
                    ssim = torch.cat(ssims, dim=0)
                    psnr_reg = torch.cat(psnrs_reg, dim=0)
                    ssim_reg = torch.cat(ssims_reg, dim=0)

            if rank == 0:
                print0(f"saving samples to {sample_path}.")
                save_grid(
                    image,
                    sample_path + f"/epoch_{epoch:04d}.png",
                    columns=4,
                )
                writer.add_scalar("PSNR", psnr.mean(), epoch)
                writer.add_scalar("SSIM", ssim.mean(), epoch)
                writer.add_scalar("PSNR_reg", psnr_reg.mean(), epoch)
                writer.add_scalar("SSIM_reg", ssim_reg.mean(), epoch)
                # save last checkpoint
                torch.save(
                    eval_model.state_dict(), os.path.join(ckpt_path, f"last.pth")
                )

                if (  # save checkpoint every n epochs
                    config["trainer"]["checkpoint_interval"] > 0
                    and (epoch + 1) % config["trainer"]["checkpoint_interval"] == 0
                ):
                    torch.save(
                        eval_model.state_dict(),
                        os.path.join(ckpt_path, f"epoch_{epoch:04d}.pth"),
                    )

    if rank == 0:
        writer.close()
    if world_size > 1:
        dist.destroy_process_group()
    total_time = time.time() - total_start
    print0(f"train completed, total time: {total_time}.")


def shared_step(
    images,
    guidance,
    weight,
    model,
    loss_fn,
    inferer,
    config,
):
    device = images.device
    noise = torch.randn_like(images, device=device)
    timesteps = (
        torch.randint(
            0,
            inferer.scheduler.num_train_timesteps,
            (images.shape[0],),
        )
        .long()
        .to(device)
    )

    # get target
    if (
        not config.model.get("thick_target_slices", 0)
        and config.data.get("slice_thickness", 1) > 1
    ):
        chan_ind_start = config.data.slice_thickness // 2
        chan_ind_end = chan_ind_start + 1
    else:
        chan_ind_start, chan_ind_end = 0, images.shape[1] + 1
    target = inferer.get_pred_target(
        images[:, chan_ind_start:chan_ind_end],
        noise[:, chan_ind_start:chan_ind_end],
        timesteps,
    )

    # predict noise
    noise_pred = inferer(
        inputs=images.float(),  # don't use fp16 to avoid casting of scheduler
        diffusion_model=model,
        noise=noise,
        timesteps=timesteps,
        condition=guidance,
        mode="concat",
    )
    if isinstance(weight, torch.Tensor):
        weight = weight[:, chan_ind_start:chan_ind_end]

    loss = loss_fn(noise_pred, target) * weight
    return loss.mean()


if __name__ == "__main__":
    default_config_path = "configs/defaults.yml"
    default_config = OmegaConf.load(default_config_path)

    parser = argparse.ArgumentParser(description="Merge multiple configuration files.")
    parser.add_argument(
        "-n", "--name", type=str, default="debug", help="Name of the run"
    )
    parser.add_argument(
        "--resume_ckpt",
        "-r",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "configs",
        nargs="*",
        type=str,
        help="Paths to configuration files",
    )

    add_arguments_from_config(parser, default_config)
    args = parser.parse_args()

    # merge individual configs and overwrite with CLI args
    merged_config = merge_cli_config(args, default_config_path)
    merged_config = link_arguments(merged_config)

    if args.resume_ckpt is None and "resume_ckpt" in merged_config:
        args.resume_ckpt = merged_config.resume_ckpt
    if args.resume_ckpt is not None:
        merged_config.resume_ckpt = args.resume_ckpt

    main(args.name, merged_config, args.resume_ckpt)
