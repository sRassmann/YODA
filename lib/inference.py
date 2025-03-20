import os
import warnings
from matplotlib import pyplot as plt

from typing import Any, Hashable, Mapping, Optional, Sequence, Union, List, Dict, Tuple

import numpy as np
import torch
import cv2
from monai.data import MetaTensor
from tqdm import tqdm
import nibabel as nib

from generative.networks.schedulers import DDIMScheduler
from generative.metrics import SSIMMetric


from lib.custom_nets.concat_inferer import ThickSliceInferer
from lib.utils.etc import (
    remove_module_in_state_dict,
    get_ema_checkpoint,
)
from lib.utils.monai_helper import scheduler_factory
from generative.networks.nets import DiffusionModelUNet


def rms_combine(echos):
    echos = torch.stack(echos, dim=0) + 1
    echos = torch.clamp(echos, 0, 2)
    for i in range(1, len(echos)):
        if torch.allclose(echos[i - 1], echos[i]):
            warnings.warn(
                "Equal or similar echos found! This is unexpected and is likely caused by wrong torch seeding."
            )
    return torch.sqrt(torch.mean(echos**2, dim=0)) - 1


def image_to_int(o, norm_by_quantile=False):
    if norm_by_quantile:
        l, u = np.quantile(o, 0.05), np.quantile(o, 0.995)
        o = (o - l) / (u - l) * 255
    else:
        o = (o + 1) * 255 / 2
        o = np.clip(o, 0, 255)
    o = np.clip(o, 0, 255)
    o = o.astype(np.uint8)
    return o


# noinspection PyTypeChecker
def save_output_png(sequences, indices, output_dir, pred, vol, quantile_norm=False):
    # store images as png
    for i, slice_ind in enumerate(indices):
        os.makedirs(os.path.join(output_dir, vol["subject_ID"]), exist_ok=True)
        cv2.imwrite(
            os.path.join(
                output_dir,
                vol["subject_ID"],
                f"{slice_ind:03d}_pred.png",
            ),
            image_to_int(pred[i, 0], quantile_norm),
        )

        for seq in sequences:
            if seq != "mask":
                cv2.imwrite(
                    os.path.join(
                        output_dir,
                        vol["subject_ID"],
                        f"{slice_ind:03d}_{seq}.png",
                    ),
                    image_to_int(vol[seq][0, slice_ind].cpu().numpy(), quantile_norm),
                )
            else:
                cv2.imwrite(
                    os.path.join(
                        output_dir,
                        vol["subject_ID"],
                        f"{slice_ind:03d}_mask.png",
                    ),
                    vol["mask"][0, slice_ind].cpu().numpy().astype(np.uint8),
                )


def save_output_volume(
    data_dict: Dict[str, MetaTensor],
    output_path: str,
    affine_sequence: Optional[str] = None,
    target_sequence: str = "flair",
    save_keys: Sequence[str] = ("t1", "t2", "flair", "pred"),
    quantile_norm: bool = False,
) -> None:
    """
    Save images from a dict to as nifti files.

    Images are stored as {output_path}/{data_dict['subject_ID']}/<name>.nii.gz.

    Args:
        data_dict: dict with at least the keys "subject_ID", "pred", at least one
          guidance sequence and optional other sequences.
        output_path: path to the output directory.
        affine_sequence: the sequence to use for the affine and header information for
          the predicted sequence. If None, the first sequence in the dict will be used.
        target_sequence: name of the target sequence, used for the output file name.
        save_keys: the keys to save, defaults to ("t1", "t2", "flair", "pred").
        quantile_norm: whether to normalize the images by quantiles.
    """
    if not target_sequence:
        target_sequence = "flair"
    save_keys = [k for k in save_keys if k in data_dict]
    subj_output_path = os.path.join(output_path, data_dict["subject_ID"])
    os.makedirs(subj_output_path, exist_ok=True)

    if "pred" not in save_keys:
        print("pred not in save_keys, adding it")
        save_keys = save_keys + ("pred",)

    if affine_sequence is None:
        affine_sequence = save_keys[0]

    for key in save_keys:
        # create nibabel object from tensor with original affine and header
        o = data_dict[key].squeeze().numpy()
        if key != "mask":
            o = image_to_int(o, quantile_norm)
        else:
            o = o.astype(np.uint8)

        if "pred" in key:
            nib.save(
                nib.Nifti1Image(o, data_dict[affine_sequence].meta["affine"]),
                f"{subj_output_path}/{key}_{target_sequence}.nii.gz",
            )
        else:
            nib.save(
                nib.Nifti1Image(o, data_dict[key].meta["affine"]),
                f"{subj_output_path}/{key}.nii.gz",
            )


class Sr3InferenceModel(torch.nn.Module):
    def __init__(
        self,
        config,
        ckpt,
        device="cuda",
        dtype=torch.float16,
        verbose=True,
        motion_score=0,
    ):
        super().__init__()
        self.config = config

        model = DiffusionModelUNet(**config["model"]["unet"])

        state_dict = remove_module_in_state_dict(get_ema_checkpoint(torch.load(ckpt)))
        model.load_state_dict(state_dict)
        self.model = model.to(device).eval()

        scheduler = scheduler_factory(config["model"]["train_noise_sched"])
        scheduler.set_timesteps(config["model"]["num_inference_steps"])
        self.inferer = ThickSliceInferer(
            scheduler,
            config.data.get("slice_thickness", 1),
            config.model.get("thick_target", False),
        )
        self.scheduler = scheduler
        self.motion_score = motion_score

        self.device = device
        self.dtype = dtype
        self.verbose = verbose

    @torch.no_grad()
    def predict_slice(self, guidance):
        if (
            self.config.model.get("adj_slice_dropout_p", 0) == 0
            and self.config.data.get("slice_thickness", 1) > 1
        ):
            warnings.warn(
                "adj_slice_dropout_p is set to 0 for training, hence the model was not trained on the normal sampling algorithm which pads black space as guidance."
            )
        noise = torch.rand(
            guidance.shape[0],
            1,
            guidance.shape[2],
            guidance.shape[3],
            device=self.device,
        ).to(torch.float16)
        guidance = guidance.to(self.device).to(self.dtype)
        image = self.inferer.sample(
            noise,
            self.model,
            guidance_sequences=guidance,
            adjacent_guidance=None,
            verbose=True,
            dtype=self.dtype,
        )
        return image

    @torch.no_grad()
    def synchronous_volume_denoising(self, guidance, batch_size=8):
        """
        slice-wise parallel translation of a guidance sequence (C D H W)

        Note: it is assumed that the guidance is in the right view, ie. the slicing
         direction is the second dimension (D).
        """
        # guidance = guidance[:, 100:102]  # fast debug

        # transpose to D C H W
        guidance_org = guidance.transpose(0, 1).to(self.device).to(self.dtype)

        # noise shape D C H W, where C = 1 (single output modality)
        D, C, H, W = guidance_org.shape

        # pad guidance to -1, img_size, img_size
        img_size = self.config.data.img_size[0]
        guidance = -torch.ones((D, C, img_size, img_size), device=self.device).to(
            self.dtype
        )
        offset_H = (img_size - H) // 2
        offset_W = (img_size - W) // 2
        guidance[:, :, offset_H : offset_H + H, offset_W : offset_W + W] = guidance_org

        noise = torch.randn(D, 1, img_size, img_size, device=self.device).to(self.dtype)

        if self.config.data.get("slice_thickness", 1) == 1:
            # all on GPU
            image = noise
            iterator = (
                tqdm(self.scheduler.timesteps, ncols=70)
                if self.verbose
                else self.scheduler.timesteps
            )
            for t in iterator:
                x = torch.cat([image, guidance], dim=1)
                model_outputs = []
                for i in range(0, D, batch_size):
                    with torch.autocast(dtype=self.dtype, device_type=self.device):
                        t_tens = torch.Tensor((t,)).long().to(self.device)
                        model_output = self.model(
                            x[i : i + batch_size],
                            t_tens,
                            class_labels=torch.zeros_like(t_tens).long()
                            + self.motion_score,
                        )
                    model_outputs.append(model_output)
                pred_volume = torch.cat(model_outputs, dim=0)
                image, _ = self.scheduler.step(pred_volume.float(), t, image.float())

                # if t % 20 == 19 or t == 0:
                #     for i in range(2):
                #         img = np.array(_[i].squeeze().cpu())
                #         img -= np.percentile(img, 1)
                #         img /= np.percentile(img, 99)
                #         img = np.clip(img, 0, 1)
                #         img = (img * 255).astype(np.uint8)
                #         cv2.imwrite(
                #             f"/home/rassmanns/diffusion/flairsyn/output/sr3/deregister/inference_progress/{t}_{i}.png",
                #             img,
                #         )

            image = image.transpose(0, 1)

        # multi input, multi output
        elif self.config.model.thick_target:
            thick = self.config.data.slice_thickness
            image = noise
            iterator = (
                tqdm(self.scheduler.timesteps, ncols=70)
                if self.verbose
                else self.scheduler.timesteps
            )
            for t in iterator:
                x = torch.cat([image, guidance], dim=1)
                model_outputs = []
                offset = t.item() % (thick // 2 + 1)  # e.g. thick = 5 -> 0,1,2

                i = thick - offset
                batch = [x[:thick]]  # bottom and top slices

                samples_left = True
                while samples_left:
                    # fetch batches
                    while len(batch) < batch_size:
                        batch.append(x[i : i + thick])
                        i += thick
                        if i + thick >= D:
                            samples_left = False
                            batch.append(x[-thick:])
                            break
                    with torch.autocast(dtype=self.dtype, device_type=self.device):
                        model_input = torch.stack(batch)
                        model_input = model_input.transpose(1, 2).reshape(
                            model_input.shape[0], -1, H, W
                        )
                        t_tens = torch.Tensor((t,)).long().to(self.device)
                        model_output = self.model(
                            model_input,
                            t_tens,
                            class_labels=torch.zeros_like(t_tens).long()
                            + self.motion_score,
                        )
                    model_outputs += [
                        model_output[i] for i in range(model_output.shape[0])
                    ]
                    batch = []
                top = model_outputs.pop(0)[: thick - offset]
                bottom = model_outputs.pop(-1)[-(D - i) :]
                pred_volume = torch.cat([top] + model_outputs + [bottom], dim=0)
                pred_volume.unsqueeze_(1)
                assert pred_volume.shape[0] == D
                image, _ = self.scheduler.step(pred_volume.float(), t, image.float())
            image = image.transpose(0, 1)

        # multi input, single output
        else:
            half_thick = self.config.data.slice_thickness // 2

            # append black space above and below, NOTE: Offset of the D indices from now
            black = -torch.ones(
                half_thick, 1, img_size, img_size, device=noise.device
            ).to(self.dtype)
            noise = torch.cat([black, noise, black], dim=0)

            black = -torch.ones(
                half_thick, C, img_size, img_size, device=noise.device
            ).to(self.dtype)
            guidance = torch.cat([black, guidance, black], dim=0)
            image = noise

            iterator = (
                tqdm(self.scheduler.timesteps, ncols=70)
                if self.verbose
                else self.scheduler.timesteps
            )
            for t in iterator:
                # Note that this is shorter due to missing black space
                model_outputs = []
                for i in range(0, D, batch_size):  # batching
                    batch = []
                    for j in range(min(batch_size, D - i)):
                        sl_noise = noise[i + j : i + j + 2 * half_thick + 1]
                        sl_guidance = guidance[i + j : i + j + 2 * half_thick + 1]
                        # S (1+C) H W -> S*(1+C) H W
                        sl = torch.cat(
                            [sl_noise[:, 0]] + [sl_guidance[:, c] for c in range(C)]
                        )
                        batch.append(sl)  # C' H W
                    batch = torch.stack(batch)
                    with torch.autocast(dtype=self.dtype, device_type=self.device):
                        t_tens = torch.Tensor((t,)).long().to(self.device)
                        model_output = self.model(
                            batch,
                            t_tens,
                            class_labels=torch.zeros_like(t_tens).long()
                            + self.motion_score,
                        )
                    model_outputs.append(model_output)
                pred_volume = torch.cat(model_outputs, dim=0)
                image_p, _ = self.scheduler.step(
                    pred_volume.float(), t, image[half_thick:-half_thick]
                )
                image[half_thick:-half_thick] = image_p
            image = image_p.transpose(0, 1)
        return image.cpu()[:, :, offset_H : offset_H + H, offset_W : offset_W + W]


class Sr3MultiViewInferenceModel(torch.nn.Module):
    def __init__(
        self,
        axial_config,
        coronal_config,
        sagittal_config,
        axial_ckpt,
        coronal_ckpt,
        sagittal_ckpt,
        device="cuda",
        dtype=torch.float16,
        verbose=True,
        n_exitations=1,  # number of final images to average
        mex_step=50,  # step at which to diverge to different images
        # nex_method="rms",  # method to combine predicted images
        lazy_sampling_step=0,  # set > 200 for obtaining reasonable variability
        motion_score=0,
    ):
        super().__init__()
        self.config = axial_config
        assert (
            not lazy_sampling_step
            or mex_step <= lazy_sampling_step
            or n_exitations <= 1
        )

        self.models = {}
        for view, config, ckpt in zip(
            ["axial", "coronal", "sagittal"],
            [axial_config, coronal_config, sagittal_config],
            [axial_ckpt, coronal_ckpt, sagittal_ckpt],
        ):
            if not config:
                continue
            model = DiffusionModelUNet(**config["model"]["unet"])
            state_dict = remove_module_in_state_dict(
                get_ema_checkpoint(torch.load(ckpt))
            )
            model.load_state_dict(state_dict)
            self.models[view] = model.to(device).eval()
            for key in ["prediction_type", "num_train_timesteps", "schedule"]:
                assert (
                    self.config.model.train_noise_sched[key]
                    == config.model.train_noise_sched[key]
                )
            assert self.config.data.slice_thickness == config.data.slice_thickness

        scheduler = scheduler_factory(self.config["model"]["train_noise_sched"])
        scheduler.set_timesteps(self.config["model"]["num_inference_steps"])

        self.inferer = ThickSliceInferer(
            scheduler,
            self.config.data.get("slice_thickness", 1),
            self.config.model.get("thick_target", False),
        )
        self.scheduler = scheduler
        self.nex = n_exitations
        self.mex_diversion_step = mex_step
        self.mex_method = rms_combine
        self.layz_sampling_step = lazy_sampling_step
        self.motion_score = motion_score

        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.gt_image = None
        self.psnr = []
        self.ssim = []

    @torch.no_grad()
    def synchronous_volume_denoising(self, guidance, batch_size=8):
        """
        slice-wise parallel translation of a guidance sequence (C D H W)
        """
        img_size = self.config.data.img_size[0]

        guidance = guidance.to(self.device).to(self.dtype)  # C D H W

        # noise shape D C H W, where C = 1 (single output modality)
        C, D, H, W = guidance.shape

        # pad to cube
        guidance_pad = -torch.ones(
            (C, img_size, img_size, img_size), device=self.device
        ).to(self.dtype)
        offset_D = (img_size - D) // 2
        offset_H = (img_size - H) // 2
        offset_W = (img_size - W) // 2
        guidance_pad[
            :, offset_D : offset_D + D, offset_H : offset_H + H, offset_W : offset_W + W
        ] = guidance
        noise = torch.randn((1, img_size, img_size, img_size), device=self.device).to(
            self.dtype
        )
        half_size = self.config.data.slice_thickness // 2

        image = noise
        iterator = (
            tqdm(self.scheduler.timesteps, ncols=70)
            if self.verbose
            else self.scheduler.timesteps
        )
        shared_latent = None
        for i, t in enumerate(iterator):
            if i and self.layz_sampling_step > 0 and t > self.layz_sampling_step:
                continue  # skip all steps until lazy_step
            if self.nex > 1 and t < self.mex_diversion_step:
                shared_latent = image
                break

            t_tens = torch.Tensor((t,)).long().to(self.device)
            view = list(self.models.keys())[t % len(self.models.keys())]
            t_prev = (  # next time step for manually noising unpredicted padding areas
                self.scheduler.timesteps[i + 1]
                if i < len(self.scheduler.timesteps) - 1
                else 0
            )

            image = self.volume_diffusion_step(
                (C, D, H, W),
                batch_size,
                guidance_pad,
                half_size,
                image,
                img_size,
                (offset_D, offset_H, offset_W),
                t,
                t_prev,
                t_tens,
                view,
            )

        if shared_latent is not None:
            # average multiple trajectories towards the end (simulate multi exitations)
            if self.verbose:
                print("Simulating multiple exitations from t = ", i)
            excitations = []
            timesteps = self.scheduler.timesteps[i:]

            iter = tqdm(range(self.nex)) if self.verbose else range(self.nex)
            for _ in iter:
                if self.gt_image is not None:
                    self.psnr.append(-1)
                    self.ssim.append(-1)
                image = shared_latent.clone()
                for j, t in enumerate(timesteps):
                    t_tens = torch.Tensor((t,)).long().to(self.device)
                    view = list(self.models.keys())[t % len(self.models.keys())]
                    t_prev = (  # next time step for manually noising unpredicted padding areas
                        timesteps[j + 1] if j < len(timesteps) - 1 else 0
                    )
                    image = self.volume_diffusion_step(
                        (C, D, H, W),
                        batch_size,
                        guidance_pad,
                        half_size,
                        image,
                        img_size,
                        (offset_D, offset_H, offset_W),
                        t,
                        t_prev,
                        t_tens,
                        view,
                        eta=1 if isinstance(self.scheduler, DDIMScheduler) else 0,
                    )
                excitations.append(image)
            image = self.mex_method(excitations)

        # crop image to original size
        image = image[
            :, offset_D : offset_D + D, offset_H : offset_H + H, offset_W : offset_W + W
        ]
        if self.nex <= 1:
            return image.cpu()
        else:
            excitations_cpu = []
            for echo in excitations:
                echo = echo[
                    :,
                    offset_D : offset_D + D,
                    offset_H : offset_H + H,
                    offset_W : offset_W + W,
                ]
                excitations_cpu.append(echo.cpu())
            return image.cpu(), excitations_cpu

    def volume_diffusion_step(
        self,
        real_shape,
        batch_size,
        guidance_pad,
        half_size,
        image,
        img_size,
        offsets,
        t,
        t_prev,
        t_tens,
        view,
        eta=0,
    ):
        (C, D, H, W) = real_shape
        (offset_D, offset_H, offset_W) = offsets
        if view == "axial":
            guid_t = guidance_pad
            image_t = image
            lower, upper = offset_D, offset_D + D
        if view == "coronal":
            guid_t = guidance_pad.permute(0, 2, 1, 3)
            image_t = image.permute(0, 2, 1, 3)
            lower, upper = offset_H, offset_H + H
        elif view == "sagittal":
            guid_t = guidance_pad.permute(0, 3, 1, 2)
            image_t = image.permute(0, 3, 1, 2)
            lower, upper = offset_W, offset_W + W
        pred_xprev = []
        for i in range(lower, upper, batch_size):  # for each slice
            end_slice = min(i + batch_size, upper)  # account for truncated batch
            batch = []
            for j in range(i, end_slice):  # for each target slize within the batch
                start_slice_in = j - half_size  # bounds of input slices
                end_slice_in = j + half_size + 1  # bounds of input slices

                # padding of guidance slices (multi-slice input case)
                pad_start = max(-start_slice_in, 0)
                start_slice_in = max(start_slice_in, 0)
                pad_end = max(end_slice_in - img_size, 0)
                end_slice_in = min(end_slice_in, img_size)

                black_start = -torch.ones(
                    (pad_start, img_size, img_size),
                    device=image_t.device,
                    dtype=self.dtype,
                )
                black_end = -torch.ones(
                    (pad_end, img_size, img_size),
                    device=image_t.device,
                    dtype=self.dtype,
                )

                x = [black_start, image_t[0, start_slice_in:end_slice_in], black_end]
                for c in range(C):
                    x += [
                        black_start,
                        guid_t[c, start_slice_in:end_slice_in],
                        black_end,
                    ]
                batch.append(torch.cat(x, dim=0))
            batch = torch.stack(batch)  # N C H W, where C = [img, C1, C2, ...]
            with torch.autocast(dtype=self.dtype, device_type=self.device):
                model_output = self.models[view](
                    batch.contiguous(),
                    t_tens,
                    class_labels=torch.zeros_like(t_tens).long() + self.motion_score,
                )
                # as N C H W
            pred_xprev.append(model_output.transpose(0, 1))
        pred_xprev = torch.cat(pred_xprev, dim=1)
        if eta == 0:  # circumvent issues with eta for DDPM
            pred_xprev, pred_x0 = self.scheduler.step(
                pred_xprev.float(), t, image_t[:, lower:upper].float()
            )
        else:
            pred_xprev, pred_x0 = self.scheduler.step(
                pred_xprev.float(), t, image_t[:, lower:upper].float(), eta
            )
        if self.layz_sampling_step > 0 and t > self.layz_sampling_step:
            t_prev = self.layz_sampling_step  # skip all steps until lazy_step
        sqrt_alpha_prod = self.scheduler.alphas_cumprod[t_prev] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[t_prev]) ** 0.5
        noise = torch.randn_like(image)

        if self.layz_sampling_step > 0 and t > self.layz_sampling_step:
            pred_xprev = (
                pred_x0 * sqrt_alpha_prod
                + noise[:, lower:upper] * sqrt_one_minus_alpha_prod
            )

        # simulate next timestep given that pad area is all -1 (black / pad)
        image = (-1) * sqrt_alpha_prod + noise * sqrt_one_minus_alpha_prod
        image[:, lower:upper] = pred_xprev  # replace with actual prediction
        # re-transpose to C D H W
        if view == "coronal":
            image = image.permute(0, 2, 1, 3)
        elif view == "sagittal":
            image = image.permute(0, 2, 3, 1)
        if self.gt_image is not None:
            tmp_pred = pred_x0
            if view == "axial":
                tmp_pred = tmp_pred[
                    :, :, offset_H : offset_H + H, offset_W : offset_W + W
                ]
            elif view == "coronal":
                tmp_pred = tmp_pred[
                    :, :, offset_D : offset_D + D, offset_W : offset_W + W
                ]
                tmp_pred = tmp_pred.permute(0, 2, 1, 3)
            elif view == "sagittal":
                tmp_pred = tmp_pred[
                    :, :, offset_D : offset_D + D, offset_H : offset_H + H
                ]
                tmp_pred = tmp_pred.permute(0, 2, 3, 1)

            psnr = torch.mean(
                20
                * torch.log10(
                    2 / torch.sqrt(torch.mean((tmp_pred - self.gt_image) ** 2))
                )
            ).item()
            ssim = SSIMMetric(spatial_dims=3)(
                tmp_pred.unsqueeze(dim=0), self.gt_image.unsqueeze(dim=0)
            ).item()
            self.psnr.append(psnr)
            self.ssim.append(ssim)
        return image
