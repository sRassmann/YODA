import warnings

import sys, os
from omegaconf import OmegaConf

sys.path.append(os.path.realpath(os.getcwd()))

import argparse
import torch

from lib.datasets import get_datasets
from lib.inference import (
    Sr3MultiViewInferenceModel,
    save_output_volume,
)
from lib.utils.run_utils import merge_cli_config
from lib.utils.etc import remove_module_in_state_dict, seed_from_subject_id
from lib.utils.monai_helper import parse_dtype, scheduler_factory

device = "cuda"


def main(
    run_name,
    coronal_run_name,
    sagittal_run_name,
    output_dir,
    ckpt_name,
    config,
    subject_indices,
    seed_offset=0,
    quantile_norm=False,
    tqdm_verbose=True,
    check_if_exists=False,
    save_echos=True,
    n_excitations=1,
    nex_divert_step=250,
    lazy_sampling_step=250,
    calc_scores=False,
):
    dtype = parse_dtype(config.trainer.dtype)

    cor_config = (
        OmegaConf.load(os.path.join("output", coronal_run_name, "config.yml"))
        if coronal_run_name
        else None
    )
    sag_config = (
        OmegaConf.load(os.path.join("output", sagittal_run_name, "config.yml"))
        if sagittal_run_name
        else None
    )

    cor_run = (
        os.path.join("output", coronal_run_name, "ckpt", ckpt_name)
        if coronal_run_name
        else None
    )
    sag_run = (
        os.path.join("output", sagittal_run_name, "ckpt", ckpt_name)
        if sagittal_run_name
        else None
    )

    inf_model = Sr3MultiViewInferenceModel(
        config,  # axial config --> determines the scheduler
        cor_config,
        sag_config,
        os.path.join("output", run_name, "ckpt", ckpt_name),
        cor_run,
        sag_run,
        device,
        dtype,
        verbose=tqdm_verbose,
        n_exitations=n_excitations,
        mex_step=nex_divert_step,
        lazy_sampling_step=lazy_sampling_step,
    )

    skull_strip = config.data.get("skull_strip", False)
    if skull_strip != 1 and skull_strip != 0:
        warnings.warn(
            f"Skull stripping should be either be on (1) or off (0) for the whole dataset, but is {skull_strip}. Rounding to {round(skull_strip)}."
        )
        skull_strip = round(skull_strip)

    tar = [config.data.target_sequence] if config.data.target_sequence else []
    _, val = get_datasets(
        dataset=config.data.dataset,
        data_dir=config.data.data_dir,
        relevant_sequences=config.data.guidance_sequences + tar,
        size=(max(config.data.img_size) - config.data.slice_thickness + 1,) * 3,
        cache=None,  # if config.data.cache != "persistent" else "persistent",
        subset_train=0,
        normalize_to=(-1, 1),
        skull_strip=skull_strip,
        slicing_direction="axial",  # always load as SPL (axial)
    )

    for subject_index in subject_indices:
        if subject_index >= len(val):
            break

        vol = val[subject_index]
        if check_if_exists and vol["subject_ID"] in os.listdir(output_dir):
            print(
                f"Subject {vol['subject_ID']} already exists in {output_dir}. Skipping."
            )
            continue
        print(f"Saving subject {vol['subject_ID']} to {output_dir}.")

        # seed from subject ID for reproducibility across scripts / runs
        hash = seed_from_subject_id(vol["subject_ID"])

        guidance = torch.cat(  # C D H W
            [vol[seq] for seq in config.data.guidance_sequences], dim=0
        ).astype(dtype)

        if calc_scores:
            inf_model.gt_image = vol[config.data.target_sequence].cuda()
            inf_model.psnr = []
            inf_model.ssim = []

        torch.manual_seed(hash + seed_offset)
        print(f"start predicting {vol['subject_ID']}")
        pred = inf_model.synchronous_volume_denoising(guidance)
        echo_keys = []
        if isinstance(pred, tuple):
            vol["pred"] = pred[0]
            if save_echos:
                for i, echo in enumerate(pred[1]):
                    vol[f"pred_echo_{i}"] = echo
                    echo_keys.append(f"pred_echo_{i}")
        else:
            vol["pred"] = pred

        save_output_volume(
            vol,
            output_dir,
            save_keys=list(config.data.guidance_sequences)
            + echo_keys
            + ["pred", config.data.target_sequence, "mask"],
            target_sequence=config.data.target_sequence,
            quantile_norm=quantile_norm,
        )

        if calc_scores:
            path = os.path.join(output_dir, vol["subject_ID"], "dif_metrics.csv")

            import pandas as pd

            df = pd.DataFrame({"psnr": inf_model.psnr, "ssim": inf_model.ssim})
            df.to_csv(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running inference.")

    # model configs
    parser.add_argument(
        "--run_name",
        "-r",
        type=str,
        help="Name of the axial run (parent directory). The model checkpoint, config file, and the output will be taken from / saved to this directory.",
    )
    parser.add_argument(
        "--coronal_run_name",
        "-cor",
        type=str,
        help="Name of the coronal run (parent directory).",
    )
    parser.add_argument(
        "--sagittal_run_name",
        "-sag",
        type=str,
        help="Name of the sagittal run (parent directory).",
    )

    # slicing direction
    parser.add_argument(
        "--slicing_direction",
        "-sd",
        type=str,
        default=None,
        help="Slicing direction for the input data (default: from model config, usually axial).",
    )

    parser.add_argument(
        "--guidance_sequences",
        "-guid",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--target_sequence",
        "-tar",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_json",
        "-dj",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data_dir",
        "-dd",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--out_name",
        "-o",
        default="inference_multi_view",
        type=str,
        help="Name to the output directory.",
    )
    parser.add_argument(
        "--dtype",
        "-d",
        type=str,
        default=None,
        help="Data type for inference (if none provided the train config will be used).",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "configs",
        nargs="*",
        type=str,
        help="Paths to configuration files, overwrites the model config from <run_name>/config.yml.",
    )
    parser.add_argument(
        "--steps", "-t", type=int, default=None, help="Number of steps."
    )
    parser.add_argument(
        "--number_of_excitations",
        "-nex",
        type=int,
        default=1,
        help="Number of excitation drawn for the diffusion process.",
    )
    parser.add_argument(
        "--mex_divert_step",
        "-mexds",
        type=int,
        default=100,
        help="Number of steps before diverting to the next excitation.",
    )
    parser.add_argument(
        "--lazy_sampling_step",
        "-lazy",
        default=0,
        type=int,
        help="Whether to use lazy sampling for the diffusion process, if set > 0 "
        "the diffusion process will skip from the last (T) diffusion step to the "
        "step defined by the value of this argument.",
    )

    # define checkpoint
    parser.add_argument(
        "--ckpt",
        "-c",
        type=str,
        default="last.pth",
        help="Path to the checkpoint file. If not provided, the last checkpoint from <run_name>/ckpt will be used.",
    )

    # Optional positional argument for start and end indices
    parser.add_argument(
        "--start", "-s", type=int, default=0, help="Start index for the validation set."
    )
    parser.add_argument(
        "--end", "-e", type=int, default=10000, help="End index for the validation set."
    )

    # tqdm verbosity
    parser.add_argument(
        "--no_verbose",
        "-nv",
        action="store_true",
    )

    # check if output already exists
    parser.add_argument(
        "--check_if_exists",
        "-ce",
        action="store_true",
    )

    # calculate scores during prediction process
    parser.add_argument(
        "--calc_scores",
        "-cs",
        action="store_true",
    )

    # optional_seed
    parser.add_argument(
        "--seed",
        "-seed",
        type=int,
        default=0,
        help="Change seed for the diffusion process to assess sampling stability/diversity.",
    )

    # quantile norm
    parser.add_argument(
        "--quantile_norm",
        "-q",
        type=bool,
        default=False,
        help="Whether to normalize the output by quantiles.",
    )
    parser.add_argument(  # see https://github.com/Project-MONAI/GenerativeModels/issues/397#issuecomment-1954514581
        "--timestep_spacing",
        "-ts",
        type=str,
        default=None,
        help="Timestep spacing (default: leading, option: linspace ie. forcing start at t=T) ",
    )

    parser.add_argument(
        "-f", "--force", action="store_true", help="Use the force, Luke."
    )

    args = parser.parse_args()

    run_name = args.run_name
    config = merge_cli_config(args, default_config=f"output/{run_name}/config.yml")

    # overwrite config with manual args
    if args.dtype is not None:
        config.trainer.dtype = args.dtype
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.steps is not None:
        config.model.num_inference_steps = args.steps
    if args.timestep_spacing is not None:
        config.model.train_noise_sched.timestep_spacing = args.timestep_spacing
    if args.slicing_direction is not None:
        config.data.slicing_direction = args.slicing_direction
    if args.guidance_sequences is not None:
        config.data.guidance_sequences = args.guidance_sequences
    if args.target_sequence is not None:
        config.data.target_sequence = args.target_sequence
    if args.dataset_json is not None:
        config.data.dataset = args.dataset_json
    if args.data_dir is not None:
        config.data.data_dir = args.data_dir

    if args.force:
        print("YODA predicting your images is")
        print("Patience you must have, my young Padawan!")
        print("\n" + mini_yoda + "\n")

    print(
        f"predicting {config.data.target_sequence} from {config.data.guidance_sequences}"
    )

    # create output directory
    output_dir = os.path.join("output", run_name, args.out_name)
    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists.")
    os.makedirs(output_dir, exist_ok=True)

    subject_indices = list(range(args.start, args.end))

    main(
        args.run_name,
        args.coronal_run_name,
        args.sagittal_run_name,
        output_dir,
        args.ckpt,
        config,
        subject_indices,
        args.seed,
        args.quantile_norm,
        not args.no_verbose,
        args.check_if_exists,
        n_excitations=args.number_of_excitations,
        nex_divert_step=args.mex_divert_step,
        calc_scores=args.calc_scores,
        lazy_sampling_step=args.lazy_sampling_step,
    )

    if not args.no_verbose:
        print(f"finished inference for {args.run_name}")
        print(f"output saved to {output_dir}")
        print(f"thanks for using YODA!")
    if args.force:
        print("\n" + the_force + "\n")
