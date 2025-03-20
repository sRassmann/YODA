import os
import time
import torch.distributed as dist
import torch
from omegaconf import DictConfig, OmegaConf
import subprocess

from lib.utils.etc import print0, is_rank_0


def add_arguments_from_config(parser, config, parent_key=""):
    # Recursively add arguments from the config to the parser
    for key, value in config.items():
        if isinstance(value, DictConfig):
            # For nested dictionaries, add their arguments recursively
            add_arguments_from_config(
                parser,
                value,
                parent_key=f"{parent_key}.{key}" if parent_key else key,
            )
        else:
            # For non-dictionary values, add them as command-line arguments
            arg_key = f"--{parent_key}.{key}" if parent_key else f"--{key}"
            parser.add_argument(arg_key, type=type(value), default=None)


def merge_cli_config(args, default_config="configs/defaults.yml"):
    merged_config = OmegaConf.load(default_config)

    for path in args.configs:
        config = OmegaConf.load(path)
        merged_config = OmegaConf.merge(merged_config, config)

    # overwrite with CLI args
    for key, value in vars(args).items():
        if "." not in key:  # ignore first-level args
            continue
        if value is not None:
            OmegaConf.update(merged_config, key, value, merge=False)

    return merged_config


def link_arguments(config):
    """hard-coded updates to assert that the config is correct"""
    if len(config.data.guidance_sequences) + 1 != config.model.unet.in_channels:
        print0(f"Updating in_channels to {len(config.data.guidance_sequences) + 1}")
        config.model.unet.in_channels = len(config.data.guidance_sequences) + 1
    slice_thickness = config.data.get("slice_thickness", 1)
    if slice_thickness > 1:
        config.model.unet.in_channels *= slice_thickness
        print0(
            f"Working with thick slices. Updating model to {config.model.unet.in_channels} in_channels"
        )
        thick_target = config.model.get("thick_target", False)
        if thick_target:
            config.model.unet.out_channels = slice_thickness
            print0(
                f"Working with tick target. Updating out_channels to {config.model.unet.out_channels}"
            )
    return config


def create_run_dirs(run_name, config):
    run_path, sample_path, ckpt_path, tb_path = None, None, None, None
    run_path = f"output/{run_name}"
    if os.path.exists(run_path):
        run_path = (
            f"output/{run_name}_{time.strftime('%y-%m-%d_%H:%M:%S', time.localtime())}"
        )
    sample_path = os.path.join(run_path, "val_samples")
    ckpt_path = os.path.join(run_path, "ckpt")
    tb_path = os.path.join(run_path, "tb")
    if is_rank_0():
        os.makedirs(sample_path, exist_ok=True)
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(tb_path, exist_ok=True)
        print(f"Saving run to {run_path}.")
        print("Configuration used:")
        print(OmegaConf.to_yaml(config))

        # append the current git commit hash
        process = subprocess.Popen(
            ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
        )
        git_head_hash = process.communicate()[0].strip()

        # write yml to text and comment git hash above
        with open(os.path.join(run_path, "config.yml"), "w") as f:
            f.write(f"# git head hash: {git_head_hash.decode('utf-8')}\n")
            f.write(OmegaConf.to_yaml(config))

    return run_path, sample_path, ckpt_path, tb_path


def setup_ddp():
    rank = 0
    world_size = 1
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
    torch.cuda.set_device(rank)

    return rank, world_size
