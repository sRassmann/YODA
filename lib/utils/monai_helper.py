import os
from typing import Any, Hashable, Mapping, Optional, Sequence, Union, List, Dict, Tuple
import numpy as np
import torch
from monai import transforms
from monai.config import KeysCollection
from monai.data import CacheDataset, Dataset, PersistentDataset
from generative.networks.schedulers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
)
from monai.transforms import MapTransform, Randomizable
from glob import glob

from generative.networks.schedulers import NoiseSchedules
from lib.utils.etc import print0


def append_datadir(files, datadir="../data/RS/conformed"):
    """Append the datadir to the paths in the files dict."""

    def is_path(value):
        if not isinstance(value, str):
            return False
        return ".nii.gz" in value or ".nii" in value or ".mgz" in value

    for i, file in enumerate(files):
        for key, value in file.items():
            if isinstance(value, list):
                if is_path(value[0]):
                    files[i][key] = [
                        os.path.join(os.path.abspath(datadir), p) for p in value
                    ]
            else:
                if is_path(value):
                    files[i][key] = os.path.join(os.path.abspath(datadir), value)
    return files


class RandomChannelSelectiond(MapTransform, Randomizable):
    """Randomly select one channel from the given sequence of images."""

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        Randomizable.__init__(self)
        self._channel = 0

    def randomize(self, data: Union[torch.Tensor, np.ndarray]) -> None:
        self._channel = np.random.randint(data.shape[0])

    def __call__(
        self, data: Mapping[Hashable, Union[torch.Tensor, np.ndarray]]
    ) -> Mapping[Hashable, Union[torch.Tensor, np.ndarray]]:
        d = dict(data)
        for key in self.keys:
            self.randomize(d[key])
            d[key] = d[key][self._channel : self._channel + 1]
        return d


class RandomMaskIntensityd(MapTransform, Randomizable):
    """Randomly select one channel from the given sequence of images."""

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        prob: float = 0.0,
        mask_key: str = "brain_mask",
        masked_intensity: float = -1,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        Randomizable.__init__(self)
        self.mask_key = mask_key
        self.prob = prob
        self._apply = False
        self.masked_intensity = masked_intensity

    def randomize(self, data=None) -> None:
        self._apply = np.random.rand() < self.prob

    def __call__(
        self, data: Mapping[Hashable, Union[torch.Tensor, np.ndarray]]
    ) -> Mapping[Hashable, Union[torch.Tensor, np.ndarray]]:
        d = dict(data)
        mask = d[self.mask_key] > 0
        self.randomize(None)
        if self._apply:
            for key in self.keys:
                if key != self.mask_key:
                    d[key] = d[key] * mask + ~mask * self.masked_intensity
        return d


def remove_channel_dim(x):
    return x.squeeze(0)


def is_greater_0(x):
    return x > 0


def parse_dtype(dtype):
    return getattr(torch, dtype)


def create_maybe_cached_dataset(
    cache: str,
    num_workers: int,
    ds: Dict,
    applied_transforms: transforms.Compose,
    persistent_cache_path="../data/monai_cache",
) -> Dataset:
    """Wrapper to create a dataset using different caching strategies."""
    if cache is None or cache == "none" or not cache:
        print0("Using non-cached datasets, this will be slow for training.")
        cache_ds = Dataset(data=ds, transform=applied_transforms)

    elif cache in ["preload", "runtime", "processes"]:
        print0(f"Using cached datasets ({cache}).")
        if cache == "preload":
            cache = False
        if cache == "runtime":
            cache = True
        cache_ds = CacheDataset(
            data=ds,
            transform=applied_transforms,
            runtime_cache=cache,  # cache only on actual request
            num_workers=num_workers,
            copy_cache=False,
        )

    elif cache == "persistent":  # about 40 GB for RS
        cache_path = maybe_local_cache(persistent_cache_path)
        print0(f"Using persistent cached datasets at {cache_path}")
        cache_ds = PersistentDataset(
            data=ds,
            transform=applied_transforms,
            cache_dir=cache_path,
        )

    else:
        raise ValueError(f"Unknown cache option {cache}")
    return cache_ds


def maybe_local_cache(
    path,
    local_cache_path="/localmount/*/users/rassmanns/data",
    cached_dir_name="data",
):
    """
    search if data is available locally, if so, return the local path, otherwise
    return the original path
    """
    if hpc_cache_path := os.getenv("HPCWORK", ""):
        local_cache_path = os.path.join(hpc_cache_path, "data")

    # search for cached_dir_name in the path

    if cached_dir_name not in os.path.split(path):
        print0(
            f"Using original path as cache dir not {cached_dir_name} not found in data path {path}"
        )
        return path
    subpath = path.split(cached_dir_name)[1]
    search_path = os.path.join(
        os.path.dirname(local_cache_path), cached_dir_name, subpath[1:]
    )
    if local_path := glob(search_path):
        assert len(local_path) == 1, "More than one local path found"
        print0(
            f"Using local path {local_path[0]} (found {len(os.listdir(local_path[0]))} files)"
        )
        return local_path[0]
    else:
        print0(f"No local data found at {search_path}, using original path")
        return path


def set_timesteps(self, num_inference_steps, device=None) -> None:
    """
    Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

    Args:
        num_inference_steps: number of diffusion steps used when generating samples with a pre-trained model.
        device: target device to put the data.
    """
    if num_inference_steps > self.num_train_timesteps:
        raise ValueError(
            f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.num_train_timesteps`:"
            f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
            f" maximal {self.num_train_timesteps} timesteps."
        )

    self.num_inference_steps = num_inference_steps
    if num_inference_steps > 1:
        timesteps = (
            np.linspace(0, self.num_train_timesteps - 1, num_inference_steps)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
    else:
        timesteps = np.array([self.num_train_timesteps - 1], dtype=np.int64).copy()
    self.timesteps = torch.from_numpy(timesteps).to(device)
    self.timesteps += self.steps_offset


def betas_for_alpha_bar(
    num_diffusion_timesteps, alpha_bar, max_beta=0.999
):  # https://github.com/openai/improved-diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/gaussian_diffusion.py#L45C1-L62C27
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


@NoiseSchedules.add_def("cosine_poly", "Cosine schedule")
def _cosine_beta(num_train_timesteps: int, s: float = 8e-3, order: float = 2, *args):
    return betas_for_alpha_bar(
        num_train_timesteps,
        lambda t: np.cos((t + s) / (1 + s) * np.pi / 2) ** order,
    )


def scheduler_factory(scheduler_config):
    """helper function to create scheduler from omegaconf"""
    # default: DDPM
    if "type" not in scheduler_config:
        return DDPMScheduler(**scheduler_config)

    sched_type = scheduler_config.pop("type")  # remove type from dict

    if sched_type == "DDPM":
        sched = DDPMScheduler(**scheduler_config)
        if scheduler_config.num_train_timesteps == 1:
            print0(
                "Training with single timestep, scheduler is modified to simulate direct non-diffusive regression"
            )
            sched.alphas[0] = 0
            sched.alphas_cumprod[0] = 0
            sched.betas[0] = 1
        return sched
    elif sched_type == "DDIM":
        time_spacing_method = None
        if scheduler_config.get("timestep_spacing"):
            time_spacing_method = scheduler_config.pop("timestep_spacing")
        sched = DDIMScheduler(**scheduler_config)

        if time_spacing_method == "linspace":
            # replace set_timesteps method with custom one
            sched.set_timesteps = set_timesteps.__get__(sched)
        return sched
    elif sched_type == "PNDM":
        return PNDMScheduler(**scheduler_config)
    else:
        raise ValueError(f"Unknown scheduler type {scheduler_config.type}")
