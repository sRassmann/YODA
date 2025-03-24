import os
import warnings

from typing import Any, Hashable, Mapping, Optional, Sequence, Union, List, Dict, Tuple

import torch
from monai import transforms
from monai.data import Dataset, MetaTensor, partition_dataset
import numpy as np

from torch.utils.data import DataLoader
import torch.distributed as dist
import nibabel as nib
import json

from lib.utils.monai_helper import (
    append_datadir,
    RandomChannelSelectiond,
    RandomMaskIntensityd,
    create_maybe_cached_dataset,
    maybe_local_cache,
    remove_channel_dim,
    is_greater_0,
)
from lib.utils.etc import close_mask, erode


def cacheable_transforms(
    relevant_sequences,
    crop_to_brain_mask_margin=(5, 10, 20),
    normalize_to=(-1, 1),
):
    t = [
        # remove unnecessary keys and load the remaining images
        transforms.SelectItemsd(
            keys=relevant_sequences + ["subject_ID"],
            allow_missing_keys=True,
        ),
        transforms.LoadImaged(
            keys=relevant_sequences,
            ensure_channel_first=True,
            dtype=torch.uint8,
            image_only=False,
        ),
        # put in into standard orientation i.e. C D W H
        transforms.Orientationd(keys=relevant_sequences, axcodes="SPL"),
    ]
    if crop_to_brain_mask_margin is not None:
        t.append(
            # crop to brain
            transforms.CropForegroundd(
                keys=relevant_sequences,
                source_key="mask",
                allow_smaller=True,
                margin=crop_to_brain_mask_margin,  # as D H W
            )
        )
    t.append(transforms.CopyItemsd(keys=["mask"], names=["brain_mask"]))
    t.append(transforms.Lambdad(keys=["brain_mask"], func=close_mask))
    # normalize, note that this uses the original image, could be an issue when loading the skull-stripped and renormalized images
    t.append(
        transforms.ScaleIntensityRanged(
            keys=set(relevant_sequences) - {"mask"},
            a_min=0.0,
            a_max=255.0,  # assumed to be 8-bit
            b_min=normalize_to[0],
            b_max=normalize_to[1],
            clip=True,
            dtype=torch.float16,
        )
    )
    return t


def cache_interrupt_transforms(relevant_sequences):
    float_sequences = [seq for seq in relevant_sequences if seq != "mask"]
    t = [RandomChannelSelectiond(keys=float_sequences)]
    return t


def get_2d_sample_function(
    aug_transforms, pad_val, relevant_sequences, size, skull_strip_p=0.0
):
    if len(size) == 2:
        slice_thickness = 1
    elif len(size) == 3:
        slice_thickness = size[0]
        size = size[1:]
    else:
        raise ValueError(f"Invalid size {size}")
    sample = [
        # Now, we have 1 D W H
        transforms.RandSpatialCropd(
            keys=relevant_sequences + ["brain_mask"],
            roi_size=[slice_thickness, -1, -1],  # dynamic spatial_size for W and H
            random_size=False,
        ),
        RandomMaskIntensityd(
            keys=relevant_sequences,
            mask_key="brain_mask",
            prob=skull_strip_p,
            masked_intensity=pad_val,
        ),
        transforms.DeleteItemsd(keys=["brain_mask"]),
        transforms.Lambdad(keys=relevant_sequences, func=remove_channel_dim),
        aug_transforms,
        # Pad or crop to fixed size
        transforms.ResizeWithPadOrCropd(
            keys=[seq for seq in relevant_sequences if seq != "mask"],
            spatial_size=size,
            value=pad_val,
        ),
        transforms.ResizeWithPadOrCropd(
            keys=["mask"],
            spatial_size=size,
            value=0,
        ),
        transforms.ToTensord(keys=relevant_sequences),
    ]
    return sample


def get_datasets(
    dataset: str = "../data/RS/RS_train_split.json",
    data_dir: str = "../data/RS/conformed_mask_reg",
    slicing_direction: str = "axial",
    random_slicing_direction: bool = False,
    relevant_sequences: Union[str, List[str]] = ["flair", "t1", "t2"],
    size: Union[Tuple[int, int], Tuple[int, int, int], None] = (224, 224),
    slice_thickness: Optional[int] = 1,
    normalize_to: Tuple[float, float] = (-1, 1),
    crop_to_brain_margin: Union[Tuple[int, int, int], None] = (5, 10, 20),
    skull_strip: float = 0.0,
    aug_transforms: transforms.Compose = None,
    num_workers: int = 8,
    cache: Union[str, None] = "persistent",
    subset_train: Optional[int] = None,
    subset_val: Optional[int] = None,
    ddp_split: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Create train and validation datasets in 2D or 3D.

    Args:
        dataset: Path to the dataset json file, defines the train/val split.
            Assumed to contain the `target_sequence` and all `guidance_sequence` keys, a
            `mask` key, and the `subject_ID` key.
        data_dir: Path to the data directory (relative to the dataset json file).
        slicing_direction: The direction in which the slices will be sampled. Can be
            "axial", "sagittal", or "coronal".
        random_slicing_direction: If True, the slicing direction will be randomly
            selected from the three possible directions.
        relevant_sequences: The sequence(s) to be loaded for reconstruction. Note, that
            caching irrelevant sequences would be unnecessarily slow.
        size: The size of the input patches, can be length 2 (H W) or 3 (D H W) for 2D
            or 3D, respectively. If None, the whole image (cropped to mask if specified)
            will be used.
        slice_thickness: The slice thickness in case of 2D sampling. If None,
             a single slice (slice thickness 1) will be sampled.
        normalize_to: The interval to which the input patches will be normalized.
        crop_to_brain_margin: Margin for the brain mask cropping in voxels as
            D H W (SPL). If None, no cropping will be performed.
        skull_strip: Probability of skull stripping (i.e. masking out the background).
        aug_transforms: Augmentation transforms to be applied to the input patches,
            could be affine, noise, bias field, blur, saturation, etc.
        num_workers: Number of workers for caching the Dataset.
        cache: If "preload", the whole dataset will be cached in memory. If "runtime",
            the dataset will be cached on the fly. If None, no caching will be
            performed.
        subset_train: If not None, only use the first `subset_train` subjects for
            training.
        subset_val: If not None, only use the first `subset_val` subjects for
            validation.
        ddp_split: If True, the dataset will be split among the DDP processes. If False,
            each process will use the whole dataset.
    """
    assert slice_thickness % 2, "Slice thickness must be odd"
    # crop_to_brain_margin = (5, 10, 0)  # SPL, ie. Axial, Coronal, Sagittal
    if slice_thickness and slice_thickness > 1:
        size = (slice_thickness,) + size
        # sample additional slices on top and bottom
        if crop_to_brain_margin is not None:
            depth = crop_to_brain_margin[0] + slice_thickness // 2
            crop_to_brain_margin = (depth,) + tuple(crop_to_brain_margin[1:])

    data_dir = maybe_local_cache(data_dir)
    preprocess = cacheable_transforms(
        relevant_sequences=relevant_sequences + ["mask"],
        crop_to_brain_mask_margin=crop_to_brain_margin,
        normalize_to=normalize_to,
    ) + cache_interrupt_transforms(relevant_sequences=relevant_sequences + ["mask"])

    # Drop unused sequences (if in cache)
    drop_sequences = {"flair", "t1", "t2"} - set(relevant_sequences)
    if drop_sequences:
        preprocess.append(transforms.DeleteItemsd(keys=drop_sequences))

    orient_aug = orientation_augs(
        random_slicing_direction, relevant_sequences, slicing_direction
    )
    if orient_aug is not None:
        preprocess.append(orient_aug)

    if aug_transforms is None:
        aug_transforms = transforms.Identityd(keys=relevant_sequences)

    sample_func = (
        get_2d_sample_function
        if (size is not None) and (len(size) == 2 or slice_thickness > 1)
        else get_3d_sample_function
    )

    train_sample = sample_func(
        aug_transforms,
        normalize_to[0],
        relevant_sequences + ["mask"],
        size,
        skull_strip,
    )
    val_sample = sample_func(
        aug_transforms,
        normalize_to[0],  # pad with min value
        relevant_sequences + ["mask"],
        size,
        skull_strip,
    )

    train, val = create_data_dicts(
        data_dir, dataset, ddp_split, subset_train, subset_val
    )

    persistent_cache_path = "../data/monai_cache"
    train = create_maybe_cached_dataset(
        cache,
        num_workers,
        train,
        transforms.Compose(preprocess + train_sample),
        persistent_cache_path=persistent_cache_path,
    )
    val = create_maybe_cached_dataset(
        cache,
        num_workers,
        val,
        transforms.Compose(preprocess + val_sample),
        persistent_cache_path=persistent_cache_path,
    )
    return train, val


def orientation_augs(random_slicing_direction, relevant_sequences, slicing_direction):
    if random_slicing_direction:
        # random slicing direction change
        return transforms.OneOf(
            [
                transforms.Orientationd(
                    keys=relevant_sequences + ["mask", "brain_mask"],
                    axcodes=slicing_axcodes["coronal"],
                ),
                transforms.Orientationd(
                    keys=relevant_sequences + ["mask", "brain_mask"],
                    axcodes=slicing_axcodes["sagittal"],
                ),
                # Noop for axial
                transforms.Identityd(keys=relevant_sequences),
            ],
            weights=[0.33, 0.33, 0.33],
        )

    elif slicing_direction != "axial":
        # adjust slicing direction
        return transforms.Orientationd(
            keys=relevant_sequences + ["mask", "brain_mask"],
            axcodes=slicing_axcodes[slicing_direction],
            allow_missing_keys=True,
        )
    return None


def get_3d_sample_function(
    aug_transforms, pad_val, relevant_sequences, size, skull_strip_p=0.0
):
    sample = [
        aug_transforms,
        RandomMaskIntensityd(
            keys=[seq for seq in relevant_sequences if seq != "mask"],
            mask_key="brain_mask",
            prob=skull_strip_p,
            masked_intensity=pad_val,
            allow_missing_keys=True,
        ),
    ]
    if size is not None:
        sample += [
            transforms.DeleteItemsd(keys=["brain_mask"]),
            transforms.ResizeWithPadOrCropd(
                keys=[seq for seq in relevant_sequences if seq != "mask"],
                spatial_size=size,
                value=pad_val,
            ),
            transforms.ResizeWithPadOrCropd(
                keys=["mask"],
                spatial_size=size,
                value=0,
                allow_missing_keys=True,
            ),
        ]
    return sample


slicing_axcodes = {
    "axial": "SPL",
    "sagittal": "LSP",
    "coronal": "PSL",
}


def create_data_dicts(data_dir, dataset, ddp_split, subset_train, subset_val):
    with open(dataset, "r") as f:
        train = json.load(f)["training"]
    if subset_train is not None:
        train = train[:subset_train]
    train = append_datadir(train, data_dir)
    if ddp_split and dist.is_initialized() and len(train):
        train = partition_dataset(
            data=train,
            num_partitions=dist.get_world_size(),
            shuffle=True,
            seed=0,
            drop_last=True,
            even_divisible=True,
        )[dist.get_rank()]
    with open(dataset, "r") as f:
        val = json.load(f)["validation"]
    if subset_val is not None:
        val = val[:subset_val]
    val = append_datadir(val, data_dir)
    if ddp_split and dist.is_initialized() and len(val):
        val = partition_dataset(
            data=val,
            num_partitions=dist.get_world_size(),
            shuffle=False,
            seed=0,
            drop_last=False,
            even_divisible=False,
        )[dist.get_rank()]
    return train, val


def create_loaders(
    dataset: str = "../data/RS/RS_train_split.json",
    data_dir: str = "../data/RS/conformed",
    batch_size: int = 24,
    img_size: Union[Tuple[int, int], int] = (224, 224),
    crop_to_brain_margin=(5, 10, 20),
    slicing_direction: str = "axial",
    slice_thickness: Optional[int] = 1,
    skull_strip: float = 0.0,
    cache: str = "persistent",
    num_workers: int = 1,
    target_sequence: str = "flair",
    guidance_sequences: Sequence[str] = ("t1", "t2"),
    subset_train: Optional[int] = None,
    subset_val: Optional[int] = None,
    random_slicing_direction: bool = False,
):
    if not num_workers and dist.is_initialized():
        warnings.warn(
            f"Using DDP with 0 workers can stall the training process, setting to 1"
        )
        num_workers = 1
    relevant_sequences = [target_sequence] + list(guidance_sequences)

    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    augs = []

    train, val = get_datasets(
        dataset=dataset,
        data_dir=data_dir,
        relevant_sequences=list(relevant_sequences),
        normalize_to=(-1, 1),
        size=img_size,
        slice_thickness=slice_thickness,
        slicing_direction=slicing_direction,
        skull_strip=skull_strip,
        cache=cache,
        subset_val=subset_val,
        subset_train=subset_train,
        random_slicing_direction=random_slicing_direction,
        aug_transforms=transforms.Compose(augs),
        crop_to_brain_margin=crop_to_brain_margin,
    )
    train_loader = MultiEpochsDataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    val_loader = MultiEpochsDataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    print(f"Rank {os.getenv('LOCAL_RANK')} - Train size: {len(train)}")
    return train_loader, val_loader


# from https://discuss.pytorch.org/t/enumerate-dataloader-slow/87778/3 and
# https://github.com/huggingface/pytorch-image-models/blob/d72ac0db259275233877be8c1d4872163954dfbb/timm/data/loader.py#L209-L238
class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
