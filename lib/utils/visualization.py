import os
import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import logging
import nibabel as nib
from lib.utils.etc import running_in_ddp, is_rank_0, print0
import torch.distributed as dist

logger = logging.getLogger(__name__)


def create_val_batch(
    sample_path,
    val_cache_path,
    val_loader,
    relevant_sequences,
    num_examples=8,
):
    """Create a batch of validation samples and save them to disk."""
    print0(f"Creating validation batch with {num_examples} examples.")
    if not os.path.exists(val_cache_path):
        if is_rank_0():
            val_batch = sample_batches(val_loader, num_examples)
            os.makedirs(val_cache_path, exist_ok=True)
            for seq in relevant_sequences:
                torch.save(val_batch[seq], os.path.join(val_cache_path, f"{seq}.pt"))
        if running_in_ddp():
            dist.barrier()

    val_batch = {}
    for seq in relevant_sequences:
        val_batch[seq] = torch.load(os.path.join(val_cache_path, f"{seq}.pt"))

    for seq in relevant_sequences:
        val_batch[seq] = val_batch[seq][:num_examples]
    if sample_path:
        logger.info(f"Saving samples to {os.path.join(sample_path)}.")
        os.makedirs(sample_path, exist_ok=True)
        for seq in relevant_sequences:
            save_grid(val_batch[seq], sample_path + f"/val_{seq}.png", columns=4)

    if running_in_ddp():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if world_size > 1:
            # distribute val batch
            fct = np.ceil(
                val_batch[relevant_sequences[0]].shape[0] / world_size
            ).astype(int)
            for k, v in val_batch.items():
                val_batch[k] = v[rank * fct : (rank + 1) * fct]
    return val_batch


def sample_batches(val_loader, num_examples):
    batches = []
    total_examples = 0

    while total_examples < num_examples:
        batch = next(iter(val_loader))
        batches.append(batch)
        total_examples += len(batch[list(batch.keys())[0]])

    # Concatenate along the batch dimension
    val_batch = {
        key: torch.cat([b[key] for b in batches], dim=0)[0:num_examples]
        for key in batches[0].keys()
        if isinstance(batches[0][key], torch.Tensor)
    }

    return val_batch


def save_grid(samples: torch.Tensor, path, columns, save_tiff=False):
    samples = samples.detach().cpu().float()
    # samples shape is (N, C, H, W)
    # normalize each sample to [0, 1] base on percentile
    min_per_sample = torch.quantile(samples.view(samples.shape[0], -1), 0.01, dim=1)
    max_per_sample = torch.quantile(samples.view(samples.shape[0], -1), 0.99, dim=1)

    samples = (samples - min_per_sample[:, None, None, None]) / (
        max_per_sample[:, None, None, None] - min_per_sample[:, None, None, None]
    )

    samples = torch.clamp(samples, 0, 1)
    if samples.shape[1] > 1:  # take mid slice
        index = samples.shape[1] // 2
        samples = samples[:, index : index + 1, :, :]

    grid = make_grid(samples, nrow=columns).numpy()[0]
    path = path.replace(".png", "").replace(".tiff", "").replace(".jpg", "")
    if save_tiff:
        cv2.imwrite(f"{path}.tiff", grid)
    grid = grid * 255
    grid = grid.astype(np.uint8)
    cv2.imwrite(f"{path}.png", grid)


def vol_view(vol):
    """Assuming MONAI default SPL orientation."""
    if isinstance(vol, torch.Tensor):
        vol = vol.cpu().numpy()
        if len(vol.shape) == 4:
            vol = vol[0]
    if isinstance(vol, nib.nifti1.Nifti1Image):
        vol = vol.get_fdata()

    # flip the first axis
    vol = np.flip(vol, axis=0)

    # sagittal slice
    plt.subplot(1, 3, 1)
    plt.imshow((vol[:, :, vol.shape[2] // 2]), cmap="gray")
    plt.axis("off")

    # coronal slice
    plt.subplot(1, 3, 2)
    plt.imshow((vol[:, vol.shape[1] // 2, :]), cmap="gray")
    plt.axis("off")

    # axial slice
    plt.subplot(1, 3, 3)
    plt.imshow((vol[vol.shape[0] // 2, :, :]), cmap="gray")
    plt.axis("off")
