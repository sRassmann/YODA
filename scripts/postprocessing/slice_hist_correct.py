import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import argparse


@torch.no_grad()
def erode(mask, erode_size=3):
    org_shape, org_dtype = mask.shape, mask.dtype
    mask = (mask > 0).float()

    filter = (
        torch.ones((erode_size, erode_size, erode_size))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(mask.device)
    )
    conv_res = torch.nn.functional.conv3d(
        mask.unsqueeze(0), filter, padding=erode_size // 2
    ).squeeze(dim=0)
    erode = conv_res == filter.sum()

    assert erode.shape == org_shape

    return erode.to(org_dtype)


class GradientLoss(nn.Module):
    def __init__(self, mask, pyramid=None):
        super(GradientLoss, self).__init__()
        self.pyramid = [1] if pyramid is None else pyramid
        self.mask = mask[1:]

    def forward(self, x):
        dz = x[1:] - x[:-1]
        dz = dz * self.mask
        return torch.mean(dz**2, dim=(1, 2))


class LearnableGammaCorrection(nn.Module):
    def __init__(self, n_slices, learn_a=True, learn_gamma=True, learn_c=True):
        super(LearnableGammaCorrection, self).__init__()
        self.A = nn.Parameter(torch.ones(n_slices, 1, 1), requires_grad=learn_a)
        self.gamma = nn.Parameter(torch.ones(n_slices, 1, 1), requires_grad=learn_gamma)
        self.c_out = nn.Parameter(torch.zeros(n_slices, 1, 1), requires_grad=learn_c)
        self.c_in = nn.Parameter(torch.zeros(n_slices, 1, 1))  # unused

    def forward(self, x):
        return self.A * torch.pow(x, self.gamma) + self.c_out

    def regularize(self):
        return torch.mean((self.A - 1) ** 2 + (self.gamma - 1) ** 2 + self.c_out**2)


def slice_correct(
    x,
    max_steps=10000,
    reg_weight=0.02,
    lr=10,
    verbose=False,
    learn_a=True,
    learn_gamma=True,
    learn_c=True,
):
    mask_org = x > 0
    mask = erode(mask_org, erode_size=5)

    slices_mask = torch.sum(mask, dim=(1, 2)) > 20
    x = x / x.max()
    x_org = x.clone()

    x = x[slices_mask]
    mask = mask[slices_mask]

    cor = LearnableGammaCorrection(
        x.shape[0], learn_a=learn_a, learn_gamma=learn_gamma, learn_c=learn_c
    ).to(x.device)
    loss = GradientLoss(mask)
    optimizer = torch.optim.SGD([p for p in cor.parameters() if p.requires_grad], lr=lr)

    last_loss = 42
    converge_counter = 0
    log_loss, reg_loss, gamma, A, C_in, C_out = [], [], [], [], [], []
    for i in range(max_steps):
        optimizer.zero_grad()
        x_cor = cor(x)
        loss_val = torch.mean(loss(x_cor))
        reg = cor.regularize()
        loss_val += reg_weight * reg
        loss_val.backward()
        optimizer.step()
        if abs(last_loss - loss_val.item()) < 1e-11:
            converge_counter += 1
            if converge_counter > 20:
                print(f"Converged after {i} steps")
                break
        else:
            converge_counter = 0
            last_loss = loss_val.item()

        log_loss.append(loss_val.item())
        reg_loss.append(reg.item())
        gamma.append(cor.gamma.mean().item())
        A.append(cor.A.mean().item())
        C_in.append(cor.c_in.mean().item())
        C_out.append(cor.c_out.mean().item())

        if verbose and i % 100 == 0:
            print(f"Step {i}: Loss {loss_val.item()}, Reg {reg.item()}")

    with torch.no_grad():
        x = cor(x)
    x_org[slices_mask] = x
    true_indices = torch.nonzero(slices_mask, as_tuple=True)[0]
    first_index = true_indices[0].item()
    last_index = true_indices[-1].item() + 1
    x_org[:first_index] = (
        cor.A[0] * torch.pow(x_org[:first_index], cor.gamma[0]) + cor.c_out[0]
    )
    x_org[last_index:] = (
        cor.A[-1] * torch.pow(x_org[last_index:], cor.gamma[-1]) + cor.c_out[-1]
    )
    del x, mask, optimizer, cor
    torch.cuda.empty_cache()
    return (x_org * mask_org) * 255


def main(
    input,
    output=None,
    target_sequence="pred_flair",
    view="ax",
    fix_a=True,
    fix_gamma=True,
    fix_c=True,
):
    device = "cuda"
    output = input + "_cor" if output is None else output

    for f in glob(f"{input}/*/{target_sequence}.nii.gz"):
        subj = f.split("/")[-2]
        if os.path.exists(os.path.join(output, subj, target_sequence + ".nii.gz")):
            continue
        os.makedirs(os.path.join(output, subj), exist_ok=True)
        ni = nib.load(f)
        x = ni.get_fdata()
        if view == "cor":
            x = np.moveaxis(x, 0, 1)
        elif view == "sag":
            x = np.moveaxis(x, 2, 0)

        x = torch.tensor(x).to(device)
        x_cor = slice_correct(
            x,
            max_steps=10000,
            verbose=False,
            learn_a=not fix_a,
            learn_gamma=not fix_gamma,
            learn_c=not fix_c,
        )
        x_cor = x_cor.cpu().detach().numpy()
        x_cor = np.clip(x_cor, 0, 255).astype(np.uint8)

        if view == "cor":
            x_cor = np.moveaxis(x_cor, 1, 0)
        elif view == "sag":
            x_cor = np.moveaxis(x_cor, 0, 2)
        ni = nib.Nifti1Image(x_cor, ni.affine)
        nib.save(ni, os.path.join(output, subj, f"{target_sequence}.nii.gz"))
        for guid in os.listdir(os.path.join(input, subj)):
            if guid[0] == "_" or not guid.endswith(".nii.gz"):
                continue
            os.symlink(
                os.path.join(
                    input,
                    subj,
                    guid,
                ),
                os.path.join(
                    output, subj, "orig.nii.gz" if target_sequence in guid else guid
                ),
            )
        del x, x_cor, ni


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-processing script for slice histogram correction."
    )
    parser.add_argument("input", type=str, help="Output directory for corrected files")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory for corrected files",
        default=None,
    )
    parser.add_argument(
        "--target", type=str, default="pred_flair", help="Target sequence"
    )
    parser.add_argument(
        "-v",
        "--view",
        choices=["ax", "sag", "cor"],
        help="view of image (main slicing plain)",
        default="ax",
    )

    # contraining the optimized parameters:
    # -fA -fC --> only optimize gamma --> default Gamma correction
    # -fG --> only optimize A,C --> linear correction
    # non of this : generized Gamma correction
    parser.add_argument(
        "-fA",
        "--fix_A",
        action="store_true",
        help="Fix A (A not learnable)",
    )
    parser.add_argument(
        "-fG",
        "--fix_gamma",
        action="store_true",
        help="Fix gamma (gamma not learnable)",
    )
    parser.add_argument(
        "-fC", "--fix_C", action="store_true", help="Fix C (C not learnable)"
    )
    args = parser.parse_args()
    if all([args.fix_A, args.fix_gamma, args.fix_C]):
        raise RuntimeError("Cannot fix all params (a, gamma, c)")

    main(
        args.input,
        args.output,
        args.target,
        args.view,
        args.fix_A,
        args.fix_gamma,
        args.fix_C,
    )

# TODO remove comands and add to YODA repo
