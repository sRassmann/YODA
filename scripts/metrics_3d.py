import warnings
import argparse
import os, sys

sys.path.append(os.path.realpath(os.getcwd()))

from lib.utils.monai_helper import is_greater_0
from metrics_2d import Evaluator2d
from lib.utils.etc import (
    close_mask,
    merge_fs_labels,
    cross_filter,
    erode,
)

import numpy as np
from monai.data import Dataset
from monai import transforms
from tqdm import tqdm
from generative.metrics import FIDMetric, MultiScaleSSIMMetric, SSIMMetric
import torchvision
import torch
import pandas as pd

device = "cuda"
batch_size = 1  # not supporting batching for now

# 3d FID based on https://github.com/Warvito/generative_brain/blob/main/src/python/testing/compute_fid.py
# Note that FID is severly biased by the number of samples: https://arxiv.org/pdf/1801.01401.pdf , https://arxiv.org/pdf/1801.01401.pdf


def get_dataset(
    path,
    skull_strip=False,
    mask_path=None,
    pred="pred_flair.nii.gz",
    gt="flair.nii.gz",
    erode_mask=False,
):
    samples_datalist = []

    if gt is None:
        gt = pred.replace("pred_", "")
    mask_path = mask_path if mask_path else path

    for subj in os.listdir(path):
        sample = {
            "pred": os.path.join(path, subj, pred),
            "gt": os.path.join(path, subj, gt),
            "subject_ID": subj,
            "mask": os.path.join(mask_path, subj, "mask.nii.gz"),
        }
        assert os.path.exists(sample["mask"]), f"{sample['mask']} does not exist."
        samples_datalist.append(sample)
    if len(samples_datalist) == 0:
        raise ValueError("No samples found.")

    # create monai dataset
    ds = Dataset(
        samples_datalist,
        transform=transforms.Compose(
            [
                transforms.LoadImaged(
                    image_only=True,
                    ensure_channel_first=True,
                    keys=("pred", "gt", "mask"),
                ),
                transforms.ScaleIntensityRanged(
                    keys=("pred", "gt"),
                    a_min=0.0,
                    a_max=255.0,
                    b_min=0.0,
                    b_max=1.0,
                ),
                transforms.Orientationd(keys=("pred", "gt", "mask"), axcodes="SPL"),
                transforms.CropForegroundd(  # restrict to usual FOV
                    keys=("pred", "gt", "mask"),
                    source_key="mask",
                    allow_smaller=True,
                    margin=(5, 10, 20),
                ),  # as D H W
                transforms.CopyItemsd(keys=["mask"], names=["brain_mask"]),
                transforms.Lambdad(
                    keys=("brain_mask"), func=erode if erode_mask else close_mask
                ),
                transforms.MaskIntensityd(
                    keys=("pred", "gt"),
                    mask_key="brain_mask",
                    select_fn=is_greater_0,
                )
                if skull_strip
                else transforms.Identityd(keys=("pred", "gt")),
                transforms.ToTensord(
                    keys=("pred", "gt", "mask"),
                    dtype=torch.float,
                ),
            ]
        ),
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
    return loader


class MedicalNetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load(
            "Warvito/MedicalNet-models", "medicalnet_resnet50_23datasets"
        )
        self.pad = transforms.SpatialPad(spatial_size=(224, 224, 224))

    @torch.no_grad()
    def forward(self, x):
        x = self.pad(x.squeeze(dim=0))
        x -= x.mean()
        x /= x.std()
        feat = self.backbone(x.unsqueeze(dim=0))
        feat = torch.nn.functional.adaptive_avg_pool3d(feat, (1, 1, 1))
        return feat.view(feat.size(0), -1)


class Evaluator3d:
    def __init__(
        self,
        device="cuda",
        models_path_2d="../data/pretrained_models/RadImageNet_pytorch",
        quick_2d_fid=True,
        no_fid=False,
        no_2d=False,
    ):
        self.device = device
        self.no_fid = no_fid
        self.no_2d = no_2d

        if not no_2d:
            two_d_args = {
                "device": device,
                "quick_fid": quick_2d_fid,
                "no_fid": no_fid,
                "models_path": models_path_2d,
            }
            self.eval_axial = Evaluator2d(**two_d_args)
            self.eval_coronal = Evaluator2d(**two_d_args)
            self.eval_sagittal = Evaluator2d(**two_d_args)

        if not self.no_fid:
            self.model_3d = MedicalNetModel().eval().to(device)

        self.ssim_m = SSIMMetric(spatial_dims=3)
        ms_weights = np.array((0.0448, 0.2856, 0.3001, 0.2363))
        ms_weights /= ms_weights.sum()
        self.msssim_m = MultiScaleSSIMMetric(spatial_dims=3, weights=ms_weights)
        self.fid_m = FIDMetric()

        self.df = []

        self.pred_features_3d, self.gt_features_3d = [], []

    @staticmethod
    def z_gradient(x):
        slice = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        slice = torch.tensor(slice, dtype=x.dtype, device=x.device)
        sobel_z_kernel = torch.stack([slice, torch.zeros_like(slice), -slice])

        def add_dims(a):
            return a.unsqueeze(dim=0).unsqueeze(dim=0)

        gradient_z = torch.nn.functional.conv3d(add_dims(x), add_dims(sobel_z_kernel))
        gradient_z = gradient_z.squeeze(dim=0).squeeze(dim=0)

        return gradient_z

    @staticmethod
    @torch.no_grad()
    def noise_eps(mask: torch.Tensor, x: torch.Tensor, erode_size=3) -> torch.Tensor:
        mask = merge_fs_labels(mask)

        x_bar = torch.zeros_like(x)  # sum up x_bar values across labels
        mask_bar = torch.zeros_like(mask)  # memorize all non-boundary voxels

        for v in mask.unique()[1:]:  # for each label
            # erode masks to remove boundary voxels

            # filter = torch.ones((erode_size, erode_size, erode_size))
            filter = cross_filter().unsqueeze(0).unsqueeze(0).to(mask)
            mask_lab = (
                torch.nn.functional.conv3d(
                    (mask == v).to(mask),
                    filter,
                    padding=erode_size // 2,
                )
                == filter.sum()
            ).float()
            mask_lab = mask_lab.squeeze(0).squeeze(0)

            # mask x for label and divide by mask --> x_bar for label v
            lab = torchvision.transforms.functional.gaussian_blur(
                x.squeeze() * mask_lab.squeeze(), 5
            ) / torchvision.transforms.functional.gaussian_blur(mask_lab.squeeze(), 5)
            lab[lab != lab] = 0  # remove NaNs from div by 0
            lab = lab * (mask == v).float()
            x_bar += lab
            mask_bar += mask_lab

        eps = x - x_bar

        # get average noise for non-boundary voxels
        noise_eps = torch.mean(eps[mask_bar > 0] ** 2) ** 0.5
        wm_eps = torch.mean(eps[(mask == 41) & (mask_bar > 0)] ** 2) ** 0.5
        return noise_eps, wm_eps

    def __call__(self, pred, gt, subject_ID, mask=None, fs_seg=None):
        assert len(subject_ID) == 1, "Batch size must be 1."
        subject_ID = subject_ID[0]
        pred = pred.to(self.device)
        gt = gt.to(self.device)

        # 3D pixel based metrics
        ssim = self.ssim_m(pred, gt)
        try:
            msssim = self.msssim_m(pred, gt)
        except Exception as e:
            print(f"Error creating MS-SSIM metric: {e}")
            msssim = torch.tensor([np.nan])

        # pixel based metrics
        metrics = {
            "subject_ID": subject_ID,
            "ssim": ssim.item(),
            "msssim": msssim.item(),
            "dz": (
                torch.sum(self.z_gradient(pred.squeeze(dim=[0, 1])) ** 2) ** 0.5
            ).item(),
        }
        if fs_seg is not None:
            eps = self.noise_eps(fs_seg * mask, pred)
            metrics["noise_eps"] = eps[0].item()
            metrics["wm_eps"] = eps[1].item()
        metrics.update(Evaluator2d.pixel_dif_metrics(pred, gt))

        if not self.no_fid:
            # 3D FID
            self.pred_features_3d.append(self.model_3d(pred).cpu())
            self.gt_features_3d.append(self.model_3d(gt).cpu())
            metrics["cos_feat"] = torch.nn.functional.cosine_similarity(
                self.pred_features_3d[-1], self.gt_features_3d[-1]
            ).item()

        self.df.append(metrics)

        if not self.no_2d:
            batch_size_2d = 32
            # axial
            for i in range(pred.shape[2] // batch_size_2d):
                pred_i = pred[0, :, i * batch_size_2d : (i + 1) * batch_size_2d]
                pred_i = pred_i.transpose(0, 1)
                gt_i = gt[0, :, i * batch_size_2d : (i + 1) * batch_size_2d]
                gt_i = gt_i.transpose(0, 1)
                if mask is not None:
                    mask_i = mask[0, :, i * batch_size_2d : (i + 1) * batch_size_2d]
                    mask_i = mask_i.transpose(0, 1)
                else:
                    mask_i = None
                self.eval_axial(pred_i, gt_i, subject_ID, mask=mask_i)
            # coronal
            for i in range(pred.shape[3] // batch_size_2d):
                pred_i = pred[0, :, :, i * batch_size_2d : (i + 1) * batch_size_2d]
                pred_i = pred_i.transpose(1, 2).transpose(0, 1).flip(2)
                gt_i = gt[0, :, :, i * batch_size_2d : (i + 1) * batch_size_2d]
                gt_i = gt_i.transpose(1, 2).transpose(0, 1).flip(2)
                if mask is not None:
                    mask_i = mask[0, :, :, i * batch_size_2d : (i + 1) * batch_size_2d]
                    mask_i = mask_i.transpose(1, 2).transpose(0, 1).flip(2)
                else:
                    mask_i = None
                self.eval_coronal(pred_i, gt_i, subject_ID, mask=mask_i)
            # sagittal
            for i in range(pred.shape[4] // batch_size_2d):
                pred_i = pred[0, :, :, :, i * batch_size_2d : (i + 1) * batch_size_2d]
                pred_i = pred_i.transpose(3, 1).transpose(0, 1).transpose(2, 3).flip(2)
                gt_i = gt[0, :, :, :, i * batch_size_2d : (i + 1) * batch_size_2d]
                gt_i = gt_i.transpose(3, 1).transpose(0, 1).transpose(2, 3).flip(2)
                if mask is not None:
                    mask_i = mask[
                        0, :, :, :, i * batch_size_2d : (i + 1) * batch_size_2d
                    ]
                    mask_i = (
                        mask_i.transpose(3, 1).transpose(0, 1).transpose(2, 3).flip(2)
                    )
                else:
                    mask_i = None
                self.eval_sagittal(pred_i, gt_i, subject_ID, mask=mask_i)

    def summary(self):
        df = pd.DataFrame(self.df)

        # summary stats per subject (mean and std)
        agg = (
            df.groupby("subject_ID")
            .mean()
            .reset_index()
            .drop(columns=["subject_ID"])
            .agg(["mean", "std"])
        )

        if not self.no_fid:
            # 3D FID
            fid = self.fid_m(
                torch.cat(self.pred_features_3d), torch.cat(self.gt_features_3d)
            )
            agg["fid_3d"] = [fid.item(), np.nan]

        if not self.no_2d:
            # 2d results
            for axis, eval in zip(
                ["axial", "coronal", "sagittal"],
                [self.eval_axial, self.eval_coronal, self.eval_sagittal],
            ):
                agg_s, subj_wise = eval.summary()

                agg_s = agg_s.drop(columns=["psnr", "rmse", "mad"])  # ~ same as 3d
                agg_s.columns = [f"{axis}_{c}" for c in agg_s.columns]
                agg = pd.concat([agg, agg_s], axis=1)

                # append to df
                subj_wise = subj_wise.drop(columns=["subject_ID"])
                subj_wise.columns = [f"{axis}_{c}" for c in subj_wise.columns]
                df = pd.concat([df, subj_wise], axis=1)

        return agg, df


def main(
    path,
    skull_strip=False,
    out_path=None,
    mask_path=None,
    no_fid=False,
    device=device,
    erode_mask=False,
    no_2d=False,
    modality="flair",
    pred_file="",
):
    if out_path is None:
        if path[-1] == "/":
            path = path[:-1]
        out_path = f"{path}_metrics_3D.csv"
        if skull_strip:
            out_path = f"{path}_metrics_3D_skull_strip.csv"
        if erode_mask:
            out_path = out_path.replace(".csv", "_erode_mask.csv")
    else:
        out_path = os.path.join(os.path.dirname(path), out_path)
    print(f"Saving metrics to {out_path}")

    loader = get_dataset(
        path,
        skull_strip=skull_strip,
        mask_path=mask_path,
        erode_mask=erode_mask,
        pred=pred_file,
        gt=f"{modality}.nii.gz",
    )
    evaluator = Evaluator3d(device=device, no_fid=no_fid, no_2d=no_2d)

    for i, batch in enumerate(tqdm(loader)):
        pred = batch["pred"].to(device)
        gt = batch["gt"].to(device)
        mask = batch["brain_mask"].to(device) if "brain_mask" in batch else None
        fs_seg = batch["mask"].to(device) if "mask" in batch else None
        subject_ID = batch["subject_ID"]

        evaluator(pred, gt, subject_ID, mask=mask, fs_seg=fs_seg)

    agg, subj_wise = evaluator.summary()

    print(agg)
    agg.to_csv(out_path)
    subj_wise.to_csv(f"{out_path.replace('.csv', '')}_subject_wise.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to compute metrics for 3D volumes."
    )

    parser.add_argument("path", type=str, help="Path to the directory of the run.")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="flair",
        help="Modality (without the .nii.gz and pred_)",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        default=None,
        help="Name of the prediction file, default is pred_modality.nii.gz.",
    )
    parser.add_argument(
        "-s", "--skull_strip", action="store_true", help="Skull strip images."
    )
    parser.add_argument(
        "-e", "--erode_mask", action="store_true", help="Erode mask for evaluation."
    )

    parser.add_argument(
        "-m",
        "--mask_path",
        type=str,
        default=None,
        help="Path to the mask directory, if None the mask will be searched in the same directory as the input images.",
    )
    parser.add_argument(
        "-nfid",
        "--no_fid",
        action="store_true",
        help="Do not compute FID (feature space distance) metrics to speed up computation.",
    )
    parser.add_argument(
        "-n2d",
        "--no_2d",
        action="store_true",
        help="Do not compute 2D metrics to speed up computation.",
    )

    parser.add_argument("-o", "--output", type=str, default=None, help="Output path.")

    args = parser.parse_args()

    if args.pred_file is None:
        args.pred_file = f"pred_{args.file}.nii.gz"

    main(
        args.path,
        skull_strip=args.skull_strip,
        out_path=args.output,
        mask_path=args.mask_path,
        no_fid=args.no_fid,
        erode_mask=args.erode_mask,
        no_2d=args.no_2d,
        modality=args.file,
        pred_file=args.pred_file,
    )
