import warnings

import argparse
import os
from glob import glob

import sys

sys.path.append("..")

import numpy as np
from monai.data import Dataset
from monai import transforms
from omegaconf import OmegaConf
from tqdm import tqdm
from generative.metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric
from torchvision.models import resnet50, densenet121, inception_v3
import torch
import pandas as pd
from lpips import LPIPS

from torchmetrics.image import VisualInformationFidelity

from lib.utils.monai_helper import is_greater_0

device = "cuda"
batch_size = 16


def get_dataset(path, target, skull_strip=False):
    samples_datalist = []

    alt_path = None
    if skull_strip and not glob(f"{path}/*/*_mask.png"):
        warnings.warn("No masks found. Using alternative path.")
        alt_path = "output/masks"

    for sample_path in sorted(list(glob(f"{path}/*/*pred.png"))):
        sample = {
            "pred": sample_path,
            "gt": sample_path.replace("_pred.png", f"_{target}.png"),
            "subject_ID": os.path.basename(os.path.dirname(sample_path)),
        }
        if skull_strip:
            p = sample_path if not alt_path else sample_path.replace(path, alt_path)
            sample["mask"] = p.replace("_pred.png", "_mask.png")
            assert os.path.exists(sample["mask"]), f"{sample['mask']} does not exist."
        samples_datalist.append(sample)

    # create monai dataset
    ds = Dataset(
        samples_datalist,
        transform=transforms.Compose(
            [
                transforms.LoadImaged(
                    image_only=True,
                    ensure_channel_first=True,
                    keys=("pred", "gt", "mask"),
                    allow_missing_keys=True,
                ),
                transforms.ScaleIntensityRanged(
                    keys=("pred", "gt"),
                    allow_missing_keys=True,
                    a_min=0.0,
                    a_max=255.0,
                    b_min=0.0,
                    b_max=1.0,
                ),
                # pad mask to 224x224
                transforms.SpatialPadd(
                    spatial_size=(224, 224),
                    keys=("mask"),
                    allow_missing_keys=True,
                ),
                transforms.MaskIntensityd(
                    keys=("pred", "gt"),
                    mask_key="mask",
                    allow_missing_keys=False,
                    select_fn=is_greater_0,
                )
                if skull_strip
                else transforms.Identityd(keys=("pred", "gt")),
                transforms.ToTensord(keys=("pred", "gt"), dtype=torch.float),
            ]
        ),
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=False,
        shuffle=False,
    )
    return loader


# from https://github.com/Warvito/generative_brain_controlnet/blob/main/src/python/testing/compute_fid.py
class RadNetModelMonai(torch.nn.Module):
    def __init__(self, device=device):
        super().__init__()
        self.model = torch.hub.load(
            "Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True
        ).to(device)
        self.model.eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1) - self.mean
        # x = x / self.std

        feat = self.model(x)
        return feat.mean(dim=[2, 3])


class RadImageNetBackbone(torch.nn.Module):
    def __init__(
        self, model_path="../data/pretrained_models/RadImageNet_pytorch/ResNet50.pt"
    ):
        # see https://github.com/BMEII-AI/RadImageNet/blob/main/pytorch_example.ipynb
        # weights from https://drive.google.com/drive/folders/1FUir_Y_kbQZWih1TMVf9Sz8Pdk9NF2Ym?usp=sharing
        super().__init__()
        if "ResNet" in model_path:
            base_model = resnet50(pretrained=False)
        elif "DenseNet" in model_path:
            base_model = densenet121(pretrained=False)
        elif "Inception" in model_path:
            # https://github.com/BMEII-AI/RadImageNet/issues/16
            base_model = inception_v3(pretrained=False, aux_logits=False)
        else:
            raise ValueError(f"Unknown model {model_path}")
        encoder_layers = list(base_model.children())
        self.backbone = torch.nn.Sequential(*encoder_layers[:-1])

        state_dict = torch.load(model_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[9:]] = v

        # center crop or pad to 224x224
        self.transform = transforms.ResizeWithPadOrCrop((-1, 224, 224))

        print(self.backbone.load_state_dict(new_state_dict))

    @torch.no_grad()
    def forward(self, x):
        x = self.transform(x)
        x = (x.repeat(1, 3, 1, 1) - 0.5) * 2
        pred = self.backbone(x)
        return pred.mean(dim=[2, 3])


class Evaluator2d:
    def __init__(
        self,
        device,
        models_path="../data/pretrained_models/RadImageNet_pytorch",
        quick_fid=False,
        no_fid=False,
    ):
        self.quick_fid = quick_fid or no_fid
        self.device = device
        self.models_radnet = (
            {}
            if no_fid
            else {
                os.path.basename(m)
                .replace(".pt", ""): RadImageNetBackbone(m)
                .eval()
                .to(device)
                for m in (
                    [
                        f"{models_path}/ResNet50.pt",
                        f"{models_path}/DenseNet121.pt",
                        f"{models_path}/InceptionV3.pt",
                    ]
                    if not quick_fid
                    else [f"{models_path}/ResNet50.pt"]
                )
            }
        )
        self.model_monai = RadNetModelMonai(device=device) if not quick_fid else None
        self.ssim_m = SSIMMetric(spatial_dims=2)
        ms_weights = np.array((0.0448, 0.2856, 0.3001, 0.2363))
        ms_weights /= ms_weights.sum()
        self.msssim_m = MultiScaleSSIMMetric(spatial_dims=2, weights=ms_weights)
        self.lpips_m = LPIPS(net="alex", verbose=True).to(device)
        self.mmm_m = MMDMetric()
        self.fid_m = FIDMetric()
        self.vid = VisualInformationFidelity().to(device)

        self.df = []

        self.mmm = []
        self.pred_features_monai = []
        self.gt_features_monai = []
        self.pred_features_radnet = {k: [] for k in self.models_radnet.keys()}
        self.gt_features_radnet = {k: [] for k in self.models_radnet.keys()}

    @staticmethod
    def pixel_dif_metrics(pred, gt, max_val=1):
        mse = ((pred - gt) ** 2).mean()
        psnr_value = 20 * torch.log10(max_val / torch.sqrt(mse))
        rmse = torch.sqrt(mse)
        mad = torch.mean(torch.abs(pred - gt))
        return {"psnr": psnr_value.item(), "rmse": rmse.item(), "mad": mad.item()}

    def __call__(self, pred, gt, subject_ID, mask=None):
        pred = pred.to(self.device).contiguous()
        gt = gt.to(self.device).contiguous()
        if isinstance(subject_ID, str):
            subject_ID = [subject_ID] * pred.shape[0]

        if mask is not None:
            valid_slices = mask.sum(dim=[1, 2, 3]) > 20
            if valid_slices.sum() == 0:
                return
            pred = pred[valid_slices]
            gt = gt[valid_slices]
            mask = mask[valid_slices]
            subject_ID = [s for i, s in enumerate(subject_ID) if valid_slices[i]]

        cos_feat = torch.zeros(pred.shape[0], device=self.device)

        if not self.quick_fid:
            pred_feat_monai = self.model_monai(pred)
            gt_feat_monai = self.model_monai(gt)
            self.pred_features_monai.append(pred_feat_monai.cpu())
            self.gt_features_monai.append(gt_feat_monai.cpu())

        for k, model in self.models_radnet.items():
            pred_feat_radnet = model(pred)
            gt_feat_radnet = model(gt)
            self.pred_features_radnet[k].append(pred_feat_radnet.cpu())
            self.gt_features_radnet[k].append(gt_feat_radnet.cpu())

            if k == "ResNet50":
                cos_feat = torch.nn.functional.cosine_similarity(
                    pred_feat_radnet, gt_feat_radnet, dim=1
                )

        ssim = self.ssim_m(pred, gt)
        vif = self.vid(pred, gt)
        try:
            msssim = self.msssim_m(pred, gt)
        except Exception as e:
            print(f"Error creating MS-SSIM metric: {e}")
            msssim = ssim * torch.nan
        lpips = self.lpips_m(pred, gt)
        self.mmm.append(self.mmm_m(pred, gt))

        for i in range(pred.shape[0]):
            pred_i = pred[i, 0]
            gt_i = gt[i, 0]

            metrics = {"subject_ID": subject_ID[i]}
            metrics.update(self.pixel_dif_metrics(pred_i, gt_i))
            metrics["ssim"] = ssim[i].item()
            metrics["msssim"] = msssim[i].item()
            metrics["vif"] = vif.item()
            metrics["lpips"] = lpips[i].item()
            metrics["cos_feat"] = cos_feat[i].item()
            self.df.append(metrics)

    def summary(self):
        df = pd.DataFrame(self.df)

        # Aggregate statistics
        agg = (
            df.groupby("subject_ID")
            .mean()
            .reset_index()
            .drop(columns=["subject_ID"])
            .agg(["mean", "std"])
        )
        subject_wise_results = df.groupby("subject_ID").mean().reset_index()

        # Compute FID for MONAI and RadNet models
        if not self.quick_fid:
            fid_monai = self.fid_m(
                torch.cat(self.pred_features_monai), torch.cat(self.gt_features_monai)
            )
            agg["fid_monai"] = [fid_monai.item(), np.nan]

        for k in self.models_radnet.keys():
            fid_radnet = self.fid_m(
                torch.cat(self.pred_features_radnet[k]),
                torch.cat(self.gt_features_radnet[k]),
            )
            agg[f"fid_radnet_{k}"] = [fid_radnet.item(), np.nan]

        # Compute mean MMD
        agg["mmm"] = [torch.mean(torch.stack(self.mmm)).item(), np.nan]

        return agg, subject_wise_results


def main(path, target, skull_strip=False, device=device):
    loader = get_dataset(path, target, skull_strip=skull_strip)

    evaluator = Evaluator2d(device)

    for batch in tqdm(loader):
        mask = batch["mask"] if skull_strip else None
        evaluator(batch["pred"], batch["gt"], batch["subject_ID"], mask=mask)

    agg, subj_wise_res = evaluator.summary()
    print(agg)
    out_path = f"{path}_metrics.csv"
    if skull_strip:
        out_path = f"{path}_metrics_skull_strip.csv"
    print(f"Saving metrics to {out_path}")
    agg.to_csv(out_path)
    subj_wise_res.to_csv(f"{out_path.replace('.csv', '')}_subject_wise.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to compute metrics for 2D slices."
    )

    parser.add_argument("path", type=str, help="Path to the directory.")

    parser.add_argument(
        "-s", "--skull_strip", action="store_true", help="Skull strip images."
    )

    args = parser.parse_args()

    config = OmegaConf.load(os.path.join(args.path, "..", "config.yml"))
    target = config.data.target_sequence

    main(args.path, target, args.skull_strip)
