import os
import numpy as np
import nibabel as nib
from scipy.spatial.distance import directed_hausdorff
import argparse
import pandas as pd
from tqdm import tqdm
from metrics_lesions import getHD
import SimpleITK as sitk

LABELS = {
    1: "NCR",  # necrotic tumor core
    2: "ED",  # edematous/invaded tissue  --> FLAIR hyperintense
    3: "ET",  # enhancing tumor
}


def load_nifti(file_path, get_affine=False):
    """Load a NIfTI file and return the image data."""
    img = nib.load(file_path)
    if get_affine:
        return img.get_fdata(), img.affine
    return img.get_fdata()


def dice_coefficient(pred, gt, label):
    """Calculate the Dice coefficient for a specific label."""
    pred_label = pred == label
    gt_label = gt == label
    intersection = np.sum(pred_label * gt_label)
    if np.sum(pred_label) + np.sum(gt_label) == 0:
        return 1.0
    return 2.0 * intersection / (np.sum(pred_label) + np.sum(gt_label))


def hausdorff_distance(pred: np.array, gt: np.array, label: int):
    """Calculate the Hausdorff distance for a specific label."""
    if np.sum(pred == label) == 0 and np.sum(pred == label) == 0:
        return 0.0
    elif np.sum(pred == label) == 0 or np.sum(pred == label) == 0:
        return 100.0
    pred_itk = sitk.GetImageFromArray((pred == label).astype(np.uint8))
    gt_itk = sitk.GetImageFromArray((gt == label).astype(np.uint8))
    return getHD(pred_itk, gt_itk)


def evaluate_segmentation(pred_dir):
    """Evaluate the segmentation for all subjects in the given directories."""
    subjects = os.listdir(pred_dir)
    df = []

    for subj in tqdm(subjects):
        gt_file = os.path.join(pred_dir, subj, "seg.nii.gz")
        pred_file = os.path.join(pred_dir, subj, "results/tumor_isen2020_class.nii.gz")

        if not os.path.exists(pred_file) or not os.path.exists(gt_file):
            print(f"Missing files for subject {subj}, skipping...")
            continue

        pred = load_nifti(pred_file)
        pred[pred == 4] = 3  # changes in brats labels
        gt = load_nifti(gt_file)

        dice_scores = {
            f"dice_{label}": dice_coefficient(pred, gt, ind)
            for ind, label in LABELS.items()
        }
        hausdorff_distances = {
            f"hd_{label}": hausdorff_distance(pred, gt, ind)
            for ind, label in LABELS.items()
        }

        df.append({"subject": subj, **dice_scores, **hausdorff_distances})

    df = pd.DataFrame(df)
    df["dice"] = df[[f"dice_{label}" for label in LABELS.values()]].mean(axis=1)
    df["hd"] = df[[f"hd_{label}" for label in LABELS.values()]].mean(axis=1)

    # print summary
    print(df.iloc[:, 1:].mean())
    print(f"Number of subjects: {len(df)}")

    if pred_dir[-1] == "/":
        pred_dir = pred_dir[:-1]

    df.to_csv(f"{pred_dir}_brats_eval.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BraTS segmentation.")
    parser.add_argument(
        "directory", type=str, help="Directory with predicted segmentations."
    )
    args = parser.parse_args()

    evaluate_segmentation(args.directory)
