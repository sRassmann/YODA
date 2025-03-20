import os
import sys

import nibabel
import numpy as np

sys.path.append("..")
sys.path.append("../flairsyn")
from flairsyn.brats_eval import *
import SimpleITK as sitk


def evaluate_segmentation(pred_dir, gt_dir, calc_hd=False):
    """Evaluate the segmentation for all subjects in the given directories."""
    subjects = os.listdir(pred_dir)
    df = {}
    assigned_labels = set()
    gt_labels = set()

    for subj in tqdm(subjects):
        unified_label_mask = None
        subj_df = {}
        mask, aff = load_nifti(
            os.path.join(pred_dir, subj, "mask.nii.gz"), get_affine=True
        )

        for label_id, label in tqdm(
            enumerate(sorted(os.listdir(os.path.join(gt_dir, subj, "segmentations"))))
        ):
            if label == "spinal_cord.nii.gz" or "vertebrae" in label:
                continue
            gt = load_nifti(os.path.join(gt_dir, subj, "segmentations", label))
            gt = gt * mask

            if np.sum(gt > 0) < 5:
                os.remove(os.path.join(gt_dir, subj, "segmentations", label))
                print(label)
                continue

            pred = load_nifti(os.path.join(pred_dir, subj, "segmentations", label))
            pred = pred * mask
            label = label.replace(".nii.gz", "")

            if unified_label_mask is None:
                unified_label_mask = np.zeros_like(gt)

            unified_label_mask[pred > 0] = label_id + 1

            subj_df["dice_" + label] = dice_coefficient(pred, gt, 1)
            if calc_hd and np.sum(gt > 0) > 0:
                subj_df["hd_" + label] = hausdorff_distance(pred, gt, 1)

        df[subj] = subj_df
        # save unified label mask
        unified_label_mask = nib.Nifti1Image(unified_label_mask, aff)
        nib.save(
            unified_label_mask, os.path.join(pred_dir, subj, "pred_ct_totalseg.nii.gz")
        )

    df = pd.DataFrame(df).T
    df["dice"] = df[[c for c in df.columns if "dice" in c]].mean(axis=1)
    if calc_hd:
        df["hd"] = df[[c for c in df.columns if "hd" in c]].mean(axis=1)

    if pred_dir[-1] == "/":
        pred_dir = pred_dir[:-1]

    print(df[["dice", "hd"]] if calc_hd else df["dice"])
    print("Note: HD is calibrated to image resolution (i.e. in mm)")

    # call index "subject" instead of "index"
    df.index.name = "subject_ID"
    # df.to_csv(f"{pred_dir}_totalsegmentor_eval.csv", index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BraTS segmentation.")
    parser.add_argument(
        "directory", type=str, help="Directory with predicted segmentations."
    )
    parser.add_argument(
        "--gt_directory",
        type=str,
        help="Directory with ground truth segmentations.",
        default="/home/rassmanns/diffusion/flairsyn/output/original/inference_ct",
    )
    parser.add_argument(
        "--calc_hd", action="store_true", help="Calculate Hausdorff distance."
    )
    args = parser.parse_args()

    evaluate_segmentation(args.directory, args.gt_directory, args.calc_hd)
