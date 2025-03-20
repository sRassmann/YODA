import sys

sys.path.append("..")
sys.path.append("../flairsyn")

import os
import argparse
import tempfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from flairsyn.brats_eval import load_nifti, dice_coefficient, hausdorff_distance


gt_dir = "/groups/ag-reuter/projects/flair_synthesis/public_test/IXI/fastsurfer"


def process_subject(subj, pred_dir, pred_file, calc_hd, tmp_dir):
    """Process a single subject for segmentation evaluation."""
    gt_file = os.path.join(gt_dir, subj, "mri/aseg.auto_noCCseg.mgz")
    pred_file = os.path.join(pred_dir, subj, f"{pred_file}.nii.gz")
    if not os.path.exists(pred_file) or not os.path.exists(gt_file):
        print(f"Missing files for subject {subj}, skipping...")
        return None

    tmp_file = f"{tmp_dir}/{subj}_mask.nii.gz"
    os.system(
        f"mri_vol2vol --mov {pred_file} --targ {gt_file} --regheader --o {tmp_file}"
    )

    pred = load_nifti(tmp_file)
    pred[pred == 24] = 0  # remove CSF label
    gt = load_nifti(gt_file)
    gt[gt == 24] = 0

    labels = set(np.unique(gt).astype(int)) - {77, 63, 31, 0}

    dice_scores = {f"dice_{ind}": dice_coefficient(pred, gt, ind) for ind in labels}
    hausdorff_distances = {}
    if calc_hd:
        hausdorff_distances = {
            f"hd_{ind}": hausdorff_distance(pred, gt, ind) for ind in labels
        }

    return {
        "subject": subj,
        **dice_scores,
        **hausdorff_distances,
        "labels": labels,
    }


def evaluate_segmentation(pred_dir, calc_hd=False, subject=-1, file="pred_t2_synthseg"):
    """Evaluate the segmentation for all subjects in the given directories in parallel."""
    tmp = tempfile.TemporaryDirectory(suffix="syntheseg_eval")

    subjects = os.listdir(pred_dir)
    if subject >= 0:
        subjects = [subjects[subject]]

    results = []
    assigned_labels = set()
    gt_labels = set()

    with ProcessPoolExecutor(max_workers=64) as executor:
        future_to_subj = {
            executor.submit(
                process_subject, subj, pred_dir, file, calc_hd, tmp.name
            ): subj
            for subj in subjects
        }
        for future in tqdm(as_completed(future_to_subj), total=len(subjects)):
            result = future.result()
            if result:
                results.append(result)
                assigned_labels.update(result["labels"])
                gt_labels.update(result["labels"])

    df = pd.DataFrame(results)
    print(df)
    # drop labels
    df = df.drop("labels", axis=1)
    df["dice"] = df[[f"dice_{label}" for label in gt_labels]].mean(axis=1)
    if calc_hd:
        df["hd"] = df[[f"hd_{label}" for label in gt_labels]].mean(axis=1)

    if pred_dir[-1] == "/":
        pred_dir = pred_dir[:-1]

    df.to_csv(f"{pred_dir}_syntheseg_eval_{file}.csv", index=False)

    print(df.iloc[:, 1:].mean())
    print(f"Number of subjects: {len(df)}")

    print("Assigned labels:", assigned_labels)
    print("GT labels:", gt_labels)
    print("Missing labels:", gt_labels - assigned_labels)

    tmp.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BraTS segmentation.")
    parser.add_argument(
        "directory", type=str, help="Directory with predicted segmentations."
    )
    parser.add_argument(
        "--calc_hd", action="store_true", help="Calculate Hausdorff distance."
    )
    parser.add_argument("--subject", "-s", type=int, default=-1, help="Subject index.")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="pred_t2_synthseg_resample",
        help="Suffix of the prediction file.",
    )
    args = parser.parse_args()

    evaluate_segmentation(args.directory, args.calc_hd, args.subject, args.file)
