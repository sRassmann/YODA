import os
import nibabel as nib
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import sys

sys.path.append("..")
from flairsyn.lib.utils.fastsurfer_conform import rescale


def main(args):
    os.makedirs(args.out_path, exist_ok=True)

    if not args.subject:
        subjects = sorted(os.listdir(args.in_path))
        print(f"Found {len(subjects)} subjects in {args.in_path}.")
    else:
        subjects = [args.subject]
        print(f"Processing subject {args.subject}.")

    for subj in tqdm(subjects):
        os.makedirs(os.path.join(args.out_path, subj), exist_ok=True)
        for seq in args.seqs:
            if not ("nii" in seq or "mgz" in seq):
                seq += ".nii.gz"
            f = os.path.join(args.in_path, subj, seq)
            try:
                conform(f, os.path.join(args.out_path, subj, seq), dry_run=False)
            except FileNotFoundError:
                print(f"File {f} not found, skipping...")
                continue


def conform(src, tar, dry_run=False):
    print(f"Conforming files in {src} to {tar}")

    if dry_run:
        return

    # open as nibabel image, apply conform, save as nibabel image keeping header
    img = nib.load(src)
    img_data = img.get_fdata()
    img_data_conformed = rescale(img_data, 0, 255.999, 0.0, 0.999).astype(np.uint8)
    img_conformed = nib.Nifti1Image(img_data_conformed, img.affine, img.header)

    # save conformed image
    nib.save(img_conformed, tar)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--in_path",
        "-i",
        type=str,
        default="/groups/ag-reuter/projects/flair_synthesis/raw_dataset/scans",
    )
    parser.add_argument(
        "--out_path",
        "-o",
        type=str,
        default="/groups/ag-reuter/projects/flair_synthesis/raw_dataset/conformed",
    )
    parser.add_argument(
        "--seqs",
        type=str,
        nargs="+",
        default=[
            "FLAIR.nii.gz",
            "T1.nii.gz",
            "T2.nii.gz",
            "T1_RMS.nii.gz",
            "T2_caipi.nii.gz",
            "FLAIR_0p8.nii.gz",
            "FLAIR_0p8_repeat.nii.gz",
        ],
    )
    parser.add_argument(
        "--subject", "-s", type=str, default="", help="Process only this subject"
    )
    args = parser.parse_args()
    main(args)
