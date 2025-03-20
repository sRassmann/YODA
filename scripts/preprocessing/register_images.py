import os
from glob import glob
import sys
import nibabel as nib
import torch
from argparse import ArgumentParser
import subprocess
import numpy as np

from tqdm import tqdm

import sys

sys.path.append("..")
from flairsyn.lib.utils.fastsurfer_conform import rescale


def main(
    org_path="/groups/ag-reuter/projects/flair_synthesis/raw_dataset/scans",
    target_path="/groups/ag-reuter/projects/flair_synthesis/raw_dataset/registered_skull_stripped",
    target_seq="FLAIR.nii.gz",
    conform_target_path="/groups/ag-reuter/projects/flair_synthesis/conformed_mask_reg",
    start=0,
    end=10000,
    create_softlinks=False,
    dry_run=False,
):
    os.makedirs(target_path, exist_ok=True)
    if conform_target_path:
        os.makedirs(conform_target_path, exist_ok=True)

    subjects = sorted(os.listdir(org_path))
    print(f"Found {len(subjects)} subjects in {org_path}.")

    subjects = subjects[start:end]
    print(f"Processing subjects {start} to {end}.")

    for subj in tqdm(subjects):
        files = register(
            org_path,
            subj,
            target_path,
            target_seq,
            dry_run=dry_run,
            create_softlinks=create_softlinks,
        )
        if not files or not conform_target_path:
            continue

        else:
            os.makedirs(os.path.join(conform_target_path, subj), exist_ok=True)
            for f in files:
                conform(
                    f,
                    os.path.join(conform_target_path, subj, os.path.basename(f)),
                    dry_run=dry_run,
                )


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


def register(
    org_path,
    subj,
    target_path,
    target_seq,
    create_softlinks=False,
    dry_run=False,
    compare_old=False,
):
    os.makedirs(os.path.join(target_path, subj), exist_ok=True)
    tar = os.path.join(org_path, subj, target_seq)
    # create brainmask
    if not os.path.exists(tar):
        print(f"Target sequence ({target_seq}) not found for {subj}")
        return []

    ref_mask = os.path.join(
        target_path, subj, f"{target_seq.split('.')[0]}_brainmask.nii.gz"
    )
    if not dry_run and not os.path.exists(ref_mask):
        subprocess.run(
            f"mri_synthstrip --gpu -i {tar} -m {ref_mask}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    try:
        os.symlink(tar, os.path.join(target_path, subj, target_seq))
    except FileExistsError:
        print(f"File {target_seq} already exists")
    created_volumes = [os.path.join(target_path, subj, target_seq)]

    for f in glob(os.path.join(org_path, subj, "T*.nii.gz")):
        print(f)
        if f == tar:
            continue

        base = os.path.basename(f).split(".")[0]
        reg_file_name = f"{base}_to_{target_seq.split('.')[0]}.lta"
        lta_file_path = os.path.join(target_path, subj, reg_file_name)

        # create brain mask
        mov_mask = os.path.join(target_path, subj, f"{base}_brainmask.nii.gz")
        if not dry_run:
            if not os.path.exists(mov_mask):
                print(f"Creating brain mask for {f}")
                subprocess.run(
                    f"mri_synthstrip --gpu -i {f} -m {mov_mask}",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            # register mov to ref
            if not os.path.exists(lta_file_path):
                print(f"Registering {f} to {tar}")
                subprocess.run(
                    f"mri_coreg --mov {f} --mov-mask {mov_mask} --ref-mask {ref_mask}"
                    f" --ref {tar} --reg {lta_file_path} --threads 16",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            # apply registration to ref and resample
            resample_file = os.path.join(target_path, subj, base + ".nii.gz")
            if not os.path.exists(resample_file):
                print(f"Resampling {f} to {tar}, saving to {resample_file}")
                subprocess.run(
                    f"mri_vol2vol --cubic --mov {f} "
                    f"--targ {tar} "
                    f"--lta {lta_file_path} "
                    f"--o {resample_file}",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            created_volumes.append(resample_file)

        if create_softlinks:
            try:
                os.symlink(
                    os.path.join(org_path, subj, f),
                    os.path.join(target_path, subj, base + "_org.nii.gz"),
                )
            except FileExistsError:
                print(f"File {base}_org.nii.gz already exists")

        if compare_old:
            old_path = f"/home/rassmanns/diffusion/data/RS/registered/volume/{subj}/{base}.nii.gz"
            if os.path.exists(old_path):
                try:
                    os.symlink(
                        old_path,
                        os.path.join(target_path, subj, base + "_old.nii.gz"),
                    )
                except FileExistsError:
                    print(f"File {base}_old.nii.gz already exists")

    return created_volumes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "org_path",
        type=str,
        # default="/groups/ag-reuter/projects/flair_synthesis/raw_dataset/scans",
    )
    parser.add_argument(
        "register_target_path",
        type=str,
        # default="/groups/ag-reuter/projects/flair_synthesis/raw_dataset/registered_skull_stripped",
    )
    parser.add_argument("--target_seq", type=str, default="FLAIR.nii.gz")

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10000)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--create_softlinks",
        action="store_true",
        help="create softlinks to original (unregistered) files",
    )
    parser.add_argument(
        "--conform_target_path",
        type=str,
        default=None,
        # default="/groups/ag-reuter/projects/flair_synthesis/conformed_mask_reg",
    )

    args = parser.parse_args()

    main(
        args.org_path,
        args.register_target_path,
        args.target_seq,
        args.conform_target_path,
        args.start,
        args.end,
        args.create_softlinks,
        args.dry_run,
    )
