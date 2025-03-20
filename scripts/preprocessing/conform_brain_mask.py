# Resample to the space of the input image
# Crop the brain mask to the same size as the input image and save it as a nifti file

import os
import nibabel as nib
from nibabel.processing import resample_from_to
from argparse import ArgumentParser


def main(src, target, subj, reference="FLAIR.nii.gz"):
    # Load the mask and FLAIR as reference images
    # Note, that T1/T2 are in the same space as FLAIR
    if os.path.exists(os.path.join(target, subj, "mask.nii.gz")):
        print(f"Mask for {subj} already exists, skipping.")
        return
    mask_nii = nib.load(os.path.join(src, subj, "mri", "aseg.auto_noCCseg.mgz"))
    image_nii = nib.load(os.path.join(target, subj, reference))

    # Resample mask to match image dimensions
    resampled_mask_nii = resample_from_to(mask_nii, image_nii, order=0)

    # create target folder and save image
    os.makedirs(os.path.join(target, subj), exist_ok=True)
    nib.save(resampled_mask_nii, os.path.join(target, subj, "mask.nii.gz"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("src", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("subj", type=str)
    parser.add_argument("--reference", type=str, default="FLAIR.nii.gz")
    args = parser.parse_args()
    main(args.src, args.target, args.subj, args.reference)
