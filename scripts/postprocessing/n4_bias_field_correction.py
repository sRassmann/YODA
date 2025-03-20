# based on https://github.com/Deep-MI/FastSurfer/blob/dc87146944153af21d93fd30a1e9f1f43908f58e/recon_surf/N4_bias_correct.py#L31
# Authors: Martin Reuter, David Kuegler
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import SimpleITK as sitk
import os
from tqdm import tqdm
from argparse import ArgumentParser
from glob import glob


def itk_n4_bfcorrection(
    itk_image: sitk.Image,
    itk_mask: sitk.Image = None,
    shrink: int = 4,
    levels: int = 4,
    numiter: int = 50,
    thres: float = 0.0,
) -> sitk.Image:
    # Convert image and mask to float if necessary
    if itk_image.GetPixelIDTypeAsString() != "32-bit float":
        itk_image = sitk.Cast(itk_image, sitk.sitkFloat32)

    if itk_mask and itk_mask.GetPixelIDTypeAsString() != "32-bit float":
        itk_mask = sitk.Cast(itk_mask, sitk.sitkFloat32)

    if itk_mask:
        itk_mask = itk_mask > 0
    else:
        itk_mask = sitk.Abs(itk_image) >= 0
        itk_mask.CopyInformation(itk_image)

    itk_orig = itk_image

    if shrink > 1:
        itk_image = sitk.Shrink(itk_image, [shrink] * itk_image.GetDimension())
        itk_mask = sitk.Shrink(itk_mask, [shrink] * itk_image.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([numiter] * levels)
    corrector.SetConvergenceThreshold(thres)
    corrector.Execute(itk_image, itk_mask)

    log_bias_field = corrector.GetLogBiasFieldAsImage(itk_orig)
    itk_bfcorr_image = itk_orig / sitk.Exp(log_bias_field)

    return itk_bfcorr_image


def process_subject(args, subj):
    try:
        # Load the images
        flair_image = sitk.ReadImage(os.path.join(args.path, subj, args.tar_img))
        mask_image = sitk.ReadImage(os.path.join(args.mask_path, subj, args.mask_img))

        # copy information
        mask_image.CopyInformation(flair_image)

        # Perform the bias field correction
        bf_corrected_image = itk_n4_bfcorrection(flair_image, mask_image)

        # Save the corrected image
        sitk.WriteImage(
            bf_corrected_image,
            os.path.join(
                args.path, subj, args.tar_img.replace(".nii.gz", "_n4.nii.gz")
            ),
        )
    except Exception as e:
        pass


if __name__ == "__main__":
    # argparse
    parser = ArgumentParser(description="N4 bias field correction.")
    parser.add_argument(
        "path", type=str, help="Path to the directory containing the images."
    )
    parser.add_argument(
        "--tar_img", "-t", type=str, default="pred_flair.nii.gz", help="Target image."
    )
    parser.add_argument(
        "--mask_img", "-m", type=str, default="mask.nii.gz", help="Mask image."
    )
    parser.add_argument(
        "--mask_path", "-mp", type=str, default=None, help="Mask image path."
    )
    parser.add_argument(
        "--num_workers", "-n", type=int, default=0, help="Number of workers."
    )
    args = parser.parse_args()
    if args.mask_path is None:
        args.mask_path = args.path

    subjects = os.listdir(args.path)

    if args.num_workers > 1:
        process_subject = partial(process_subject, args)
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            list(tqdm(executor.map(process_subject, subjects), total=len(subjects)))
    else:
        for subj in tqdm(subjects):
            process_subject(args, subj)
