import os
import numpy as np
import nibabel as nib
import argparse
from tqdm import tqdm
import shutil

# read in multiple echo of an image, average it and save it
# we assume that the echos to be organized as follows:
# <input_dir_1>/<subject_id>/<sequence_name.nii.gz>
# <input_dir_2>/<subject_id>/<sequence_name.nii.gz>
# ...

# output will be saved in the specified output directory as follows:
# <output_dir>/<subject_id>/<sequence_name.nii.gz>

# fs74 && python scripts/postprocessing/average_echos.py \
# output/sr3/baseline/inference_wmh output/sr3/baseline_cor/inference_wmh output/sr3/baseline_sag/inference_wmh \
#  -o output/sr3/baseline_average/inference


def rms_combine(echos):
    return np.sqrt(np.mean(echos**2, axis=0))


def average_combine(echos):
    return np.mean(echos, axis=0)


def median_combine(echos):
    return np.median(echos, axis=0)


comb_methods = {
    "rms": rms_combine,
    "average": average_combine,
    "median": median_combine,
}


def main(args):
    args.input_dir = os.path.abspath(args.input_dir)
    outdir = args.output_dir
    if outdir is None:
        import re

        outdir = re.sub(r"nex_\d+", f"nex_{args.target_nex}", args.input_dir)
    print(f"Saving to {outdir}")
    for subj in tqdm(os.listdir(args.input_dir)):
        img = nib.load(
            os.path.join(args.input_dir, subj, f"pred_echo_0_{args.sequence}.nii.gz")
        )
        echos = [img.get_fdata()] + [
            nib.load(
                os.path.join(
                    args.input_dir, subj, f"pred_echo_{i}_{args.sequence}.nii.gz"
                )
            ).get_fdata()
            for i in range(1, args.target_nex)
        ]
        echos = np.array(echos)
        combined = comb_methods[args.combine_method](echos)

        os.makedirs(os.path.join(outdir, subj), exist_ok=True)
        img = nib.Nifti1Image(combined, img.affine)
        nib.save(img, os.path.join(outdir, subj, f"pred_{args.sequence}.nii.gz"))

        # symlink the mask and the original image
        for f in ["mask", args.sequence]:
            try:
                os.symlink(
                    os.path.join(args.input_dir, subj, f"{f}.nii.gz"),
                    os.path.join(outdir, subj, f"{f}.nii.gz"),
                )
            except:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="input directories of echos",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default=None,
        type=str,
        help="output directory",
    )
    parser.add_argument(
        "--sequence",
        "-s",
        type=str,
        help="sequence name",
        default="flair",
    )
    parser.add_argument(
        "--combine_method",
        type=str,
        default="rms",
        choices=["rms", "average", "median"],
        help="method to combine echos",
    )
    parser.add_argument(
        "-nex",
        "--target_nex",
        default=4,
        type=int,
    )

    args = parser.parse_args()
    main(args)
