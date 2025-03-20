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
    for subj in tqdm(os.listdir(args.input_dirs[0])):
        echos = []

        for i, input_dirs in enumerate(args.input_dirs):
            if args.register:
                if not i:
                    img = nib.load(os.path.join(input_dirs, subj, args.sequence))
                    dst = os.path.join(input_dirs, subj, args.sequence)
                else:
                    # register the second and after to first image
                    tmp_lta = f"/tmp/req_{subj}_{i}.lta"
                    tmp_mov = f"/tmp/vol_{subj}_{i}.nii.gz"
                    mov = os.path.join(input_dirs, subj, args.sequence)
                    mask_path = os.path.join(input_dirs, subj, "mask.nii.gz")
                    os.system(
                        f"mri_robust_register --satit --mov {mov} --dst {dst} --lta {tmp_lta} --maskdst {mask_path} --maskmov {mask_path}"
                    )
                    os.system(
                        f"mri_vol2vol --mov {mov} --targ {dst} --reg {tmp_lta} --o {tmp_mov} --cubic"
                    )
                    img = nib.load(tmp_mov)
                    os.remove(tmp_lta)
                    os.remove(tmp_mov)
            else:
                img = nib.load(os.path.join(input_dirs, subj, args.sequence))

            echos.append(img.get_fdata())

        echos = np.array(echos)
        combined = comb_methods[args.combine_method](echos)

        os.makedirs(os.path.join(args.output_dir, subj), exist_ok=True)
        img = nib.Nifti1Image(combined, img.affine)
        nib.save(img, os.path.join(args.output_dir, subj, args.sequence))

        # symlink the mask
        mask_path = os.path.join(args.input_dirs[0], subj, "mask.nii.gz")
        if os.path.exists(mask_path):
            shutil.copy(
                os.path.realpath(mask_path),
                os.path.realpath(os.path.join(args.output_dir, subj, "mask.nii.gz")),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dirs",
        type=str,
        nargs="+",
        help="input directories of echos",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        help="output directory",
    )
    parser.add_argument(
        "--sequence",
        "-s",
        type=str,
        help="sequence name",
        default="pred_flair.nii.gz",
    )
    parser.add_argument(
        "--combine_method",
        type=str,
        default="rms",
        choices=["rms", "average", "median"],
        help="method to combine echos",
    )
    parser.add_argument(
        "--register",
        "-r",
        action="store_true",
        help="register echos to the first echo",
    )

    args = parser.parse_args()
    main(args)
