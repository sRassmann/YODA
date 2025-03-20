from argparse import ArgumentParser
import os
from glob import glob
import shutil

# create output folder like the one from single echo inference (i.e. with pred_flair.nii.gz)
# from a MEX (NEX > 1) inference folder


def main(src, target):
    os.makedirs(target, exist_ok=True)
    for subj in os.listdir(src):
        subj_path = os.path.join(src, subj)
        os.makedirs(os.path.join(target, subj), exist_ok=True)
        for file in glob(subj_path + "/*"):
            if "pred" not in os.path.basename(file):
                shutil.copy(file, os.path.join(target, subj, os.path.basename(file)))
            elif "pred_echo_0" in file:
                seq = os.path.basename(file).replace("pred_echo_0_", "")
                seq = "pred_" + seq
                shutil.copy(file, os.path.join(target, subj, seq))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("src", type=str)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    if args.target is None:
        target_name = os.path.basename(args.src).split("_nex")[0]
        args.target = os.path.join(os.path.dirname(args.src), target_name)

    main(args.src, args.target)
