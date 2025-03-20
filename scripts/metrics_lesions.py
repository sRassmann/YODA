# Based on https://github.com/hjkuijf/wmhchallenge/blob/master/evaluation.py

# -*- coding: utf-8 -*-

import numpy as np
import os
import SimpleITK as sitk
import scipy.spatial
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import binary_dilation


def acc_metrics(
    path,
    pred_file="pred_flair_wmh_seg.nii.gz",
    gt_path="/home/rassmanns/diffusion/flairsyn/output/original/inference_wmh",
    gt_file="wmhs_resampled.nii.gz",
    hd=True,
    contrast=True,
    pv_sep=False,  # separate periventricular from deep white matter hyperintensities
):
    """Main function"""
    vol_thr = 10  # threshold for small lesions (mm**3), corresponds ~60th percentile

    df = []

    subjs = os.listdir(path)

    for s in tqdm(subjs):
        pred = os.path.join(path, s, pred_file)
        gt = os.path.join(gt_path, s, gt_file)
        testImage, resultImage = getImages(gt, pred)

        pred = sitk.GetArrayFromImage(resultImage)
        gt = sitk.GetArrayFromImage(testImage)
        dice = 2 * np.sum(pred * gt) / (np.sum(pred) + np.sum(gt))
        gt_vol = np.sum(gt)
        pred_vol = np.sum(pred)
        log_vol_ratio = np.abs(np.log2(pred_vol / gt_vol))

        if pred_vol == 0:
            continue

        res = {
            "subject_ID": s,
            "dice": dice,
            "log_vol_diff": log_vol_ratio,  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7590957/  2.D
            "pred_vol": pred_vol,
            "gt_vol": gt_vol,
        }
        cc_metrics, gt_ccs = getLesionDetection(testImage, resultImage, vol_thr=vol_thr)
        res.update(cc_metrics)

        if contrast:
            # assess contrast between lesion and surrounding tissue (WM only)
            flair = pred_file.replace("_wmh_seg", "").replace("_pvs_seg", "")
            flair = os.path.join(path, s, flair)
            mask = os.path.join(path, s, "mask.nii.gz")
            res.update(lesion_contrast(flair, mask, gt_ccs))

        if hd:
            res["hd95"] = getHD(testImage, resultImage)

        if pv_sep:
            mask = open_mask(path, s)
            res.update(perivent_deep_wmh(testImage, resultImage, mask))

            vent = np.isin(mask, [4, 43])
            vent = binary_dilation(vent, iterations=10)

            # pv
            pred_pv = pred * vent
            gt_pv = gt * vent
            dice_pv = 2 * np.sum(pred_pv * gt_pv) / (np.sum(pred_pv) + np.sum(gt_pv))

            # deep
            pred_deep = pred * ~vent
            gt_deep = gt * ~vent
            dice_deep = (
                2 * np.sum(pred_deep * gt_deep) / (np.sum(pred_deep) + np.sum(gt_deep))
            )
            res.update({"dice_pv": dice_pv, "dice_deep": dice_deep})

        df.append(res)

    df = pd.DataFrame(df)
    return df


def open_mask(path, s):
    mask_path = os.path.join(path, s, "mask.nii.gz")
    if not os.path.exists(mask_path):
        mask_path = os.path.join("output/original/inference_wmh", s, "mask.nii.gz")
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    return mask


def lesion_contrast(flair_img_path, mask_path, gt_ccs):
    flair = sitk.GetArrayFromImage(sitk.ReadImage(flair_img_path))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    wm = np.isin(mask, [2, 41])  # white matter mask

    inds = np.unique(gt_ccs)[1:]  # for each CC (ie. lesion)
    contrasts = np.ones(len(inds)) * -1
    stds = np.ones(len(inds)) * -1
    weights = np.ones(len(inds)) * -1

    for i, v in enumerate(inds):
        lesion = gt_ccs == v  # current lesion

        # get mask for local WM surrounding the lesion
        surround = binary_dilation(lesion, iterations=3) & ~lesion & wm
        if np.sum(surround) < np.sum(lesion) or np.sum(lesion) <= 1:
            continue

        contrasts[i] = flair[lesion].mean() - flair[surround].mean()
        stds[i] = flair[surround].std()
        weights[i] = np.sum(lesion)

    # exclude invalid lesions
    contrasts = contrasts[weights >= 0]
    stds = stds[weights >= 0]
    weights = weights[weights >= 0]

    # use weighted average of log contrast to not under-represent large lesions
    weighted_snr = np.sum((contrasts / stds) * weights) / np.sum(weights)  # legacy
    unweighted_snr = np.mean(contrasts / stds)
    unnorm_contrast = np.mean(contrasts)
    return {
        "contrast": weighted_snr,
        "contrast_unweighted": unweighted_snr,
        "absolute_contrast": unnorm_contrast,
    }


def getImages(testFilename, resultFilename):
    """Return the test and result images, thresholded and non-WMH masked."""

    testImage = sitk.ReadImage(testFilename)
    resultImage = sitk.ReadImage(resultFilename)

    # Check for equality
    assert (
        testImage.GetSize() == resultImage.GetSize()
    ), f"Image size mismatch for {testFilename} and {resultFilename}"

    # Get meta data from the test-image, needed for some sitk methods that check this
    resultImage.CopyInformation(testImage)

    pred = sitk.BinaryThreshold(resultImage, 1, 1000, 1, 0)
    gt = sitk.BinaryThreshold(testImage, 1, 1000, 1, 0)

    return gt, pred


def getHD(testImage, resultImage):
    """Compute the Hausdorff distance."""

    # Hausdorff distance is only defined when something is detected
    resultStatistics = sitk.StatisticsImageFilter()
    resultStatistics.Execute(resultImage)
    if resultStatistics.GetSum() == 0:
        return float("nan")

    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
    eTestImage = sitk.BinaryErode(testImage, (1, 1, 0))
    eResultImage = sitk.BinaryErode(resultImage, (1, 1, 0))

    hTestImage = sitk.Subtract(testImage, eTestImage)
    hResultImage = sitk.Subtract(resultImage, eResultImage)

    hTestArray = sitk.GetArrayFromImage(hTestImage)
    hResultArray = sitk.GetArrayFromImage(hResultImage)

    # Convert voxel location to world coordinates. Use the coordinate system of the test image
    # np.nonzero   = elements of the boundary in numpy order (zyx)
    # np.flipud    = elements in xyz order
    # np.transpose = create tuples (x,y,z)
    # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
    testCoordinates = [
        testImage.TransformIndexToPhysicalPoint(x.tolist())
        for x in np.transpose(np.flipud(np.nonzero(hTestArray)))
    ]
    resultCoordinates = [
        testImage.TransformIndexToPhysicalPoint(x.tolist())
        for x in np.transpose(np.flipud(np.nonzero(hResultArray)))
    ]

    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]

    # Compute distances from test to result; and result to test
    dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
    dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)

    return max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))


def getLesionDetection(testImage, resultImage, vol_thr=10):
    """Lesion detection metrics, both recall and F1."""
    pred = sitk.GetArrayFromImage(resultImage)
    gt = sitk.GetArrayFromImage(testImage)

    # Connected components will give the background label 0, so subtract 1 from all results
    ccFilter = sitk.ConnectedComponentImageFilter()
    ccFilter.SetFullyConnected(True)

    # Connected components on the test image, to determine the number of true WMH.
    # And to get the overlap between detected voxels and true WMH
    ccTest = ccFilter.Execute(testImage)
    lResult = sitk.Multiply(ccTest, sitk.Cast(resultImage, sitk.sitkUInt32))

    ccTestArray = sitk.GetArrayFromImage(ccTest)  # gt lesions
    lResultArray = sitk.GetArrayFromImage(lResult)  # detected lesions

    # recall = (number of detected WMH) / (number of true WMH)
    nWMH = len(np.unique(ccTestArray)) - 1
    if nWMH == 0:
        recall = 1.0
    else:
        recall = float(len(np.unique(lResultArray)) - 1) / nWMH

    # Connected components of results, to determine number of detected lesions
    ccResult = ccFilter.Execute(resultImage)
    lTest = sitk.Multiply(ccResult, sitk.Cast(testImage, sitk.sitkUInt32))

    ccResultArray = sitk.GetArrayFromImage(ccResult)
    lTestArray = sitk.GetArrayFromImage(lTest)

    # precision = (number of detections that intersect with WMH) / (number of all detections)
    nDetections = len(np.unique(ccResultArray)) - 1
    if nDetections == 0:
        precision = 1.0
    else:
        precision = float(len(np.unique(lTestArray)) - 1) / nDetections

    f1 = f1_score(precision, recall)

    vol_recall = calc_vol_recall(ccTestArray, pred)
    vol_precision = calc_vol_recall(ccResultArray, gt)

    small_lesion_recall = vol_recall.query(f"true_vol <= {vol_thr}").vol_recall.mean()
    small_lesion_precision = vol_precision.query(
        f"true_vol <= {vol_thr}"
    ).vol_recall.mean()
    small_lesion_f1 = f1_score(small_lesion_precision, small_lesion_recall)

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "small_lesion_recall": small_lesion_recall,
        "small_lesion_precision": small_lesion_precision,
        "small_lesion_f1": small_lesion_f1,
    }, ccTestArray


def perivent_deep_wmh(testImage, resultImage, mask):
    def get_cc_recall_pv_deep(testImage, resultImage, vent_mask):
        # Connected components will give the background label 0, so subtract 1 from all results
        ccFilter = sitk.ConnectedComponentImageFilter()
        ccFilter.SetFullyConnected(True)

        # Connected components on the test image, to determine the number of true WMH.
        # And to get the overlap between detected voxels and true WMH
        ccTest = ccFilter.Execute(testImage)
        lResult = sitk.Multiply(
            ccTest, sitk.Cast(resultImage, sitk.sitkUInt32)
        )  # found gt

        ccTestArray = sitk.GetArrayFromImage(ccTest)  # gt lesions
        lResultArray = sitk.GetArrayFromImage(lResult)  # detected lesions

        # stats
        gt = set(np.unique(ccTestArray)[1:])  # all annotated lesions
        found = set(np.unique(lResultArray)[1:])  # found lesions

        # separate periventricular from deep white matter hyperintensities
        pv_gt = set(np.unique(vent_mask * ccTestArray)[1:])  # inds for det. pv lesions
        deep_gt = gt - pv_gt  # inds for det. deep lesions

        # all found pv lesions / all pv lesions
        if len(pv_gt) == 0:
            recall_pv = 0.0
        else:
            recall_pv = len(pv_gt & found) / len(pv_gt)
        # all found deep lesions / all deep lesions
        if len(deep_gt) == 0:
            recall_deep = 0.0
        else:
            recall_deep = len(deep_gt & found) / len(deep_gt)

        # recall = len(gt & found) / len(gt)  # all found lesions / all lesions
        # assert recall == len(found) / len(gt)  # same as above
        #
        # c = np.where(np.isin(ccTestArray, list(pv_gt)), 1, np.zeros_like(ccTestArray))
        # c = np.where(np.isin(ccTestArray, list(deep_gt)), 2, c)
        # c = c + vent_mask * 3
        # c = np.swapaxes(c, 0, 2)
        #
        # import nibabel as nib
        #
        # subj = "1b337ffb-297a-4ce9-b015-a3025c3699ec"
        #
        # m = nib.load(
        #     f"/home/rassmanns/diffusion/flairsyn/output/original/inference_wmh/{subj}/mask.nii.gz"
        # )
        # cni = nib.Nifti1Image(c.astype(np.uint8), m.affine)
        # nib.save(
        #     cni,
        #     f"/home/rassmanns/Desktop/{subj}_cc.nii.gz",
        # )

        return recall_deep, recall_pv

    vent_mask = np.isin(mask, [4, 43])  # ventricle labels
    vent_mask = binary_dilation(vent_mask, iterations=1)

    recall_deep, recall_pv = get_cc_recall_pv_deep(testImage, resultImage, vent_mask)
    precision_deep, precision_pv = get_cc_recall_pv_deep(
        resultImage, testImage, vent_mask
    )
    return {
        "recall_deep": recall_deep,
        "precision_deep": precision_deep,
        "f1_deep": f1_score(precision_deep, recall_deep),
        "recall_pv": recall_pv,
        "precision_pv": precision_pv,
        "f1_pv": f1_score(precision_pv, recall_pv),
    }


def f1_score(precision, recall):
    return (
        2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )


def calc_vol_recall(ccTestArray, pred):
    # compare ccTestArray (gt lesions) and lResultArray (detected lesions)
    gt = ccTestArray
    if gt.max() == 1:  # no lesions, only background
        return pd.DataFrame({"true_vol": [0], "found_vol": [0], "vol_recall": [1.0]})
    true = np.histogram(gt, bins=gt.max() - 1, range=(1, gt.max()))[0]
    found = gt * (pred > 0)
    if found.max() == 1:  # no lesions, only background
        return pd.DataFrame(
            {
                "true_vol": true,
                "found_vol": [0] * len(true),
                "vol_recall": [0] * len(true),
            }
        )
    found = np.histogram(found, bins=gt.max() - 1, range=(1, gt.max()))[0]
    rec = found / true
    df = pd.DataFrame({"true_vol": true, "found_vol": found, "vol_recall": rec})
    df = df.iloc[1:]  # remove background CC
    assert np.all(df.vol_recall <= 1.0)
    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("pred_path", type=str)
    parser.add_argument(
        "--pred_file_name", type=str, default="pred_flair_wmh_seg.nii.gz"
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="/home/rassmanns/diffusion/flairsyn/output/original/inference_wmh",
    )
    parser.add_argument(
        "--gt_file_name", type=str, default="wmhs_resampled.nii.gz"
    )  # could use the segmentation on the acquired gt images here ("flair_wmh_seg.nii.gz")
    parser.add_argument(
        "--no_hd", action="store_true", help="don't compute Hausdorff distance"
    )
    parser.add_argument(
        "--no_contrast", action="store_true", help="don't compute contrast"
    )
    parser.add_argument(
        "--pv_sep",
        action="store_true",
        help="separate metrics for periventricular and deep WMHs",
    )

    parser.add_argument(
        "-pvs",
        action="store_true",
        help="shortcut to set the filenames and paths to pvs task",
    )

    args = parser.parse_args()

    if args.pvs:
        args.pred_file_name = "pred_flair_n4_pvs_seg.nii.gz"
        args.gt_file_name = "pvs_resampled.nii.gz"
        args.gt_path = (
            "/home/rassmanns/diffusion/flairsyn/output/original/inference_pvs"
        )
        args.no_contrast = True
        # args.no_hd = True

    df = acc_metrics(
        args.pred_path,
        pred_file=args.pred_file_name,
        gt_path=args.gt_path,
        gt_file=args.gt_file_name,
        hd=not args.no_hd,
        contrast=not args.no_contrast,
        pv_sep=args.pv_sep,
    )

    if args.pred_file_name not in [
        "pred_flair_wmh_seg.nii.gz",
        "pred_flair_pvs_seg.nii.gz",
    ]:
        suffix = "-" + args.pred_file_name.replace(".nii.gz", "")
    else:
        suffix = ""
    path = f"{args.pred_path}{suffix}_lesion_metrics_subject_wise.csv"
    df.to_csv(path, index=False)

    path = f"{args.pred_path}{suffix}_lesion_metrics.csv"
    agg = (
        df.groupby("subject_ID")
        .mean()
        .reset_index()
        .drop(columns=["subject_ID"])
        .agg(["mean", "std"])
    )
    agg.to_csv(path, index=False)
