# from https://github.com/Deep-MI/FastSurfer/blob/dev/FastSurferCNN/data_loader/conform.py at commit 7cd0161
# fmt: off

import numpy as np
from typing import Tuple

def getscale(
        data: np.ndarray,
        dst_min: float,
        dst_max: float,
        f_low: float = 0.0,
        f_high: float = 0.999
        ) -> Tuple[float, float]:
    """Get offset and scale of image intensities to robustly rescale to range dst_min..dst_max.

    Equivalent to how mri_convert conforms images.

    Parameters
    ----------
    data : np.ndarray
        image data (intensity values)
    dst_min : float
        future minimal intensity value
    dst_max : float
        future maximal intensity value
    f_low : float
        robust cropping at low end (0.0 no cropping, default)
    f_high : float
        robust cropping at higher end (0.999 crop one thousandth of high intensity voxels, default)

    Returns
    -------
    float src_min
        (adjusted) offset
    float
        scale factor

    """
    # get min and max from source
    src_min = np.min(data)
    src_max = np.max(data)

    if src_min < 0.0:
        print("WARNING: Input image has value(s) below 0.0 !")

    print("Input:    min: " + format(src_min) + "  max: " + format(src_max))

    if f_low == 0.0 and f_high == 1.0:
        return src_min, 1.0

    # compute non-zeros and total vox num
    nz = (np.abs(data) >= 1e-15).sum()
    voxnum = data.shape[0] * data.shape[1] * data.shape[2]

    # compute histogram
    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, bin_edges = np.histogram(data, histosize)

    # compute cumulative sum
    cs = np.concatenate(([0], np.cumsum(hist)))

    # get lower limit
    nth = int(f_low * voxnum)
    idx = np.where(cs < nth)

    if len(idx[0]) > 0:
        idx = idx[0][-1] + 1

    else:
        idx = 0

    src_min = idx * bin_size + src_min

    # get upper limit
    nth = voxnum - int((1.0 - f_high) * nz)
    idx = np.where(cs >= nth)

    if len(idx[0]) > 0:
        idx = idx[0][0] - 2

    else:
        print("ERROR: rescale upper bound not found")

    src_max = idx * bin_size + src_min

    # scale
    if src_min == src_max:
        scale = 1.0

    else:
        scale = (dst_max - dst_min) / (src_max - src_min)

    print(
        "rescale:  min: "
        + format(src_min)
        + "  max: "
        + format(src_max)
        + "  scale: "
        + format(scale)
        )

    return src_min, scale


def scalecrop(
        data: np.ndarray,
        dst_min: float,
        dst_max: float,
        src_min: float,
        scale: float
        ) -> np.ndarray:
    """Crop the intensity ranges to specific min and max values.

    Parameters
    ----------
    data : np.ndarray
        Image data (intensity values)
    dst_min : float
        future minimal intensity value
    dst_max : float
        future maximal intensity value
    src_min : float
        minimal value to consider from source (crops below)
    scale : float
        scale value by which source will be shifted

    Returns
    -------
    np.ndarray
        scaled image data

    """
    data_new = dst_min + scale * (data - src_min)

    # clip
    data_new = np.clip(data_new, dst_min, dst_max)
    print(
        "Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max())
        )

    return data_new


def rescale(
        data: np.ndarray,
        dst_min: float,
        dst_max: float,
        f_low: float = 0.0,
        f_high: float = 0.999
        ) -> np.ndarray:
    """Rescale image intensity values (0-255).

    Parameters
    ----------
    data : np.ndarray
        image data (intensity values)
    dst_min : float
        future minimal intensity value
    dst_max : float
        future maximal intensity value
    f_low : float
        robust cropping at low end (0.0 no cropping, default)
    f_high : float
        robust cropping at higher end (0.999 crop one thousandth of high intensity voxels, default)

    Returns
    -------
    np.ndarray
        scaled image data

    """
    src_min, scale = getscale(data, dst_min, dst_max, f_low, f_high)
    data_new = scalecrop(data, dst_min, dst_max, src_min, scale)
    return data_new