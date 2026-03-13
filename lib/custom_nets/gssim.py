# based on https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/metrics/ssim.py

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from monai.metrics.regression import RegressionMetric
from monai.utils import MetricReduction, StrEnum, convert_data_type, ensure_tuple_rep
from monai.utils.type_conversion import convert_to_dst_type
from generative.metrics.ssim import _gaussian_kernel


class KernelType(StrEnum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"


class GSSIMMetric(RegressionMetric):
    r"""
    Computes the Structural Similarity Index Measure (SSIM).

    .. math::
        \operatorname {GSSIM}(x,y) =\frac {(2 \mu_x \mu_y + c_1)(2 \sigma_{xy} + c_2)}{((\mu_x^2 + \
                \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}

    For more info, visit
        https://vicuesoft.com/glossary/term/ssim-ms-ssim/

    GSSIM reference paper:
        Wang, Zhou, et al. "Image quality assessment: from error visibility to structural
        similarity." IEEE transactions on image processing 13.4 (2004): 600-612.

    Args:
        spatial_dims: number of spatial dimensions of the input images.
        data_range: value range of input images. (usually 1.0 or 255)
        kernel_type: type of kernel, can be "gaussian" or "uniform".
        kernel_size: size of kernel
        kernel_sigma: standard deviation for Gaussian kernel.
        k1: stability constant used in the luminance denominator
        k2: stability constant used in the contrast denominator
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans)
    """

    def __init__(
        self,
        spatial_dims: int,
        data_range: float = 1.0,
        kernel_type: KernelType | str = KernelType.GAUSSIAN,
        kernel_size: int | Sequence[int, ...] = 11,
        kernel_sigma: float | Sequence[float, ...] = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)

        self.spatial_dims = spatial_dims
        self.data_range = data_range
        self.kernel_type = kernel_type

        if not isinstance(kernel_size, Sequence):
            kernel_size = ensure_tuple_rep(kernel_size, spatial_dims)
        self.kernel_size = kernel_size

        if not isinstance(kernel_sigma, Sequence):
            kernel_sigma = ensure_tuple_rep(kernel_sigma, spatial_dims)
        self.kernel_sigma = kernel_sigma

        self.k1 = k1
        self.k2 = k2

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted image.
                It must be a 2D or 3D batch-first tensor [B,C,H,W] or [B,C,H,W,D].
            y: Reference image.
                It must be a 2D or 3D batch-first tensor [B,C,H,W] or [B,C,H,W,D].

        Raises:
            ValueError: when `y_pred` is not a 2D or 3D image.
        """
        dims = y_pred.ndimension()
        if self.spatial_dims == 2 and dims != 4:
            raise ValueError(
                f"y_pred should have 4 dimensions (batch, channel, height, width) when using {self.spatial_dims} "
                f"spatial dimensions, got {dims}."
            )

        if self.spatial_dims == 3 and dims != 5:
            raise ValueError(
                f"y_pred should have 4 dimensions (batch, channel, height, width, depth) when using {self.spatial_dims}"
                f" spatial dimensions, got {dims}."
            )

        ssim_value_full_image, _ = compute_ssim_and_cs(
            y_pred=y_pred,
            y=y,
            spatial_dims=self.spatial_dims,
            data_range=self.data_range,
            kernel_type=self.kernel_type,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            k1=self.k1,
            k2=self.k2,
        )

        ssim_per_batch: torch.Tensor = ssim_value_full_image.view(
            ssim_value_full_image.shape[0], -1
        ).mean(1, keepdim=True)

        return ssim_per_batch


def _image_gradient_3d(x, return_components=False):
    """
    Compute the gradient of an image using a Sobel operator in 3D
    """
    # Define filters
    dx = (
        torch.tensor(
            [
                [
                    [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                    [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
                    [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                ]
            ],
            dtype=torch.float32,
        )
        .view(1, 1, 3, 3, 3)
        .to(x.device)
    )
    dy = dx.permute(0, 1, 3, 2, 4)
    dz = dx.permute(0, 1, 4, 3, 2)

    # Filter
    x_dx = F.conv3d(x, dx, padding=1)
    x_dy = F.conv3d(x, dy, padding=1)
    x_dz = F.conv3d(x, dz, padding=1)
    magn = torch.sqrt(x_dx**2 + x_dy**2 + x_dz**2)
    if return_components:
        return magn, x_dx, x_dy, x_dz
    else:
        return magn


def _image_gradient_2d(x, return_components=False):
    """
    Compute the gradient of an image using a Sobel operator.
    """
    # Define filters
    dx = (
        torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .to(x.device)
    )
    dy = dx.permute(0, 1, 3, 2)

    # Filter
    x_dx = F.conv2d(x, dx, padding=1)
    x_dy = F.conv2d(x, dy, padding=1)

    magn = torch.sqrt(x_dx**2 + x_dy**2)
    if return_components:
        return magn, x_dx, x_dy
    else:
        return magn


def compute_ssim_and_cs(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    spatial_dims: int,
    data_range: float = 1.0,
    kernel_type: KernelType | str = KernelType.GAUSSIAN,
    kernel_size: Sequence[int, ...] = 11,
    kernel_sigma: Sequence[float, ...] = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Function to compute the Structural Similarity Index Measure (SSIM) and Contrast Sensitivity (CS) for a batch
    of images.

    Args:
        y_pred: batch of predicted images with shape (batch_size, channels, spatial_dim1, spatial_dim2[, spatial_dim3])
        y: batch of target images with shape (batch_size, channels, spatial_dim1, spatial_dim2[, spatial_dim3])
        spatial_dims: number of spatial dimensions of the images (2, 3)
        data_range: the data range of the images.
        kernel_type: the type of kernel to use for the SSIM computation. Can be either "gaussian" or "uniform".
        kernel_size: the size of the kernel to use for the SSIM computation.
        kernel_sigma: the standard deviation of the kernel to use for the SSIM computation.
        k1: the first stability constant.
        k2: the second stability constant.

    Returns:
        ssim: the Structural Similarity Index Measure score for the batch of images.
        cs: the Contrast Sensitivity for the batch of images.
    """
    if y.shape != y_pred.shape:
        raise ValueError(
            f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}."
        )

    y_pred = convert_data_type(y_pred, output_type=torch.Tensor, dtype=torch.float)[0]
    y = convert_data_type(y, output_type=torch.Tensor, dtype=torch.float)[0]

    num_channels = y_pred.size(1)

    if kernel_type == KernelType.GAUSSIAN:
        kernel = _gaussian_kernel(spatial_dims, num_channels, kernel_size, kernel_sigma)
    elif kernel_type == KernelType.UNIFORM:
        kernel = torch.ones((num_channels, 1, *kernel_size)) / torch.prod(
            torch.tensor(kernel_size)
        )

    kernel = convert_to_dst_type(src=kernel, dst=y_pred)[0]

    c1 = (k1 * data_range) ** 2  # stability constant for luminance
    c2 = (k2 * data_range) ** 2  # stability constant for contrast

    conv_fn = getattr(F, f"conv{spatial_dims}d")
    mu_x = conv_fn(y_pred, kernel, groups=num_channels)
    mu_y = conv_fn(y, kernel, groups=num_channels)
    luminance = (2 * mu_x * mu_y + c1) / (mu_x**2 + mu_y**2 + c1)

    grad_fun = _image_gradient_3d if spatial_dims == 3 else _image_gradient_2d
    y_pred_grad = grad_fun(y_pred)
    y_grad = grad_fun(y)

    mu_x = conv_fn(y_pred_grad, kernel, groups=num_channels)
    mu_y = conv_fn(y_grad, kernel, groups=num_channels)

    mu_xx = conv_fn(y_pred_grad * y_pred_grad, kernel, groups=num_channels)
    mu_yy = conv_fn(y_grad * y_grad, kernel, groups=num_channels)
    mu_xy = conv_fn(y_pred_grad * y_grad, kernel, groups=num_channels)

    sigma_x = mu_xx - mu_x * mu_x
    sigma_y = mu_yy - mu_y * mu_y
    sigma_xy = mu_xy - mu_x * mu_y

    contrast_sensitivity = (2 * sigma_xy + c2) / (sigma_x + sigma_y + c2)
    ssim_value_full_image = luminance * contrast_sensitivity

    return ssim_value_full_image, contrast_sensitivity
