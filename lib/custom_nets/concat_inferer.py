from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import Inferer
from monai.utils import optional_import
from monai.transforms import SpatialPad, CenterSpatialCrop
from torch.cuda.amp import autocast

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")


class DiffusionInferer(Inferer):
    """
    DiffusionInferer takes a trained diffusion model and a scheduler and can be used to perform a signal forward pass
    for a training iteration, and sample from the model.


    Args:
        scheduler: diffusion scheduler.
    """

    def __init__(self, scheduler: nn.Module) -> None:
        Inferer.__init__(self)
        self.scheduler = scheduler

    def __call__(
        self,
        inputs: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        mode: str = "crossattn",
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: Input image to which noise is added.
            diffusion_model: diffusion model.
            noise: random noise, of the same shape as the input.
            timesteps: random timesteps.
            condition: Conditioning for network input.
            mode: Conditioning mode for the network.
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        noisy_image = self.scheduler.add_noise(
            original_samples=inputs, noise=noise, timesteps=timesteps
        )
        if mode == "concat":
            noisy_image = torch.cat([noisy_image, condition], dim=1).contiguous()
            condition = None
        prediction = diffusion_model(
            x=noisy_image,
            timesteps=timesteps,
            context=condition,
            class_labels=class_labels,
        )

        return prediction

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        if not scheduler:
            scheduler = self.scheduler
        image = input_noise
        conditioning = conditioning.to(image.device)
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        for t in progress_bar:
            # 1. predict noise model_output
            if mode == "concat":
                model_input = torch.cat([image, conditioning], dim=1).contiguous()
                model_output = diffusion_model(
                    x=model_input,
                    timesteps=torch.Tensor((t,)).to(input_noise.device),
                    context=None,
                    class_labels=class_labels,
                )
            else:
                model_output = diffusion_model(
                    image,
                    timesteps=torch.Tensor((t,)).to(input_noise.device),
                    context=conditioning,
                    class_labels=class_labels,
                )

            # 2. compute previous image: x_t -> x_t-1
            image, _ = scheduler.step(model_output, t, image)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)
        if save_intermediates:
            return image, intermediates
        else:
            return image

    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
        verbose: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Computes the log-likelihoods for an input.

        Args:
            inputs: input images, NxCxHxW[xD]
            diffusion_model: model to compute likelihood from
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
        """

        if not scheduler:
            scheduler = self.scheduler
        if scheduler._get_name() != "DDPMScheduler":
            raise NotImplementedError(
                f"Likelihood computation is only compatible with DDPMScheduler,"
                f" you are using {scheduler._get_name()}"
            )
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        noise = torch.randn_like(inputs).to(inputs.device)
        total_kl = torch.zeros(inputs.shape[0]).to(inputs.device)
        for t in progress_bar:
            timesteps = torch.full(inputs.shape[:1], t, device=inputs.device).long()
            noisy_image = self.scheduler.add_noise(
                original_samples=inputs, noise=noise, timesteps=timesteps
            )
            if mode == "concat":
                noisy_image = torch.cat([noisy_image, conditioning], dim=1)
                model_output = diffusion_model(
                    noisy_image, timesteps=timesteps, context=None
                )
            else:
                model_output = diffusion_model(
                    x=noisy_image, timesteps=timesteps, context=conditioning
                )
            # get the model's predicted mean,  and variance if it is predicted
            if model_output.shape[1] == inputs.shape[
                1
            ] * 2 and scheduler.variance_type in ["learned", "learned_range"]:
                model_output, predicted_variance = torch.split(
                    model_output, inputs.shape[1], dim=1
                )
            else:
                predicted_variance = None

            # 1. compute alphas, betas
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[t - 1] if t > 0 else scheduler.one
            )
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            if scheduler.prediction_type == "epsilon":
                pred_original_sample = (
                    noisy_image - beta_prod_t ** (0.5) * model_output
                ) / alpha_prod_t ** (0.5)
            elif scheduler.prediction_type == "sample":
                pred_original_sample = model_output
            elif scheduler.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * noisy_image - (
                    beta_prod_t**0.5
                ) * model_output
            # 3. Clip "predicted x_0"
            if scheduler.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample_coeff = (
                alpha_prod_t_prev ** (0.5) * scheduler.betas[t]
            ) / beta_prod_t
            current_sample_coeff = (
                scheduler.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t
            )

            # 5. Compute predicted previous sample µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            predicted_mean = (
                pred_original_sample_coeff * pred_original_sample
                + current_sample_coeff * noisy_image
            )

            # get the posterior mean and variance
            posterior_mean = scheduler._get_mean(
                timestep=t, x_0=inputs, x_t=noisy_image
            )
            posterior_variance = scheduler._get_variance(
                timestep=t, predicted_variance=predicted_variance
            )

            log_posterior_variance = torch.log(posterior_variance)
            log_predicted_variance = (
                torch.log(predicted_variance)
                if predicted_variance
                else log_posterior_variance
            )

            if t == 0:
                # compute -log p(x_0|x_1)
                kl = -self._get_decoder_log_likelihood(
                    inputs=inputs,
                    means=predicted_mean,
                    log_scales=0.5 * log_predicted_variance,
                    original_input_range=original_input_range,
                    scaled_input_range=scaled_input_range,
                )
            else:
                # compute kl between two normals
                kl = 0.5 * (
                    -1.0
                    + log_predicted_variance
                    - log_posterior_variance
                    + torch.exp(log_posterior_variance - log_predicted_variance)
                    + ((posterior_mean - predicted_mean) ** 2)
                    * torch.exp(-log_predicted_variance)
                )
            total_kl += kl.view(kl.shape[0], -1).mean(axis=1)
            if save_intermediates:
                intermediates.append(kl.cpu())

        if save_intermediates:
            return total_kl, intermediates
        else:
            return total_kl

    def _approx_standard_normal_cdf(self, x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal. Code adapted from https://github.com/openai/improved-diffusion.
        """

        return 0.5 * (
            1.0
            + torch.tanh(
                torch.sqrt(torch.Tensor([2.0 / math.pi]).to(x.device))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )

    def _get_decoder_log_likelihood(
        self,
        inputs: torch.Tensor,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
    ) -> torch.Tensor:
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image. Code adapted from https://github.com/openai/improved-diffusion.

        Args:
            input: the target images. It is assumed that this was uint8 values,
                      rescaled to the range [-1, 1].
            means: the Gaussian mean Tensor.
            log_scales: the Gaussian log stddev Tensor.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
        """
        assert inputs.shape == means.shape
        bin_width = (scaled_input_range[1] - scaled_input_range[0]) / (
            original_input_range[1] - original_input_range[0]
        )
        centered_x = inputs - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + bin_width / 2)
        cdf_plus = self._approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - bin_width / 2)
        cdf_min = self._approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            inputs < -0.999,
            log_cdf_plus,
            torch.where(
                inputs > 0.999,
                log_one_minus_cdf_min,
                torch.log(cdf_delta.clamp(min=1e-12)),
            ),
        )
        assert log_probs.shape == inputs.shape
        return log_probs


class ThickSliceInferer(DiffusionInferer):
    """
    Thick target slice inference using concatenations
    """

    def __init__(
        self,
        scheduler: nn.Module,
        slice_thickness: int = 1,
        thick_target_slices: bool = False,
    ) -> None:
        Inferer.__init__(self)
        self.scheduler = scheduler
        self.slice_thickness = slice_thickness
        self.thick_target_slices = thick_target_slices

    def get_pred_target(self, images, noise, timesteps):
        if self.scheduler.prediction_type == "epsilon":
            return noise
        if self.scheduler.prediction_type == "sample":
            return images
        if self.scheduler.prediction_type == "v_prediction":
            return self.scheduler.get_velocity(images.float(), noise.float(), timesteps)

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        guidance_sequences: torch.Tensor | None = None,
        adjacent_guidance: torch.Tensor | None = None,
        verbose: bool = True,
        mode: str = "concat",
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            guidance_sequences: guidance sequences to use for sampling, assumed to be known at inference time
            adjacent_guidance: ground truth sequences to use for sampling, usually not known (but predicted in parallel) during inference time.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
        """
        if mode not in ["concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        if torch.distributed.is_initialized() and os.getenv("LOCAL_RANK") != "0":
            verbose = False

        # check index at which to retrieve the generated target slice
        tar_ind = None  # indicator for thick input but slice output
        if self.slice_thickness > 1 and not self.thick_target_slices:
            tar_ind = self.slice_thickness // 2

        if not scheduler:
            scheduler = self.scheduler
        image = input_noise
        guidance_sequences = guidance_sequences.to(image.device)
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)

        intermediates = []
        for t in progress_bar:
            with autocast(enabled=True, dtype=dtype):
                # 0. prepare slices
                # if thick input but not thick output use adjacent slices (if provided) to simulate guidance
                if tar_ind is not None:  # thick input but single slice output
                    # retrieve image from adjacent slices
                    adj_noise = torch.randn_like(image, device=image.device)
                    image = image[:, tar_ind : tar_ind + 1]

                    if adjacent_guidance is not None:
                        # simulate guidance by from partially denoised adjacent slices
                        timestep = torch.Tensor([t] * image.shape[0]).long()
                        adj_guidance = scheduler.add_noise(
                            adjacent_guidance.float().to(image.device),
                            adj_noise,
                            timestep.to(image.device),
                        )
                    else:  # test with no additional guidance
                        adj_guidance = -torch.ones_like(adj_noise)
                    adj_guidance[:, tar_ind : tar_ind + 1] = image
                    image = adj_guidance

                x = torch.cat([image, guidance_sequences], dim=1).contiguous()

                # 1. predict noise model_output
                model_output = diffusion_model(
                    x=x,
                    timesteps=torch.Tensor((t,)).to(input_noise.device),
                )

            # 2. compute previous image: x_t -> x_t-1
            if save_intermediates and t % intermediate_steps == 0:
                sample = image
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[t - 1]
                    if t > 0
                    else self.scheduler.one
                )
                beta_prod_t = 1 - alpha_prod_t
                beta_prod_t_prev = 1 - alpha_prod_t_prev

                if self.scheduler.prediction_type == "epsilon":
                    pred_original_sample = (
                        sample - beta_prod_t ** (0.5) * model_output
                    ) / alpha_prod_t ** (0.5)
                elif self.scheduler.prediction_type == "sample":
                    pred_original_sample = model_output
                elif self.scheduler.prediction_type == "v_prediction":
                    pred_original_sample = (alpha_prod_t**0.5) * sample - (
                        beta_prod_t**0.5
                    ) * model_output

                if tar_ind is not None:
                    intermediates.append(
                        pred_original_sample[:, tar_ind : tar_ind + 1]
                    ).cpu()
                else:
                    intermediates.append(pred_original_sample.cpu())
            image, _ = scheduler.step(model_output, t, image)
            image = image.float()

        if tar_ind is not None:
            image = image[:, tar_ind : tar_ind + 1]
        if save_intermediates:
            return image, intermediates
        else:
            return image
