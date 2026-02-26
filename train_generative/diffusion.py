"""
Gaussian Diffusion for perturbation prediction (PCA latent space).

Implements DDPM forward process, reverse sampling, and DDIM sampling.
The AEGaussianDiffusion class handles the source-conditioned case where
source cells are stacked as a second channel alongside noise.

Adopted from: https://github.com/lucidrains/denoising-diffusion-pytorch
"""

import math
from collections import namedtuple
from functools import partial
from random import random

import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce
from tqdm.auto import tqdm

from utils import scale_to_diffusion_space, scale_to_real_space

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def identity(t, *args, **kwargs):
    return t


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# ---------------------------------------------------------------------------
# Beta schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    return torch.linspace(scale * 0.0001, scale * 0.02, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sqrt_beta_schedule(timesteps):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = 1 - torch.sqrt(x / timesteps + 0.0001)
    alphas_cumprod = torch.clip(alphas_cumprod, 0, 1)
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# ---------------------------------------------------------------------------
# Base Gaussian Diffusion
# ---------------------------------------------------------------------------

class GaussianDiffusion1D(torch.nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps=1000,
        sampling_timesteps=None,
        objective="pred_noise",
        beta_schedule="cosine",
        ddim_sampling_eta=0.0,
        is_ddim_sampling=False,
        realspace_min=-15.0,
        realspace_max=15.0,
    ):
        super().__init__()
        self.model = model
        self.channels = model.channels
        self.self_condition = model.self_condition
        self.latent_diffusion_mode = "PCA"
        self.seq_length = seq_length
        self.objective = objective
        self.unconditional = getattr(model, "unconditional", False)

        assert objective in {"pred_noise", "pred_x0", "pred_v"}

        schedule_fn = {"linear": linear_beta_schedule, "cosine": cosine_beta_schedule, "sqrt": sqrt_beta_schedule}
        betas = schedule_fn[beta_schedule](timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.num_timesteps = int(betas.shape[0])
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.is_ddim_sampling = is_ddim_sampling
        self.ddim_sampling_eta = ddim_sampling_eta

        buf = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        buf("betas", betas)
        buf("alphas_cumprod", alphas_cumprod)
        buf("alphas_cumprod_prev", alphas_cumprod_prev)
        buf("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        buf("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        buf("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        buf("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        buf("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        buf("posterior_variance", posterior_variance)
        buf("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        buf("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        buf("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        snr = alphas_cumprod / (1 - alphas_cumprod)
        if objective == "pred_noise":
            loss_weight = torch.ones_like(snr)
        elif objective == "pred_x0":
            loss_weight = snr
        elif objective == "pred_v":
            loss_weight = snr / (snr + 1)
        buf("loss_weight", loss_weight)

        # PCA normalization
        self.normalize = partial(scale_to_diffusion_space, min=realspace_min, max=realspace_max)
        self.unnormalize = partial(scale_to_real_space, min=realspace_min, max=realspace_max)
        self.clamp_min = -1.0
        self.clamp_max = 1.0

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start

    def predict_start_from_v(self, x_t, t, v):
        return extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v

    def q_posterior(self, x_start, x_t, t):
        mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise


# ---------------------------------------------------------------------------
# AE Gaussian Diffusion (source-conditioned)
# ---------------------------------------------------------------------------

class AEGaussianDiffusion(GaussianDiffusion1D):
    """
    Gaussian diffusion conditioned on source cells.

    The source cell is stacked as a second channel alongside the noised target,
    giving in_channels=2.  At sampling time, noise + source are stacked the same way.
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def model_predictions(self, x, t, z_emb, x_self_cond=None, clip_x_start=False, rederive_pred_noise=False, w=None):
        model_output = self.model(x, t, z_emb, x_self_cond)
        maybe_clip = partial(torch.clamp, min=self.clamp_min, max=self.clamp_max) if clip_x_start else identity

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == "pred_x0":
            x_start = maybe_clip(model_output)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == "pred_v":
            x_start = maybe_clip(self.predict_start_from_v(x, t, model_output))
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, z_emb=None, x_self_cond=None, clip_denoised=True, w=None):
        preds = self.model_predictions(x, t, z_emb, x_self_cond=x_self_cond, w=w)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(self.clamp_min, self.clamp_max)
        mean, var, log_var = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return mean, var, log_var, x_start

    @torch.no_grad()
    def p_sample(self, x, t, x_self_cond=None, z_emb=None, clip_denoised=True, w=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        mean, _, log_var, x_start = self.p_mean_variance(x=x, t=batched_times, z_emb=z_emb, x_self_cond=x_self_cond, clip_denoised=clip_denoised, w=w)
        noise = torch.randn_like(x) if t > 0 else 0.0
        return mean + (0.5 * log_var).exp() * noise, x_start

    @torch.no_grad()
    def ddim_sample(self, source_cells, z_emb, clip_denoised=True, rng_state=None, w=None):
        batch, device = source_cells.shape[0], self.betas.device
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        shape = np.array(source_cells.shape)
        noise = rng_state.standard_normal(shape)
        noise = torch.tensor(noise, device=device).float()
        img = torch.stack([noise, source_cells], dim=1)

        x_start = None
        for time, time_next in tqdm(time_pairs, desc="DDIM sampling"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, z_emb, self_cond, clip_x_start=clip_denoised, w=w)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        img = img[:, 0, :]
        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def p_sample_loop(self, source_cells=None, z_emb=None, w=None, rng_state=None):
        device = self.betas.device
        shape = np.array(source_cells.shape)
        noise = rng_state.standard_normal(shape)
        noise = torch.tensor(noise, device=device).float()
        img = torch.stack([noise, source_cells], dim=1)

        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc="DDPM sampling", total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond, z_emb=z_emb, w=w)
            img[:, 1, :] = source_cells  # keep source cells unchanged

        img = img[:, 0, :]
        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def sample(self, source_cells, z_emb, w=None, rng_state=None):
        device = self.betas.device
        source_cells = torch.tensor(source_cells, device=device).float()
        source_cells = self.normalize(source_cells)

        # Pass source cells via z_emb and zero out the source channel
        z_emb = [z_emb, source_cells]
        source_cells = torch.zeros_like(source_cells, device=device)

        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(source_cells=source_cells, z_emb=z_emb, w=w, rng_state=rng_state)

    def p_losses(self, target_cells, t, source_cells, z_emb=None, noise=None):
        if noise is None:
            noise = torch.randn_like(target_cells)

        noised_target = self.q_sample(x_start=target_cells, t=t, noise=noise)
        x_input = torch.stack([noised_target, source_cells], dim=1)

        model_out = self.model(x_input, t, z_emb=z_emb, x_self_cond=None)
        model_out = model_out[:, 0, :]  # only the target prediction channel

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = target_cells
        elif self.objective == "pred_v":
            target = self.predict_v(target_cells, t, noise)

        loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return {"loss": loss.mean()}

    def forward(self, source_cells, target_cells, z_emb=None, **kwargs):
        b, device = target_cells.shape[0], target_cells.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        target_cells = self.normalize(target_cells)
        source_cells = self.normalize(source_cells)
        source_cells = torch.zeros_like(source_cells, device=device)

        return self.p_losses(target_cells, t, source_cells=source_cells, z_emb=z_emb)
