"""
Flow Matching for perturbation prediction (PCA latent space).

Uses the flow_matching library (Facebook Research) for conditional OT paths
and ODE-based sampling.
"""

from functools import partial

import torch
import torch.nn.functional as F
from einops import reduce
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from utils import scale_to_diffusion_space, scale_to_real_space


class WrappedModel(ModelWrapper):
    """Wraps model for the flow_matching ODESolver interface."""

    def forward(self, x, t, z_emb=None, **extras):
        return self.model(x=x, t=t, z_emb=z_emb, **extras)


class FlowMatching1D(torch.nn.Module):
    """
    Conditional OT Flow Matching model.

    Parameters
    ----------
    model : nn.Module
        The DiffusionAutoEncoder / DiT model.
    seq_length : int
        Dimensionality of the PCA space.
    sampling_timesteps : int
        Number of ODE steps for sampling.
    step_size : float
        ODE solver step size.
    realspace_min, realspace_max : float
        PCA space bounds for normalization to [-1, 1].
    """

    def __init__(
        self,
        model,
        *,
        seq_length,
        sampling_timesteps=250,
        step_size=0.05,
        realspace_min=-15.0,
        realspace_max=15.0,
    ):
        super().__init__()
        self.model = model
        self.channels = model.channels
        self.self_condition = model.self_condition
        self.latent_diffusion_mode = "PCA"
        self.seq_length = seq_length
        self.unconditional = getattr(model, "unconditional", False)

        self.sampling_timesteps = sampling_timesteps
        self.step_size = step_size

        self.register_buffer("vec", torch.tensor([0], dtype=torch.float32))

        # PCA normalization
        self.normalize = partial(scale_to_diffusion_space, min=realspace_min, max=realspace_max)
        self.unnormalize = partial(scale_to_real_space, min=realspace_min, max=realspace_max)

        self.path = AffineProbPath(scheduler=CondOTScheduler())

    @torch.no_grad()
    def sample(self, source_cells, z_emb, w=None, rng_state=None):
        """Generate predictions via ODE integration from source cells."""
        source_cells = torch.tensor(source_cells, device=self.vec.device).float()
        source_cells = source_cells.unsqueeze(dim=1)  # add channel dim
        source_cells = self.normalize(source_cells)

        T = torch.linspace(0, 1, self.sampling_timesteps, device=source_cells.device)
        wrapped_vf = WrappedModel(self.model)

        solver = ODESolver(velocity_model=wrapped_vf)
        sol = solver.sample(
            time_grid=T,
            x_init=source_cells,
            method="midpoint",
            step_size=self.step_size,
            return_intermediates=False,
            z_emb=z_emb,
        )

        sol = self.unnormalize(sol)
        sol = sol.squeeze(dim=1)  # remove channel dim
        return sol

    def forward(self, source_cells, target_cells, z_emb=None, **kwargs):
        """
        Compute flow matching loss.

        Returns dict with 'loss' key.
        """
        device = target_cells.device
        x_0 = self.normalize(source_cells)
        x_1 = self.normalize(target_cells)

        t = torch.rand(x_0.shape[0], device=device)
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        x_t = path_sample.x_t.unsqueeze(dim=1)
        vector_field = path_sample.dx_t.unsqueeze(dim=1)

        pred = self.model(x_t, path_sample.t, z_emb=z_emb, x_self_cond=None)

        loss = F.mse_loss(pred, vector_field, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")
        loss = loss.mean()

        return {"loss": loss}
