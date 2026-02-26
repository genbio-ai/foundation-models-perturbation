"""
Schrodinger Bridge for perturbation prediction (PCA latent space).

Uses torchcfm (SchrodingerBridgeConditionalFlowMatcher) for the forward
process and supports both SDE (torchsde) and ODE (torchdyn) sampling.
"""

from functools import partial

import torch
import torch.nn.functional as F
import torchsde
from einops import reduce
from torchdyn.core import NeuralODE
from torchcfm.conditional_flow_matching import SchrodingerBridgeConditionalFlowMatcher

from utils import scale_to_diffusion_space, scale_to_real_space


class SchrodingerBridge1D(torch.nn.Module):
    """
    Schrodinger Bridge model for perturbation prediction.

    The model predicts both a velocity field and a score, with the output
    channels split in half (out_channels = 2 * in_channels).

    Parameters
    ----------
    model : nn.Module
        The DiffusionAutoEncoder / DiT model (must have out_channels = 2).
    seq_length : int
        Dimensionality of the PCA space.
    sampling_timesteps : int
        Number of SDE/ODE steps for sampling (>= 100).
    sigma : float
        Schrodinger Bridge variance parameter.
    use_sde_sampling : bool
        Whether to use SDE (True) or ODE (False) for sampling.
    realspace_min, realspace_max : float
        PCA space bounds for normalization to [-1, 1].
    """

    def __init__(
        self,
        model,
        *,
        seq_length,
        sampling_timesteps=250,
        sigma=1.0,
        use_sde_sampling=True,
        realspace_min=-15.0,
        realspace_max=15.0,
    ):
        super().__init__()
        assert sampling_timesteps >= 100, "sampling_timesteps must be >= 100"

        self.model = model
        self.channels = model.channels
        self.self_condition = model.self_condition
        self.latent_diffusion_mode = "PCA"
        self.seq_length = seq_length
        self.unconditional = getattr(model, "unconditional", False)

        self.sigma = sigma
        self.sampling_timesteps = sampling_timesteps
        self.use_sde_sampling = use_sde_sampling

        self.register_buffer("vec", torch.tensor([0], dtype=torch.float32))

        # PCA normalization
        self.normalize = partial(scale_to_diffusion_space, min=realspace_min, max=realspace_max)
        self.unnormalize = partial(scale_to_real_space, min=realspace_min, max=realspace_max)

        self.FM = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)

        print(f"SchrodingerBridge1D: sigma={sigma}, sde_sampling={use_sde_sampling}, steps={sampling_timesteps}")

    @torch.no_grad()
    def sample(self, source_cells, z_emb, w=None, rng_state=None):
        if self.use_sde_sampling:
            return self._sample_sde(source_cells, z_emb)
        else:
            return self._sample_ode(source_cells, z_emb)

    @torch.no_grad()
    def _sample_sde(self, source_cells, z_emb):
        """SDE sampling using torchsde."""
        source_cells = torch.tensor(source_cells, device=self.vec.device).float()
        source_cells = self.normalize(source_cells)
        device = source_cells.device

        model_ref = self.model
        sigma_val = self.sigma

        class SDEWrapper(torch.nn.Module):
            noise_type = "diagonal"
            sde_type = "ito"

            def __init__(self, model, sigma):
                super().__init__()
                self.model = model
                self.sigma = sigma

            def f(self, t, y):
                pred = self.model(y.unsqueeze(dim=1), t, z_emb=z_emb)
                velo, score = pred.chunk(2, dim=1)
                velo = velo.squeeze(dim=1)
                score = score.squeeze(dim=1)
                return velo + score

            def g(self, t, y):
                return torch.ones_like(y) * self.sigma

        model_ref.eval()
        sde = SDEWrapper(model_ref, sigma=sigma_val).to(device)

        with torch.no_grad():
            traj = torchsde.sdeint(
                sde,
                source_cells,
                ts=torch.linspace(0, 1, self.sampling_timesteps, device=device),
                method="euler",
            )

        sol = traj[-1]
        sol = self.unnormalize(sol)
        return sol

    @torch.no_grad()
    def _sample_ode(self, source_cells, z_emb):
        """ODE sampling using torchdyn NeuralODE."""
        source_cells = torch.tensor(source_cells, device=self.vec.device).float()
        source_cells = self.normalize(source_cells)
        device = source_cells.device

        model_ref = self.model

        class ODEWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, t, x, *args, **kwargs):
                x_in = x.unsqueeze(dim=1)
                pred = self.model(x_in, t, z_emb=z_emb)
                vector_field, _score = pred.chunk(2, dim=1)
                return vector_field.squeeze(dim=1)

        ode_model = ODEWrapper(model_ref)
        ode_model.eval()
        node = NeuralODE(ode_model, solver="euler", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

        with torch.no_grad():
            traj = node.trajectory(
                source_cells,
                t_span=torch.linspace(0, 1, self.sampling_timesteps, device=device),
            )

        sol = traj[-1]
        sol = self.unnormalize(sol)
        return sol

    def forward(self, source_cells, target_cells, z_emb=None, **kwargs):
        """
        Compute Schrodinger Bridge loss (flow + score matching).

        Returns dict with 'loss', 'flow_loss', 'score_loss' keys.
        """
        x_0 = self.normalize(source_cells)
        x_1 = self.normalize(target_cells)

        t, noise_xt, ut, eps = self.FM.sample_location_and_conditional_flow(x_0, x_1, return_noise=True)
        lambda_t = self.FM.compute_lambda(t)

        noise_xt = noise_xt.unsqueeze(dim=1)
        ut = ut.unsqueeze(dim=1)
        eps = eps.unsqueeze(dim=1)

        pred = self.model(noise_xt, t, z_emb=z_emb, x_self_cond=None)
        pred_vf, pred_score = pred.chunk(2, dim=1)

        flow_loss = torch.mean((pred_vf - ut) ** 2)
        score_loss = torch.mean((lambda_t[:, None] * pred_score + eps) ** 2)
        loss = flow_loss + score_loss

        return {
            "loss": loss,
            "flow_loss": flow_loss.item(),
            "score_loss": score_loss.item(),
        }
