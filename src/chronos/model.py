import math

import torch
import torch.nn as nn


class MineStatisticsNetwork(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=-1))


def mine_lower_bound_stable(t_net: MineStatisticsNetwork, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    t = t_net(x, y)
    idx = torch.randperm(y.shape[0], device=y.device)
    t_m = t_net(x, y[idx])
    log_mean_exp = torch.logsumexp(t_m, dim=0) - math.log(max(1, y.shape[0]))
    return t.mean() - log_mean_exp


class CHRONOSModel(nn.Module):
    """Dual-stream CHRONOS backbone aligned with the paper.

    Static covariates are encoded with an MLP, dynamic history is encoded with a GRU,
    and the two streams are fused into a shared representation used by separate
    outcome heads (mu0/mu1) and a propensity head.
    """

    def __init__(
        self,
        cov_dim: int,
        prev_treat_dim: int,
        latent_dim: int = 64,
        rnn_layers: int = 1,
        dropout: float = 0.1,
        n_treatments: int = 1,
        n_outcomes: int = 1,
        static_dim: int = 3,
    ):
        super().__init__()
        self.n_treatments = n_treatments
        self.n_outcomes = n_outcomes
        self.static_dim = min(static_dim, cov_dim)
        self.dynamic_cov_dim = max(0, cov_dim - self.static_dim)

        self.static_encoder = nn.Sequential(
            nn.Linear(self.static_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )

        self.temporal_encoder = nn.GRU(
            input_size=self.dynamic_cov_dim + prev_treat_dim,
            hidden_size=latent_dim,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(latent_dim)

        self.mu0_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, n_outcomes),
        )
        self.mu1_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, n_outcomes),
        )
        self.propensity_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

    def encode(self, current_covariates: torch.Tensor, prev_treatments: torch.Tensor, hidden=None):
        static = current_covariates[:, 0, : self.static_dim]
        static_z = self.static_encoder(static)
        static_z = static_z.unsqueeze(1).expand(-1, current_covariates.size(1), -1)

        dyn_cov = current_covariates[:, :, self.static_dim :]
        dyn_x = torch.cat([dyn_cov, prev_treatments], dim=-1)
        dyn_z, h = self.temporal_encoder(dyn_x, hidden)

        fused = self.fusion(torch.cat([dyn_z, static_z], dim=-1))
        fused = self.norm(fused)
        return fused, h

    def predict_outcomes(self, z_fused: torch.Tensor):
        y0 = self.mu0_head(z_fused)
        y1 = self.mu1_head(z_fused)
        if z_fused.dim() == 2:
            y1_all = y1.unsqueeze(1)
        else:
            y1_all = y1.unsqueeze(2)
        return y0, y1_all

    def forward(self, current_covariates: torch.Tensor, prev_treatments: torch.Tensor, current_treatments: torch.Tensor, hidden=None):
        z_fused, h = self.encode(current_covariates, prev_treatments, hidden)
        y0 = self.mu0_head(z_fused)
        y1 = self.mu1_head(z_fused)

        if current_treatments.size(-1) != 1:
            raise ValueError("This CHRONOS implementation assumes one treatment code per edge-specific trial.")
        t = current_treatments[..., 0:1]
        logits = (1.0 - t) * y0 + t * y1
        e_logit = self.propensity_head(z_fused)
        return logits, z_fused, h, e_logit


class CHRONOSSingleTaskModel(CHRONOSModel):
    """CHRONOS backbone with the causal layer replaced by one prediction head.

    Static MLP, GRU and fusion are CHRONOS's. The mu0/mu1 and propensity heads need an
    exposure and are dropped; the head mirrors the outcome heads and emits one value per
    timestep: a log-risk for a survival target (higher = higher hazard, Cox loss), the
    prediction for a regression target, a logit for a binary target. The GRU is causal, so
    the output at day t only depends on days <= t.
    """

    def __init__(
        self,
        cov_dim: int,
        latent_dim: int = 64,
        rnn_layers: int = 1,
        dropout: float = 0.1,
        static_dim: int = 3,
    ):
        super().__init__(
            cov_dim=cov_dim,
            prev_treat_dim=0,
            latent_dim=latent_dim,
            rnn_layers=rnn_layers,
            dropout=dropout,
            n_treatments=1,
            n_outcomes=1,
            static_dim=static_dim,
        )
        del self.mu0_head, self.mu1_head, self.propensity_head
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, current_covariates: torch.Tensor) -> torch.Tensor:
        no_treatments = current_covariates.new_zeros(*current_covariates.shape[:2], 0)
        z_fused, _ = self.encode(current_covariates, no_treatments)
        return self.head(z_fused).squeeze(-1)


def contrastive_loss_from_batch(
    z_flat: torch.Tensor,
    tau_flat: torch.Tensor,
    *,
    max_pairs: int = 1024,
    margin: float = 1.0,
    perc: float = 30.0,
) -> torch.Tensor:
    N = z_flat.shape[0]
    if N < 2:
        return z_flat.new_tensor(0.0)

    n_pairs = min(max_pairs, N * (N - 1) // 2)
    if n_pairs < 1:
        return z_flat.new_tensor(0.0)

    i = torch.randint(0, N, size=(n_pairs,), device=z_flat.device)
    j = torch.randint(0, N, size=(n_pairs,), device=z_flat.device)
    j = torch.where(j == i, (j + 1) % N, j)

    dist_tau = torch.norm(tau_flat[i] - tau_flat[j], p=2, dim=-1)
    thr = torch.quantile(dist_tau, q=perc / 100.0)
    y = (dist_tau <= thr).float()

    dist_sq = ((z_flat[i] - z_flat[j]) ** 2).sum(dim=-1)
    dist = torch.sqrt(dist_sq + 1e-8)
    loss_sim = y * dist_sq
    loss_dis = (1.0 - y) * (torch.clamp(margin - dist, min=0.0) ** 2)
    return (loss_sim + loss_dis).mean()
