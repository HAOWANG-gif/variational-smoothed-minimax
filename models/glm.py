from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from sbi.utils import BoxUniform
from sbi.utils.metrics import c2st
from geomloss import SamplesLoss
from sbi.utils.user_input_checks import process_prior, process_simulator


# ============================================================
# Bernoulli GLM dimensions
# ============================================================
# The simulator now directly returns the raw 100-dimensional Bernoulli
# observation, rather than the 10-dimensional sbibm summary statistic.
x_dim = 100
theta_dim = 10
z_dim = 100
raw_x_dim = 100


# ============================================================
# Global device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# File loading utilities
# ============================================================
def _base_dir() -> Path:
    """Return the directory containing this file, falling back to cwd in notebooks."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def _find_required_file(filename: str) -> Path:
    """
    Search for a required sbibm file.

    Supported layouts:
        current_folder/stimulus_I.pt
        current_folder/design_matrix.pt
        current_folder/files/stimulus_I.pt
        current_folder/files/design_matrix.pt
    """
    base = _base_dir()
    candidates = [
        base / filename,
        base / "files" / filename,
        Path.cwd().resolve() / filename,
        Path.cwd().resolve() / "files" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path

    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Cannot find required file {filename!r}. I searched:\n{searched}\n\n"
        f"Please put {filename!r} in the current folder or in a subfolder named 'files'."
    )


# ============================================================
# 1. Target transform
# ============================================================
def target_transform(theta_batch: torch.Tensor) -> torch.Tensor:
    """
    Bernoulli GLM has no sign/permutation non-identifiability transform.
    """
    return theta_batch.clone()


# ============================================================
# 2. Object-oriented sample generator
# ============================================================
class SampleGenerator:
    """
    Bernoulli GLM simulator using fixed stimulus_I.pt and design_matrix.pt.

    Forward model for one theta:
        psi = design_matrix @ theta             # [100]
        z   = sigmoid(psi)                      # [100]
        y_t ~ Bernoulli(z_t), t = 1, ..., 100    # raw [100]

    Main changes requested:
        1. prior is BoxUniform([-3, 3]^10);
        2. generate_samples() directly returns raw 100-dimensional observations;
        3. MLE utilities are provided below.

    The hard forward values are Bernoulli samples. If use_st=True, the returned
    tensor uses a straight-through estimator so gradients can pass through a
    sigmoid relaxation while keeping the same hard forward values.
    """

    _stimulus_I: Optional[torch.Tensor] = None
    _design_matrix: Optional[torch.Tensor] = None

    # Uniform prior theta_j ~ U(-3, 3), j = 1, ..., 10.
    _prior_low: Optional[torch.Tensor] = None
    _prior_high: Optional[torch.Tensor] = None

    # Compatibility fields used by some older code.
    _prior_mean: Optional[torch.Tensor] = None
    _prior_cov: Optional[torch.Tensor] = None
    _prior_cholesky: Optional[torch.Tensor] = None

    def __init__(
        self,
        theta: torch.Tensor,
        n_samples: int,
        seed: Optional[int] = None,
        tau: float = 0.1,
        use_st: bool = True,
    ):
        if SampleGenerator._stimulus_I is None or SampleGenerator._design_matrix is None:
            SampleGenerator._setup_statics()

        self.theta = torch.as_tensor(theta, dtype=torch.float32, device=device)

        # Compatible with theta.shape = [10] or [1, 10]
        if self.theta.dim() == 2 and self.theta.shape[0] == 1:
            self.theta = self.theta.squeeze(0)

        if self.theta.shape != (theta_dim,):
            raise ValueError(
                f"theta shape error: expected [10] or [1, 10], got {tuple(self.theta.shape)}"
            )

        self.n_samples = int(n_samples)
        self.seed = seed
        self.tau = float(tau)
        self.use_st = bool(use_st)

        self._set_random_seed()

    def _set_random_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

    @classmethod
    def _setup_statics(cls):
        """
        Load fixed stimulus_I and design_matrix, and initialise U([-3, 3]^10).

        This keeps the source-code simulator convention that stimulus_I.pt and
        design_matrix.pt are loaded from disk, rather than regenerated inside
        the simulator.
        """
        stimulus_path = _find_required_file("stimulus_I.pt")
        design_path = _find_required_file("design_matrix.pt")

        cls._stimulus_I = torch.load(stimulus_path, map_location=device).to(device).float()
        cls._design_matrix = torch.load(design_path, map_location=device).to(device).float()

        if cls._stimulus_I.ndim != 1:
            cls._stimulus_I = cls._stimulus_I.reshape(-1)

        if cls._stimulus_I.shape[0] != raw_x_dim:
            raise ValueError(
                f"stimulus_I should have length {raw_x_dim}, got {tuple(cls._stimulus_I.shape)}"
            )

        if cls._design_matrix.shape != (raw_x_dim, theta_dim):
            raise ValueError(
                f"design_matrix should have shape {(raw_x_dim, theta_dim)}, "
                f"got {tuple(cls._design_matrix.shape)}"
            )

        cls._prior_low = torch.full((theta_dim,), -3.0, dtype=torch.float32, device=device)
        cls._prior_high = torch.full((theta_dim,), 3.0, dtype=torch.float32, device=device)

        # Compatibility statistics for U(-3, 3): mean = 0, var = (6^2)/12 = 3.
        cls._prior_mean = torch.zeros(theta_dim, dtype=torch.float32, device=device)
        cls._prior_cov = torch.eye(theta_dim, dtype=torch.float32, device=device) * 3.0
        cls._prior_cholesky = torch.linalg.cholesky(
            cls._prior_cov + 1e-6 * torch.eye(theta_dim, device=device)
        )

    def _unpack_theta_core(self) -> torch.Tensor:
        return self.theta

    @staticmethod
    def _summary_from_raw(y: torch.Tensor) -> torch.Tensor:
        """
        Optional helper: compute the original sbibm 10-dimensional summary.
        This is not the default output anymore.

        Input:
            y: [B, 100]
        Output:
            summary: [B, 10]
        """
        if SampleGenerator._stimulus_I is None:
            SampleGenerator._setup_statics()

        if y.ndim != 2 or y.shape[1] != raw_x_dim:
            raise ValueError(f"raw y should have shape [B, 100], got {tuple(y.shape)}")

        num_spikes = torch.sum(y, dim=1, keepdim=True)  # [B, 1]
        sta_full = F.conv1d(
            y.reshape(y.shape[0], 1, -1),
            SampleGenerator._stimulus_I.reshape(1, 1, -1),
            padding=8,
        ).squeeze(1)  # [B, 17]
        sta = sta_full[:, -9:]  # [B, 9]
        return torch.cat((num_spikes, sta), dim=1)  # [B, 10]

    def generate_samples(
        self,
        eps: Optional[torch.Tensor] = None,
        return_summary: bool = False,
        return_both: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate raw Bernoulli GLM samples for the fixed theta.

        eps:
            None: internally generate U(0, 1) noise with shape [n_samples, 100].
            Tensor: externally supplied U(0, 1) noise with shape [n_samples, 100].

        return_summary:
            False: return raw 100-dimensional spike train, shape [n_samples, 100].
            True: return original 10-dimensional summary, shape [n_samples, 10].

        return_both:
            True: return (summary, raw).
        """
        theta = self._unpack_theta_core()

        # Source model: psi = design_matrix @ theta, z = sigmoid(psi)
        psi = torch.matmul(SampleGenerator._design_matrix, theta)  # [100]
        z = torch.sigmoid(psi).reshape(1, -1).expand(self.n_samples, -1)  # [B, 100]

        if eps is None:
            eps = torch.rand(self.n_samples, raw_x_dim, device=device)
        else:
            eps = torch.as_tensor(eps, dtype=torch.float32, device=device)
            expected_shape = (self.n_samples, raw_x_dim)
            if tuple(eps.shape) != expected_shape:
                raise ValueError(
                    f"eps shape error: expected {expected_shape}, got {tuple(eps.shape)}"
                )

        eps = torch.clamp(eps, min=1e-7, max=1.0 - 1e-7)

        # Hard path: y = (U < sigmoid(psi)).float().
        y_hard = (eps < z).float()

        if self.use_st:
            # Differentiable relaxation with the same hard forward value.
            # eps < sigmoid(psi) <=> psi - logit(eps) > 0.
            logit_eps = torch.log(eps) - torch.log1p(-eps)
            y_soft = torch.sigmoid((psi.reshape(1, -1) - logit_eps) / self.tau)
            y = y_hard.detach() - y_soft.detach() + y_soft
        else:
            y = y_hard

        raw = y.reshape(self.n_samples, raw_x_dim)
        summary = self._summary_from_raw(raw)

        if return_both:
            return summary, raw

        if return_summary:
            return summary.reshape(self.n_samples, theta_dim)

        return raw

    @staticmethod
    def generate_noise(n_samples: int, seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate U(0, 1) noise for the 100 Bernoulli variables.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        return torch.rand(int(n_samples), raw_x_dim, device=device)

    @staticmethod
    def static_sample_model(
        theta: torch.Tensor,
        n_samples: int,
        seed: Optional[int] = None,
        eps: Optional[torch.Tensor] = None,
        return_summary: bool = False,
        return_both: bool = False,
        use_st: bool = True,
    ):
        """
        Static sampling interface for OURS / MWE / AdaIMLE / ANND etc.
        Default output is now [n_samples, 100].
        """
        generator = SampleGenerator(
            theta=theta,
            n_samples=n_samples,
            seed=seed,
            use_st=use_st,
        )
        return generator.generate_samples(
            eps=eps,
            return_summary=return_summary,
            return_both=return_both,
        )

    @classmethod
    def _map_unit_to_prior(cls, sample_unit: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map [0, 1]^10 samples to the BoxUniform prior U([-3, 3]^10).
        """
        if cls._prior_low is None or cls._prior_high is None:
            cls._setup_statics()

        if isinstance(sample_unit, torch.Tensor):
            u = sample_unit.detach().clone().to(device).float()
        else:
            u = torch.as_tensor(sample_unit, dtype=torch.float32, device=device)

        if u.shape[-1] != theta_dim:
            raise ValueError(f"sample_unit last dim must be {theta_dim}, got {tuple(u.shape)}")

        u = torch.clamp(u, min=0.0, max=1.0)
        theta = cls._prior_low + u * (cls._prior_high - cls._prior_low)
        return theta

    @classmethod
    def sample_prior(cls, n_samples: int, seed: Optional[int] = None) -> torch.Tensor:
        """
        Sample theta from U([-3, 3]^10).
        """
        if cls._prior_low is None or cls._prior_high is None:
            cls._setup_statics()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        sample_unit = torch.rand(int(n_samples), theta_dim, device=device)
        return cls._map_unit_to_prior(sample_unit)

    @classmethod
    def get_prior_bounds(cls) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return BoxUniform prior bounds.
        """
        if cls._prior_low is None or cls._prior_high is None:
            cls._setup_statics()
        return cls._prior_low.clone(), cls._prior_high.clone()


# ============================================================
# 3. SBI prior and simulator
# ============================================================
SampleGenerator._setup_statics()

# BoxUniform prior: theta ~ U([-3, 3]^10)
prior = BoxUniform(
    low=SampleGenerator._prior_low,
    high=SampleGenerator._prior_high,
    device=str(device),
)

# SBI recommended processing
sbi_prior, _, prior_returns_numpy = process_prior(prior)


def simulator_wrapper(theta):
    """
    SBI simulator wrapper.

    Input:
        theta: [10] or [B, 10]

    Output:
        theta [10]    -> [100]
        theta [B, 10] -> [B, 100]
    """
    theta = torch.as_tensor(theta, dtype=torch.float32, device=device)

    if theta.ndim == 1:
        generator = SampleGenerator(theta, n_samples=1)
        return generator.generate_samples().view(-1)

    if theta.ndim == 2:
        xs = []
        for th in theta:
            generator = SampleGenerator(th, n_samples=1)
            x = generator.generate_samples().view(-1)
            xs.append(x)
        return torch.stack(xs, dim=0)

    raise ValueError(f"Unsupported theta shape: {tuple(theta.shape)}")


sbi_simulator = process_simulator(
    simulator_wrapper,
    sbi_prior,
    prior_returns_numpy,
)


# ============================================================
# 4. MLE for raw 100-dimensional Bernoulli observations
# ============================================================
def _ensure_2d_raw_data(y_obs: torch.Tensor) -> torch.Tensor:
    """
    Convert y_obs to shape [n_obs, 100].

    Accepted shapes:
        [100]
        [n_obs, 100]
    """
    y = torch.as_tensor(y_obs, dtype=torch.float32, device=device)

    if y.dim() == 1:
        if y.numel() != raw_x_dim:
            raise ValueError(f"1D y_obs must have length {raw_x_dim}, got {y.numel()}")
        y = y.view(1, raw_x_dim)

    elif y.dim() == 2:
        if y.shape[1] != raw_x_dim:
            raise ValueError(f"2D y_obs must have shape [n_obs, {raw_x_dim}], got {tuple(y.shape)}")

    else:
        raise ValueError(f"Unsupported y_obs shape: {tuple(y.shape)}")

    return y.contiguous()


def glm_negative_log_likelihood(
    theta: torch.Tensor,
    y_obs: torch.Tensor,
    reduction: str = "sum",
) -> torch.Tensor:
    """
    Negative log-likelihood for raw Bernoulli GLM observations.

    Model:
        psi = X theta
        y_{i,t} ~ Bernoulli(sigmoid(psi_t))

    Args:
        theta: [10]
        y_obs: [100] or [n_obs, 100]
        reduction: "sum", "mean", or "none"

    Returns:
        scalar NLL if reduction is "sum" or "mean";
        [n_obs, 100] elementwise NLL if reduction is "none".
    """
    if SampleGenerator._design_matrix is None:
        SampleGenerator._setup_statics()

    theta = torch.as_tensor(theta, dtype=torch.float32, device=device)
    if theta.dim() == 2 and theta.shape[0] == 1:
        theta = theta.squeeze(0)
    if theta.shape != (theta_dim,):
        raise ValueError(f"theta must have shape [10] or [1, 10], got {tuple(theta.shape)}")

    y = _ensure_2d_raw_data(y_obs)
    logits = torch.matmul(SampleGenerator._design_matrix, theta)  # [100]
    logits = logits.view(1, raw_x_dim).expand_as(y)               # [n_obs, 100]

    nll = F.binary_cross_entropy_with_logits(logits, y, reduction="none")

    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    if reduction == "none":
        return nll

    raise ValueError("reduction must be 'sum', 'mean', or 'none'")


def _theta_to_phi(theta: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Inverse transform from bounded theta to unconstrained phi."""
    u = (theta - low) / (high - low)
    u = torch.clamp(u, min=1e-6, max=1.0 - 1e-6)
    return torch.log(u) - torch.log1p(-u)


def _phi_to_theta(phi: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Transform unconstrained phi to theta in [low, high]."""
    return low + (high - low) * torch.sigmoid(phi)


def fit_mle(
    y_obs: torch.Tensor,
    init_theta: Optional[torch.Tensor] = None,
    num_restarts: int = 5,
    max_iter: int = 500,
    lr: float = 1.0,
    seed: Optional[int] = 123,
    bounds: Tuple[float, float] = (-3.0, 3.0),
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compute constrained MLE under theta in [-3, 3]^10.

    Since your prior is U([-3, 3]^10), this is the likelihood maximizer over
    the prior support. It uses an unconstrained variable phi and maps it to
    theta by:
        theta = low + (high - low) * sigmoid(phi)

    Args:
        y_obs:
            Raw observations with shape [100] or [n_obs, 100].
        init_theta:
            Optional initial theta with shape [10]. If None, the first restart
            uses zeros and the remaining restarts use samples from the prior.
        num_restarts:
            Number of starting points. Use >1 to reduce local optimisation risk.
        max_iter:
            LBFGS maximum iterations per restart.
        lr:
            LBFGS learning rate.
        seed:
            Random seed for restart initialisations.
        bounds:
            Parameter box constraints. Default is (-3, 3).
        verbose:
            Whether to print per-restart NLL.

    Returns:
        dict with keys:
            theta_mle: [10] tensor
            nll: best negative log-likelihood value
            logits: [100] tensor at theta_mle
            probabilities: [100] tensor sigmoid(logits)
            success: True
            num_restarts: int
    """
    if SampleGenerator._design_matrix is None:
        SampleGenerator._setup_statics()

    y = _ensure_2d_raw_data(y_obs)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    low_value, high_value = bounds
    low = torch.full((theta_dim,), float(low_value), dtype=torch.float32, device=device)
    high = torch.full((theta_dim,), float(high_value), dtype=torch.float32, device=device)

    if not torch.all(high > low):
        raise ValueError(f"Invalid bounds: {bounds}")

    starts = []
    if init_theta is not None:
        theta0 = torch.as_tensor(init_theta, dtype=torch.float32, device=device).reshape(-1)
        if theta0.shape != (theta_dim,):
            raise ValueError(f"init_theta must have shape [10], got {tuple(theta0.shape)}")
        starts.append(torch.clamp(theta0, min=low_value + 1e-5, max=high_value - 1e-5))
    else:
        starts.append(torch.zeros(theta_dim, dtype=torch.float32, device=device))

    for _ in range(max(0, int(num_restarts) - len(starts))):
        starts.append(SampleGenerator.sample_prior(1).reshape(-1))

    best_theta = None
    best_nll = float("inf")

    for restart_idx, theta_start in enumerate(starts):
        phi = _theta_to_phi(theta_start, low, high).detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [phi],
            lr=lr,
            max_iter=int(max_iter),
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            theta_current = _phi_to_theta(phi, low, high)
            loss = glm_negative_log_likelihood(theta_current, y, reduction="sum")
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            theta_candidate = _phi_to_theta(phi, low, high)
            nll_candidate = glm_negative_log_likelihood(theta_candidate, y, reduction="sum").item()

        if verbose:
            print(f"restart {restart_idx + 1}/{len(starts)}: nll={nll_candidate:.6f}")

        if nll_candidate < best_nll:
            best_nll = nll_candidate
            best_theta = theta_candidate.detach().clone()

    logits = torch.matmul(SampleGenerator._design_matrix, best_theta)
    probabilities = torch.sigmoid(logits)

    return {
        "theta_mle": best_theta.detach(),
        "nll": float(best_nll),
        "logits": logits.detach(),
        "probabilities": probabilities.detach(),
        "success": True,
        "num_restarts": len(starts),
        "bounds": (float(low_value), float(high_value)),
    }


# Alias with a more explicit name if your experiment code expects this style.
def compute_mle(y_obs, **kwargs):
    result = fit_mle(y_obs, **kwargs)
    return result["theta_mle"]


# ============================================================
# 5. Unified evaluation metrics
# ============================================================
def _ensure_2d_data(x: torch.Tensor) -> torch.Tensor:
    """Ensure data has shape [n, x_dim]."""
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    if x.dim() == 1:
        x = x.view(1, -1)
    return x.float().contiguous()


def compute_estimation_metrics(
    sample_generator_cls,
    true_theta: torch.Tensor,
    estimate_theta: torch.Tensor,
    y_obs: torch.Tensor,
    n_data: int = 2000,
    seed: int = 42,
    device: torch.device = None,
) -> dict:
    """
    Unified evaluation function for raw 100-dimensional observations.

    Part 1: generated distribution under true_theta vs estimate_theta.
    Part 2: observed data y_obs vs generated distribution under estimate_theta.
    Part 3: parameter-space error.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------
    # Part 1: true_theta distribution vs estimate_theta distribution
    # ------------------------------------------------------------
    gen_true = sample_generator_cls(true_theta, n_samples=n_data, seed=None)
    x_true = _ensure_2d_data(gen_true.generate_samples())

    gen_estimate = sample_generator_cls(estimate_theta, n_samples=n_data, seed=None)
    x_predict = _ensure_2d_data(gen_estimate.generate_samples())

    c2st_score = c2st(x_true.cpu(), x_predict.cpu()).item()

    mu_true = x_true.mean(dim=0, keepdim=True)
    std_true = x_true.std(dim=0, keepdim=True).clamp_min(1e-8)
    x_true_norm = (x_true - mu_true) / std_true
    x_predict_norm = (x_predict - mu_true) / std_true

    sinkhorn_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, debias=False)
    data_sinkhorn_dist = sinkhorn_loss_fn(x_predict_norm, x_true_norm).item()

    energy_loss_fn = SamplesLoss(loss="energy")
    data_energy_dist = energy_loss_fn(x_predict_norm, x_true_norm).item()

    expected_distance = (2.0 * x_true.shape[1]) ** 0.5
    mmd_loss_fn = SamplesLoss(loss="gaussian", blur=expected_distance)
    data_mmd_dist = mmd_loss_fn(x_predict_norm, x_true_norm).item()

    sinkhorn_loss_fn_2 = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, debias=False)
    data_sinkhorn_dist_2 = sinkhorn_loss_fn_2(x_predict, x_true).item()

    energy_loss_fn_2 = SamplesLoss(loss="energy")
    data_energy_dist_2 = energy_loss_fn_2(x_predict, x_true).item()

    mmd_loss_fn_2 = SamplesLoss(loss="gaussian", blur=0.1)
    data_mmd_dist_2 = mmd_loss_fn_2(x_predict, x_true).item()

    # ------------------------------------------------------------
    # Part 2: observed data y_obs vs estimate_theta generated distribution
    # ------------------------------------------------------------
    y_obs = _ensure_2d_data(y_obs)
    n_obs = y_obs.shape[0]

    gen_estimate_obs = sample_generator_cls(estimate_theta, n_samples=n_obs, seed=None)
    x_predict_obs = _ensure_2d_data(gen_estimate_obs.generate_samples())

    obs_c2st_score = c2st(y_obs.cpu(), x_predict_obs.cpu()).item()

    mu_obs = y_obs.mean(dim=0, keepdim=True)
    std_obs = y_obs.std(dim=0, keepdim=True).clamp_min(1e-8)
    y_obs_norm = (y_obs - mu_obs) / std_obs
    x_predict_obs_norm = (x_predict_obs - mu_obs) / std_obs

    obs_sinkhorn_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, debias=False)
    obs_sinkhorn_norm = obs_sinkhorn_loss_fn(x_predict_obs_norm, y_obs_norm).item()

    obs_energy_loss_fn = SamplesLoss(loss="energy")
    obs_energy_norm = obs_energy_loss_fn(x_predict_obs_norm, y_obs_norm).item()

    obs_expected_distance = (2.0 * y_obs.shape[1]) ** 0.5
    obs_mmd_loss_fn = SamplesLoss(loss="gaussian", blur=obs_expected_distance)
    obs_mmd_norm = obs_mmd_loss_fn(x_predict_obs_norm, y_obs_norm).item()

    obs_sinkhorn_loss_fn_2 = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, debias=False)
    obs_sinkhorn = obs_sinkhorn_loss_fn_2(x_predict_obs, y_obs).item()

    obs_energy_loss_fn_2 = SamplesLoss(loss="energy")
    obs_energy = obs_energy_loss_fn_2(x_predict_obs, y_obs).item()

    obs_mmd_loss_fn_2 = SamplesLoss(loss="gaussian", blur=0.1)
    obs_mmd = obs_mmd_loss_fn_2(x_predict_obs, y_obs).item()

    # ------------------------------------------------------------
    # Part 3: parameter-space error
    # ------------------------------------------------------------
    true_theta_cpu = true_theta.detach().cpu().squeeze()
    estimate_theta_cpu = estimate_theta.detach().cpu().squeeze()

    true_theta_processed = target_transform(true_theta_cpu.unsqueeze(0)).squeeze(0)
    estimate_theta_processed = target_transform(estimate_theta_cpu.unsqueeze(0)).squeeze(0)

    param_mse = F.mse_loss(estimate_theta_processed, true_theta_processed).item()
    param_mae = F.l1_loss(estimate_theta_processed, true_theta_processed).item()
    param_rmse = np.sqrt(param_mse)

    dim_mae = {}
    theta_dim_local = true_theta_processed.numel()
    for i in range(theta_dim_local):
        dim_name = f"theta{i + 1}"
        dim_mae[f"{dim_name}_mae"] = F.l1_loss(
            estimate_theta_processed[i:i + 1],
            true_theta_processed[i:i + 1],
        ).item()

    metrics = {
        "c2st": c2st_score,
        "sinkhorn_norm": data_sinkhorn_dist,
        "mmd_norm": data_mmd_dist,
        "energy_norm": data_energy_dist,
        "sinkhorn": data_sinkhorn_dist_2,
        "mmd": data_mmd_dist_2,
        "energy": data_energy_dist_2,
        "obs_c2st": obs_c2st_score,
        "obs_sinkhorn_norm": obs_sinkhorn_norm,
        "obs_mmd_norm": obs_mmd_norm,
        "obs_energy_norm": obs_energy_norm,
        "obs_sinkhorn": obs_sinkhorn,
        "obs_mmd": obs_mmd,
        "obs_energy": obs_energy,
        "param_mse": param_mse,
        "param_mae": param_mae,
        "param_rmse": param_rmse,
        **dim_mae,
        "true_theta": true_theta_cpu.numpy().tolist(),
        "estimate_theta": estimate_theta_cpu.numpy().tolist(),
        "true_theta_processed": true_theta_processed.numpy().tolist(),
        "estimate_theta_processed": estimate_theta_processed.numpy().tolist(),
    }

    return metrics


# ============================================================
# 6. AdaIMLE wrapper
# ============================================================
def adaimle_simulator_wrapper(theta: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
    """
    AdaIMLE simulator wrapper for this raw-output GLM.

    Input:
        theta:
            shape [10] or [B, 10]
        epsilon:
            shape [100] or [B, 100]. It supplies the U(0, 1) noise for the
            100 Bernoulli variables.

    Output:
        generated:
            shape [B, 100]
    """
    theta = torch.as_tensor(theta, dtype=torch.float32, device=device)
    epsilon = torch.as_tensor(epsilon, dtype=torch.float32, device=device)

    if theta.dim() == 1:
        theta = theta.unsqueeze(0)

    theta_flat = theta.view(-1, theta.shape[-1])

    if epsilon.dim() == 1:
        if epsilon.numel() != raw_x_dim:
            raise ValueError(
                f"GLM AdaIMLE latent_dim should be {raw_x_dim}, got epsilon.shape={tuple(epsilon.shape)}"
            )
        eps_flat = epsilon.view(1, raw_x_dim)

    elif epsilon.dim() == 2:
        if epsilon.shape[-1] != raw_x_dim:
            raise ValueError(
                f"GLM AdaIMLE latent_dim should be {raw_x_dim}, got epsilon.shape={tuple(epsilon.shape)}"
            )
        eps_flat = epsilon.view(-1, raw_x_dim)

    else:
        raise ValueError(f"Unsupported epsilon shape: {tuple(epsilon.shape)}")

    batch_size = eps_flat.shape[0]

    if theta_flat.shape[0] == 1:
        generated = SampleGenerator.static_sample_model(
            theta=theta_flat[0],
            n_samples=batch_size,
            eps=eps_flat,
        )

    elif theta_flat.shape[0] == batch_size:
        res_list = []
        for i in range(batch_size):
            res = SampleGenerator.static_sample_model(
                theta=theta_flat[i],
                n_samples=1,
                eps=eps_flat[i:i + 1],
            )
            res_list.append(res)
        generated = torch.cat(res_list, dim=0)

    elif torch.allclose(theta_flat[0], theta_flat[-1]):
        generated = SampleGenerator.static_sample_model(
            theta=theta_flat[0],
            n_samples=batch_size,
            eps=eps_flat,
        )

    else:
        raise ValueError(
            f"theta batch and epsilon batch do not match: "
            f"theta_flat.shape={tuple(theta_flat.shape)}, eps_flat.shape={tuple(eps_flat.shape)}"
        )

    return generated.view(batch_size, x_dim)
