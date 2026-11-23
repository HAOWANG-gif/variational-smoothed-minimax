import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import qmc
from sbi.utils.metrics import c2st, unbiased_mmd_squared, wasserstein_2_squared, biased_mmd
from geomloss import SamplesLoss
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import process_prior, process_simulator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




x_dim = 2

# theta = (a1, b1, g1, k1, a2, b2, g2, k2, rho)
theta_dim = 9


z_dim = 2


def target_transform(theta_batch):
    'Apply the model-specific parameter transformation.'
    theta_target = theta_batch.clone()
    return theta_target




prior = BoxUniform(
    low=torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        dtype=torch.float32,
        device=device
    ),
    high=torch.tensor(
        [10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 1.0, 1.0],
        dtype=torch.float32,
        device=device
    ),
    device=str(device)
)

prior_base = torch.distributions.Uniform(
    torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        dtype=torch.float32,
        device=device
    ),
    torch.tensor(
        [10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 1.0, 1.0],
        dtype=torch.float32,
        device=device
    )
)

sbi_prior = torch.distributions.Independent(
    prior_base,
    reinterpreted_batch_ndims=1
)

sbi_prior, _, prior_returns_numpy = process_prior(sbi_prior)


def simulator_wrapper(theta):
    generator = SampleGenerator(theta, n_samples=1)
    return generator.generate_samples().view(-1)


sbi_simulator = process_simulator(
    simulator_wrapper,
    sbi_prior,
    prior_returns_numpy
)


class SampleGenerator:
    def __init__(self, theta, n_samples, seed=None):
        self.theta = theta.to(device).float()

        
        if self.theta.dim() == 2 and self.theta.shape[0] == 1:
            self.theta = self.theta.squeeze(0)

        if self.theta.shape != (9,):
            raise ValueError(
                f'Invalid theta shape: expected [9] or [1, 9], got {self.theta.shape}'
            )

        self.n_samples = n_samples
        self.seed = seed
        self._set_random_seed()

    def _set_random_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

    def _unpack_theta_core(self):
        'Unpack theta core.'
        a1 = self.theta[0]
        b1 = self.theta[1]
        g1 = self.theta[2]
        k1 = self.theta[3]

        a2 = self.theta[4]
        b2 = self.theta[5]
        g2 = self.theta[6]
        k2 = self.theta[7]

        rho = self.theta[8]

        
        rho = torch.clamp(rho, min=-0.999999, max=0.999999)

        return a1, b1, g1, k1, a2, b2, g2, k2, rho

    @staticmethod
    def _gk_transform(z, a, b, g, k):
        'Gk transform.'
        skew_term = 1.0 + 0.8 * torch.tanh(0.5 * g * z)
        tail_term = torch.pow(1.0 + z ** 2, k)

        x = a + b * skew_term * tail_term * z

        return x

    def generate_samples(self, eps=None):
        'Generate simulator samples, optionally using supplied latent noise.'
        a1, b1, g1, k1, a2, b2, g2, k2, rho = self._unpack_theta_core()

        if eps is None:
            eps = torch.randn(self.n_samples, 2, device=device)
        else:
            eps = eps.to(device).float()
            expected_shape = (self.n_samples, 2)

            if eps.shape != expected_shape:
                raise ValueError(
                    f'Invalid eps shape: expected {expected_shape}, got {eps.shape}'
                )

        eta1 = eps[:, 0]
        eta2 = eps[:, 1]

        
        z1 = eta1
        z2 = rho * eta1 + torch.sqrt(1.0 - rho ** 2) * eta2

        
        x1 = self._gk_transform(z1, a1, b1, g1, k1)
        x2 = self._gk_transform(z2, a2, b2, g2, k2)

        samples = torch.stack([x1, x2], dim=1)

        return samples.reshape(self.n_samples, 2)

    @staticmethod
    def generate_noise(n_samples, seed=None):
        'Generate latent-noise samples.'
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        return torch.randn(n_samples, 2, device=device)

    @staticmethod
    def static_sample_model(theta, n_samples, seed=None, eps=None):
        generator = SampleGenerator(theta, n_samples, seed)
        return generator.generate_samples(eps=eps)

    @staticmethod
    def _map_unit_to_prior(sample_unit):
        'Map unit to prior.'
        if isinstance(sample_unit, torch.Tensor):
            sample_tensor = sample_unit.detach().clone().to(device).float()
        else:
            sample_tensor = torch.tensor(
                sample_unit,
                dtype=torch.float32,
                device=device
            )

        if sample_tensor.shape[-1] != 9:
            raise ValueError(
                f'The last dimension of sample_unit must be 9, got {sample_tensor.shape}'
            )

        sample_tensor = torch.clamp(sample_tensor, min=0.0, max=1.0)

        low = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            dtype=torch.float32,
            device=device
        )

        high = torch.tensor(
            [10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 1.0, 1.0],
            dtype=torch.float32,
            device=device
        )
        return low + sample_tensor * (high - low)

    @staticmethod
    def sample_prior(n_samples, seed=None):
        'Draw parameter values from the prior.'
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        sample_unit = torch.rand(n_samples, 9, device=device)

        return SampleGenerator._map_unit_to_prior(sample_unit)

    @staticmethod
    def get_prior_bounds():
        'Return the lower and upper prior bounds.'
        prior_low = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            dtype=torch.float32,
            device=device
        )

        prior_high = torch.tensor(
            [10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 1.0, 1.0],
            dtype=torch.float32,
            device=device
        )

        return prior_low, prior_high



def compute_estimation_metrics(
        sample_generator_cls,
        true_theta: torch.Tensor,
        estimate_theta: torch.Tensor,
        y_obs: torch.Tensor,
        n_data: int = 1000,
        seed: int = 42,
        device: torch.device = None
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ============================================================
    # Part 1: true-theta distribution versus estimated-theta distribution
    # ============================================================
    gen_true = sample_generator_cls(true_theta, n_samples=n_data, seed=None)
    x_true = gen_true.generate_samples().detach().to(device).float()

    gen_estimate = sample_generator_cls(estimate_theta, n_samples=n_data, seed=None)
    x_predict = gen_estimate.generate_samples().detach().to(device).float()

    if x_true.dim() == 1:
        x_true = x_true.unsqueeze(1)
    if x_predict.dim() == 1:
        x_predict = x_predict.unsqueeze(1)

    x_true = x_true.float().contiguous()
    x_predict = x_predict.float().contiguous()

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

    # ============================================================
    # Part 2: observed data versus estimated-theta distribution
    # ============================================================
    y_obs = y_obs.detach().to(device).float()

    if y_obs.dim() == 1:
        y_obs = y_obs.unsqueeze(1)

    y_obs = y_obs.float().contiguous()

    n_obs = y_obs.shape[0]

    gen_estimate_obs = sample_generator_cls(
        estimate_theta,
        n_samples=n_obs,
        seed=None
    )

    x_predict_obs = gen_estimate_obs.generate_samples().detach().to(device).float()

    if x_predict_obs.dim() == 1:
        x_predict_obs = x_predict_obs.unsqueeze(1)

    x_predict_obs = x_predict_obs.float().contiguous()

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

    # ============================================================
    # Part 3: parameter-space error
    # ============================================================
    true_theta_cpu = true_theta.detach().cpu().squeeze()
    estimate_theta_cpu = estimate_theta.detach().cpu().squeeze()

    true_theta_processed = true_theta_cpu.clone()
    estimate_theta_processed = estimate_theta_cpu.clone()

    param_mse = F.mse_loss(
        estimate_theta_processed,
        true_theta_processed
    ).item()

    param_mae = F.l1_loss(
        estimate_theta_processed,
        true_theta_processed
    ).item()

    param_rmse = np.sqrt(param_mse)

    dim_mae = {}
    for i in range(9):
        dim_name = f"theta{i + 1}"
        dim_mae[f"{dim_name}_mae"] = F.l1_loss(
            estimate_theta_processed[i:i + 1],
            true_theta_processed[i:i + 1]
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
        "estimate_theta_processed": estimate_theta_processed.numpy().tolist()
    }

    return metrics



def adaimle_simulator_wrapper(theta: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:


    theta = theta.to(device).float()
    epsilon = epsilon.to(device).float()

    # --------------------------------------------------
    
    # --------------------------------------------------
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)

    theta_flat = theta.view(-1, theta.shape[-1])

    if theta_flat.shape[-1] != 9:
        raise ValueError(
            f'The bivariate g-and-k model requires theta_dim=9, but received theta.shape={theta.shape}'
        )

    # --------------------------------------------------
    
    # --------------------------------------------------
    if epsilon.dim() == 2:
        if epsilon.shape[-1] != 2:
            raise ValueError(
                f'The bivariate g-and-k model requires AdaIMLE latent_dim=2, but received epsilon.shape={epsilon.shape}'
            )

        eps_gk = epsilon

    else:
        raise ValueError(
            f'Unsupported epsilon dimensionality: epsilon.shape={epsilon.shape}'
        )

    batch_size = eps_gk.shape[0]

    # --------------------------------------------------
    
    # --------------------------------------------------
    if theta_flat.shape[0] == 1:
        generated = SampleGenerator.static_sample_model(
            theta=theta_flat[0],
            n_samples=batch_size,
            eps=eps_gk
        )

    # --------------------------------------------------
    
    # --------------------------------------------------
    elif theta_flat.shape[0] == batch_size:
        res_list = []

        for i in range(batch_size):
            res = SampleGenerator.static_sample_model(
                theta=theta_flat[i],
                n_samples=1,
                eps=eps_gk[i:i + 1]
            )
            res_list.append(res)

        generated = torch.cat(res_list, dim=0)

    # --------------------------------------------------
    
    # --------------------------------------------------
    elif torch.allclose(theta_flat[0], theta_flat[-1]):
        generated = SampleGenerator.static_sample_model(
            theta=theta_flat[0],
            n_samples=batch_size,
            eps=eps_gk
        )

    else:
        raise ValueError(
            f'theta and epsilon batch sizes do not match: theta_flat.shape={theta_flat.shape}, eps_gk.shape={eps_gk.shape}'
        )

    return generated.view(batch_size, -1)
