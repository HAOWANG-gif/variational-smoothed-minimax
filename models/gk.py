
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sbi.utils.metrics import c2st, unbiased_mmd_squared, wasserstein_2_squared,biased_mmd
from geomloss import SamplesLoss
from scipy.stats import qmc, norm
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import process_prior, process_simulator
# Select the compute device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_dim=1
theta_dim=4
z_dim=1

def target_transform(theta_batch):
    theta_target = theta_batch.clone()
    return theta_target


prior = BoxUniform(
    low=torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device),
    high=torch.tensor([10.0, 10.0, 10.0, 1.0], dtype=torch.float32, device=device),
    device=str(device)
)
prior_base = torch.distributions.Uniform(
    torch.tensor([0.0, 0.0, 0.0, 0.0]).to(device),
    torch.tensor([10.0, 10.0, 10.0, 1.0]).to(device)
)
sbi_prior = torch.distributions.Independent(prior_base, reinterpreted_batch_ndims=1)
sbi_prior, _, prior_returns_numpy = process_prior(sbi_prior)
def simulator_wrapper(theta):
    generator = SampleGenerator(theta, n_samples=1)
    return generator.generate_samples().view(-1)
sbi_simulator = process_simulator(simulator_wrapper, sbi_prior, prior_returns_numpy)




# ==========================================

# ==========================================
class SampleGenerator:
    def __init__(self, theta, n_samples, seed=None):
        self.theta = theta.to(device)
        self.n_samples = n_samples
        self.seed = seed
        self.c = 0.8
        self._set_random_seed()

    def _set_random_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

    def _unpack_theta_core(self):
        'Unpack theta core.'
        a = self.theta[0]
        b = torch.abs(self.theta[1])  
        g = self.theta[2]
        k = self.theta[3]
        return a, b, g, k

    def generate_samples(self, eps=None):
        'Generate simulator samples, optionally using supplied latent noise.'
        a, b, g, k = self._unpack_theta_core()

        if eps is None:
            eps = torch.randn(self.n_samples, 1, device=device)
        else:
            eps = eps.to(device)
            expected_shape = (self.n_samples, 1)
            if eps.shape != expected_shape:
                raise ValueError(f'Invalid eta shape: expected {expected_shape}, got {eta.shape}')

        z = eps
        skewness_term = 1.0 + self.c * torch.tanh(g * z / 2.0)
        kurtosis_term = torch.pow(1.0 + z ** 2, k)
        samples = a + b * skewness_term * kurtosis_term * z

        return samples.reshape(self.n_samples, 1)

    @staticmethod
    def generate_noise(n_samples, seed=None):
        'Generate latent-noise samples.'
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        return torch.randn(n_samples, 1, device=device)
    @staticmethod
    def static_sample_model(theta, n_samples, seed=None, eps=None):
        generator = SampleGenerator(theta, n_samples, seed)
        return generator.generate_samples(eps=eps)
    @staticmethod
    def _map_unit_to_prior(sample_unit):
        sample_tensor = torch.tensor(sample_unit, dtype=torch.float32, device=device)
        sample_tensor[..., 0] = sample_tensor[..., 0] * 10.0
        sample_tensor[..., 1] = sample_tensor[..., 1] * 10.0
        sample_tensor[..., 2] = sample_tensor[..., 2] * 10.0
        sample_tensor[..., 3] = sample_tensor[..., 3] * 1.0
        return sample_tensor

    @staticmethod
    def sample_prior(n_samples, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        samples = torch.rand(n_samples, 4, device=device)
        return SampleGenerator._map_unit_to_prior(samples)

def compute_estimation_metrics(
        sample_generator_cls,
        true_theta: torch.Tensor,
        estimate_theta: torch.Tensor,
        y_obs: torch.Tensor,
        n_data: int = 2000,
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
    # C2ST: y_obs vs x_predict_obs
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
    for i in range(4):
        dim_name = f"theta{i + 1}"
        dim_mae[f"{dim_name}_mae"] = F.l1_loss(
            estimate_theta_processed[i:i + 1],
            true_theta_processed[i:i + 1]
        ).item()
    metrics = {
        # --------------------------------------------------------
        
        # --------------------------------------------------------
        "c2st": c2st_score,
        "sinkhorn_norm": data_sinkhorn_dist,
        "mmd_norm": data_mmd_dist,
        "energy_norm": data_energy_dist,
        "sinkhorn": data_sinkhorn_dist_2,
        "mmd": data_mmd_dist_2,
        "energy": data_energy_dist_2,

        # --------------------------------------------------------
        
        # --------------------------------------------------------
        "obs_c2st": obs_c2st_score,
        "obs_sinkhorn_norm": obs_sinkhorn_norm,
        "obs_mmd_norm": obs_mmd_norm,
        "obs_energy_norm": obs_energy_norm,
        "obs_sinkhorn": obs_sinkhorn,
        "obs_mmd": obs_mmd,
        "obs_energy": obs_energy,

        # --------------------------------------------------------
        # Parameter-space distance
        # --------------------------------------------------------
        "param_mse": param_mse,
        "param_mae": param_mae,
        "param_rmse": param_rmse,
        **dim_mae,

        # --------------------------------------------------------
        # Parameter history
        # --------------------------------------------------------
        "true_theta": true_theta_cpu.numpy().tolist(),
        "estimate_theta": estimate_theta_cpu.numpy().tolist(),
        "true_theta_processed": true_theta_processed.numpy().tolist(),
        "estimate_theta_processed": estimate_theta_processed.numpy().tolist()
    }

    return metrics



def adaimle_simulator_wrapper(theta: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
    eps_flat = epsilon.view(-1, 1)
    batch_size = eps_flat.shape[0]
    theta_flat = theta.view(-1, theta.shape[-1])
    if theta_flat.shape[0] == 1 or torch.allclose(theta_flat[0], theta_flat[-1]):
        generated = SampleGenerator.static_sample_model(
            theta=theta_flat[0],
            n_samples=batch_size,
            eps=eps_flat
        )
    else:
        res_list = []
        for i in range(batch_size):
            t_idx = i if theta_flat.shape[0] > 1 else 0
            res = SampleGenerator.static_sample_model(
                theta=theta_flat[t_idx],
                n_samples=1,
                eps=eps_flat[i:i + 1]
            )
            res_list.append(res)
        generated = torch.cat(res_list, dim=0)
    out_shape = list(epsilon.shape)
    out_shape[-1] = 1
    return generated.view(*out_shape)
