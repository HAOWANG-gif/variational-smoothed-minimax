# -*- coding: utf-8 -*-
'MNIST binary-classification model based on a binary stochastic feedforward neural network.'

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import process_prior, process_simulator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================

# ============================================================

DEFAULT_FEATURE_DIM = 20

DEFAULT_R_DIM = 20
DEFAULT_TRAIN_SAMPLES = 1000
DEFAULT_TEST_SAMPLES = None
DEFAULT_PRIOR_BOUND = 3.0

DEFAULT_NEGATIVE_DIGIT = 1
DEFAULT_POSITIVE_DIGIT = 2


feature_dim = DEFAULT_FEATURE_DIM
r_dim = DEFAULT_R_DIM

x_dim = feature_dim + 1
obs_dim = x_dim

theta_dim = 1 + r_dim + (feature_dim + 1) * r_dim
z_dim = 1 + r_dim + 1


def compute_mnist_mixed_logit_dims(d: int, r: int) -> tuple[int, int, int, int]:
    'Compute mnist mixed logit dims.'
    feature_dim_local = int(d)
    hidden_dim_local = int(r)

    if feature_dim_local <= 0:
        raise ValueError(f'd must be a positive integer, got d={feature_dim_local}.')
    if hidden_dim_local <= 0:
        raise ValueError(f'r/m must be a positive integer, got r={hidden_dim_local}.')

    obs_dim_local = feature_dim_local + 1
    theta_dim_local = 1 + hidden_dim_local + (feature_dim_local + 1) * hidden_dim_local
    z_dim_local = 1 + hidden_dim_local + 1

    return feature_dim_local, obs_dim_local, theta_dim_local, z_dim_local


def sync_global_dims(d: int, r: int) -> None:
    'Sync global dims.'
    global feature_dim, r_dim, x_dim, obs_dim, theta_dim, z_dim

    feature_dim, obs_dim, theta_dim, z_dim = compute_mnist_mixed_logit_dims(d, r)
    r_dim = int(r)
    x_dim = obs_dim


@dataclass
class Mnist12Data:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    pca: PCA
    negative_digit: int = DEFAULT_NEGATIVE_DIGIT
    positive_digit: int = DEFAULT_POSITIVE_DIGIT


@dataclass
class MnistMixedLogitProblem:
    'MnistMixedLogitProblem.'

    data: Mnist12Data
    y_obs: torch.Tensor
    init_theta: torch.Tensor

    sample_generator_cls: type
    prior: Any
    sbi_prior: Any
    sbi_simulator: Any
    adaimle_simulator_wrapper: Any

    feature_dim: int
    obs_dim: int
    theta_dim: int
    z_dim: int
    r_dim: int

    n_train: int
    n_test: int
    seed: int
    pca_whiten: bool
    init_strategy: str
    train_samples: int | None
    max_test_samples: int | None
    negative_digit: int = DEFAULT_NEGATIVE_DIGIT
    positive_digit: int = DEFAULT_POSITIVE_DIGIT



MnistBinaryStochasticFeedforwardProblem = MnistMixedLogitProblem


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================

# ============================================================

def load_mnist_openml(data_home: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    'Load MNIST from a local npz file when available, otherwise download OpenML data_id=554.'
    data_home_path = Path(data_home).expanduser()

    env_npz = os.environ.get("MNIST_LOCAL_NPZ")
    original_windows_npz = Path.home() / "Desktop" / "mnist_784_local.npz"
    cache_npz = data_home_path / "mnist_784_local.npz"

    candidates: list[Path] = []
    if env_npz:
        candidates.append(Path(env_npz).expanduser())
    candidates.extend([original_windows_npz, cache_npz])

    for local_npz in candidates:
        if local_npz.exists():
            print(f'Loading MNIST from local file: {local_npz}')
            data = np.load(local_npz)
            X = data["X"].astype(np.float32)
            y = data["y"].astype(np.int64)

            
            if X.size > 0 and float(np.nanmax(X)) > 1.5:
                X = X / 255.0

            return X, y

    print('No local MNIST npz file was found; downloading from OpenML.')
    data_home_path.mkdir(parents=True, exist_ok=True)

    mnist = fetch_openml(
        data_id=554,
        as_frame=False,
        data_home=str(data_home_path),
        parser="auto",
    )

    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)

    np.savez_compressed(cache_npz, X=X, y=y)
    print(f'MNIST saved locally to: {cache_npz}')

    return X, y


def filter_binary_digits(
    X: np.ndarray,
    y: np.ndarray,
    negative_digit: int = DEFAULT_NEGATIVE_DIGIT,
    positive_digit: int = DEFAULT_POSITIVE_DIGIT,
) -> tuple[np.ndarray, np.ndarray]:
    'Keep two MNIST digits and encode them as binary labels.'
    negative_digit = int(negative_digit)
    positive_digit = int(positive_digit)

    if negative_digit == positive_digit:
        raise ValueError(
            f'negative_digit and positive_digit must differ; both are {negative_digit}.'
        )

    mask = (y == negative_digit) | (y == positive_digit)

    X_binary = X[mask].astype(np.float32)
    y_binary = (y[mask] == positive_digit).astype(np.float32)

    return X_binary, y_binary


def filter_digits_1_and_2(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    'Filter digits 1 and 2.'
    return filter_binary_digits(
        X=X,
        y=y,
        negative_digit=1,
        positive_digit=2,
    )


def maybe_limit_samples(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    'Maybe limit samples.'
    if max_samples is None or max_samples <= 0 or max_samples >= len(y):
        return X, y

    rng = np.random.default_rng(seed)

    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)

    n0 = min(max_samples // 2, len(idx0))
    n1 = min(max_samples - n0, len(idx1))

    keep = np.concatenate(
        [
            rng.choice(idx0, size=n0, replace=False),
            rng.choice(idx1, size=n1, replace=False),
        ]
    )

    rng.shuffle(keep)

    return X[keep], y[keep]


def load_mnist_1_2_pca(
    d: int = DEFAULT_FEATURE_DIM,
    data_home: str | os.PathLike = "./mnist_openml_cache",
    seed: int = 123,
    pca_whiten: bool = True,
    train_samples: int | None = DEFAULT_TRAIN_SAMPLES,
    max_test_samples: int | None = DEFAULT_TEST_SAMPLES,
    negative_digit: int = DEFAULT_NEGATIVE_DIGIT,
    positive_digit: int = DEFAULT_POSITIVE_DIGIT,
) -> Mnist12Data:
    'Load mnist 1 2 pca.'
    X_all, y_all = load_mnist_openml(data_home)

    if X_all.shape[0] < 70000 or y_all.shape[0] < 70000:
        raise ValueError(
            'MNIST contains fewer than 70,000 samples and cannot be split using the standard 60,000/10,000 partition.'
        )

    X_train_raw, y_train_raw = X_all[:60000], y_all[:60000]
    X_test_raw, y_test_raw = X_all[60000:70000], y_all[60000:70000]

    X_train_pool, y_train_pool = filter_binary_digits(
        X_train_raw,
        y_train_raw,
        negative_digit=negative_digit,
        positive_digit=positive_digit,
    )
    X_test_pool, y_test_pool = filter_binary_digits(
        X_test_raw,
        y_test_raw,
        negative_digit=negative_digit,
        positive_digit=positive_digit,
    )

    if d > min(X_train_pool.shape[0], X_train_pool.shape[1]):
        raise ValueError(
            f'PCA dimension d={d} is too large; full training-pool size={X_train_pool.shape[0]}, original dimension={X_train_pool.shape[1]}.'
        )

    pca = PCA(n_components=d, whiten=pca_whiten, random_state=seed)

    X_train_pool_pca = pca.fit_transform(X_train_pool).astype(np.float32)
    X_test_pool_pca = pca.transform(X_test_pool).astype(np.float32)

    X_train_pca, y_train = maybe_limit_samples(
        X_train_pool_pca,
        y_train_pool,
        train_samples,
        seed,
    )

    X_test_pca, y_test = maybe_limit_samples(
        X_test_pool_pca,
        y_test_pool,
        max_test_samples,
        seed + 1,
    )

    return Mnist12Data(
        X_train=torch.tensor(X_train_pca, dtype=torch.float32, device=device),
        y_train=torch.tensor(y_train, dtype=torch.float32, device=device),
        X_test=torch.tensor(X_test_pca, dtype=torch.float32, device=device),
        y_test=torch.tensor(y_test, dtype=torch.float32, device=device),
        pca=pca,
        negative_digit=int(negative_digit),
        positive_digit=int(positive_digit),
    )


def make_observed_rows(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cat([X, y.view(-1, 1)], dim=1).detach()


# ============================================================
# 2. Binary stochastic feedforward simulator
# ============================================================

class MixedLogitSampleGenerator:
    'MixedLogitSampleGenerator.'

    X_obs: torch.Tensor | None = None
    n_obs: int | None = None
    feature_dim: int | None = None
    r_dim: int | None = None
    theta_dim: int | None = None
    obs_dim: int | None = None
    z_dim: int | None = None

    prior_low: torch.Tensor | None = None
    prior_high: torch.Tensor | None = None

    @classmethod
    def setup(
        cls,
        X_obs: torch.Tensor,
        r_dim: int,
        prior_bound: float = DEFAULT_PRIOR_BOUND,
    ) -> None:
        cls.X_obs = X_obs.to(device).float()
        cls.n_obs = int(cls.X_obs.shape[0])
        cls.feature_dim = int(cls.X_obs.shape[1])
        cls.r_dim = int(r_dim)

        if cls.n_obs <= 0:
            raise ValueError('X_obs must not be empty.')
        if cls.r_dim <= 0:
            raise ValueError(f'r_dim/m must be a positive integer, got {cls.r_dim}.')

        cls.theta_dim = 1 + cls.r_dim + (cls.feature_dim + 1) * cls.r_dim
        cls.obs_dim = cls.feature_dim + 1
        cls.z_dim = 1 + cls.r_dim + 1

        cls.prior_low = -float(prior_bound) * torch.ones(
            cls.theta_dim,
            dtype=torch.float32,
            device=device,
        )
        cls.prior_high = float(prior_bound) * torch.ones(
            cls.theta_dim,
            dtype=torch.float32,
            device=device,
        )

        sync_global_dims(cls.feature_dim, cls.r_dim)

    def __init__(
        self,
        theta: torch.Tensor,
        n_samples: int,
        seed: int | None = None,
    ):
        if self.X_obs is None:
            raise RuntimeError(
                'Call MixedLogitSampleGenerator.setup(X_obs, r_dim) first.'
            )

        self.theta = theta.to(device).float()

        if self.theta.dim() == 1:
            self.theta = self.theta.unsqueeze(0)

        self.n_samples = int(n_samples)
        self.seed = seed

    @staticmethod
    def unpack_theta(
        theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        'Unpack theta.'
        theta = theta.to(device).float().view(-1)

        d = int(MixedLogitSampleGenerator.feature_dim)
        m = int(MixedLogitSampleGenerator.r_dim)

        expected = 1 + m + (d + 1) * m
        if theta.numel() != expected:
            raise ValueError(
                f'Invalid theta dimension: expected {expected}, got {theta.numel()}'
            )

        c = theta[0]
        v = theta[1:1 + m]
        A_aug = theta[1 + m:].view(d + 1, m)

        return c, v, A_aug

    @staticmethod
    def generate_noise(n_samples: int, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        idx_u = torch.rand(n_samples, 1, device=device)
        hidden_u = torch.rand(
            n_samples,
            int(MixedLogitSampleGenerator.r_dim),
            device=device,
        )
        y_u = torch.rand(n_samples, 1, device=device)

        return torch.cat([idx_u, hidden_u, y_u], dim=1)

    @staticmethod
    def static_sample_model(
        theta: torch.Tensor,
        n_samples: int,
        seed: int | None = None,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        generator = MixedLogitSampleGenerator(theta, n_samples, seed)
        return generator.generate_samples(eps=eps)

    def generate_samples(self, eps: torch.Tensor | None = None) -> torch.Tensor:
        if self.theta.shape[0] != 1:
            raise ValueError('This estimator optimizes one theta value at a time.')

        if eps is None:
            eps = self.generate_noise(self.n_samples, self.seed)
        else:
            eps = eps.to(device).float()

        expected_shape = (
            self.n_samples,
            int(MixedLogitSampleGenerator.z_dim),
        )
        if tuple(eps.shape) != expected_shape:
            raise ValueError(
                f'Invalid eps shape: expected {expected_shape}, got {tuple(eps.shape)}'
            )

        idx = torch.clamp(
            (eps[:, 0] * int(MixedLogitSampleGenerator.n_obs)).long(),
            min=0,
            max=int(MixedLogitSampleGenerator.n_obs) - 1,
        )

        x = MixedLogitSampleGenerator.X_obs[idx]

        m = int(MixedLogitSampleGenerator.r_dim)
        d = int(MixedLogitSampleGenerator.feature_dim)

        hidden_u = eps[:, 1:1 + m]
        bernoulli_u = eps[:, -1]

        c, v, A_aug = self.unpack_theta(self.theta.squeeze(0))
        W = A_aug[:d, :]
        a = A_aug[d, :]

        hidden_logits = x @ W + a.unsqueeze(0)
        hidden_prob = torch.sigmoid(hidden_logits)

        h_hard = (hidden_u < hidden_prob).float()

        # straight-through estimator:
        
        h_st = h_hard.detach() - hidden_prob.detach() + hidden_prob

        output_logits = c + (h_st @ v) / math.sqrt(float(m))
        output_prob = torch.sigmoid(output_logits)

        y_hard = (bernoulli_u < output_prob).float()

        
        y_st = y_hard.detach() - output_prob.detach() + output_prob

        return torch.cat([x, y_st.unsqueeze(1)], dim=1)

    @staticmethod
    def _map_unit_to_prior(sample_unit: torch.Tensor | np.ndarray) -> torch.Tensor:
        if MixedLogitSampleGenerator.prior_low is None:
            raise RuntimeError('Call MixedLogitSampleGenerator.setup(...) first.')

        sample_tensor = torch.as_tensor(
            sample_unit,
            dtype=torch.float32,
            device=device,
        )

        return (
            MixedLogitSampleGenerator.prior_low
            + sample_tensor
            * (
                MixedLogitSampleGenerator.prior_high
                - MixedLogitSampleGenerator.prior_low
            )
        )

    @staticmethod
    def sample_prior(n_samples: int, seed: int | None = None) -> torch.Tensor:
        if MixedLogitSampleGenerator.theta_dim is None:
            raise RuntimeError('Call MixedLogitSampleGenerator.setup(...) first.')

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        samples = torch.rand(
            n_samples,
            int(MixedLogitSampleGenerator.theta_dim),
            device=device,
        )

        return MixedLogitSampleGenerator._map_unit_to_prior(samples)



BinaryStochasticFeedforwardSampleGenerator = MixedLogitSampleGenerator


# ============================================================

# ============================================================

def target_transform(theta_batch):
    theta = theta_batch.clone()

    batch_size = theta.shape[0]
    m = int(MixedLogitSampleGenerator.r_dim)
    d = int(MixedLogitSampleGenerator.feature_dim)

    expected_dim = 1 + m + (d + 1) * m
    if theta.shape[1] != expected_dim:
        raise ValueError(
            f'Invalid theta dimension: expected {expected_dim}, got {theta.shape[1]}'
        )

    c = theta[:, 0].clone()
    v = theta[:, 1:1 + m].clone()
    A_aug = theta[:, 1 + m:].reshape(
        batch_size, d + 1, m
    ).clone()

    
    flip = v < 0

    c = c + torch.where(
        flip, v, torch.zeros_like(v)
    ).sum(dim=1) / math.sqrt(float(m))

    A_aug = torch.where(
        flip.unsqueeze(1),
        -A_aug,
        A_aug,
    )

    v = torch.abs(v)

    
    order = torch.argsort(v, dim=1, descending=True)

    v = torch.gather(v, 1, order)

    A_aug = torch.gather(
        A_aug,
        dim=2,
        index=order.unsqueeze(1).expand(-1, d + 1, -1),
    )

    return torch.cat(
        [
            c.unsqueeze(1),
            v,
            A_aug.reshape(batch_size, -1),
        ],
        dim=1,
    )

def make_initial_theta(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    r_dim: int,
    strategy: str = "class_mean",
    seed: int = 123,
) -> torch.Tensor:
    'Make initial theta.'
    set_seed(seed)

    X_train = X_train.to(device).float()
    y_train = y_train.to(device).float().view(-1)

    d = int(X_train.shape[1])
    m = int(r_dim)

    if m <= 0:
        raise ValueError(f'r_dim/m must be a positive integer, got {m}.')

    theta = torch.zeros(
        1 + m + (d + 1) * m,
        dtype=torch.float32,
        device=device,
    )

    if strategy == "zeros":
        return theta

    if strategy != "class_mean":
        raise ValueError("init strategy must be 'class_mean' or 'zeros'.")

    x0 = X_train[y_train < 0.5]
    x1 = X_train[y_train >= 0.5]

    if x0.numel() == 0 or x1.numel() == 0:
        return theta

    mean0 = x0.mean(dim=0)
    mean1 = x1.mean(dim=0)
    midpoint = 0.5 * (mean0 + mean1)

    direction = mean1 - mean0
    direction_norm = torch.linalg.vector_norm(direction).clamp_min(1e-6)
    unit_direction = direction / direction_norm

    
    strengths = torch.linspace(0.75, 1.25, m, device=device)
    W = unit_direction.unsqueeze(1) * strengths.unsqueeze(0)
    W = W + 0.05 * torch.randn(d, m, device=device) / math.sqrt(float(d))

    
    threshold_offsets = torch.linspace(-1.0, 1.0, m, device=device)
    a = -(midpoint @ W) + threshold_offsets

    
    v = torch.ones(m, dtype=torch.float32, device=device)
    v = v + 0.02 * torch.randn(m, device=device)

    prior_prob = y_train.mean().clamp(1e-4, 1.0 - 1e-4)
    log_odds = torch.log(prior_prob / (1.0 - prior_prob))

    with torch.no_grad():
        mean_hidden_prob = torch.sigmoid(X_train @ W + a).mean(dim=0)
        c = log_odds - torch.dot(v, mean_hidden_prob) / math.sqrt(float(m))

    A_aug = torch.cat([W, a.unsqueeze(0)], dim=0)

    theta[0] = c
    theta[1:1 + m] = v
    theta[1 + m:] = A_aug.reshape(-1)

    return theta


# ============================================================
# 4. SBI / AdaIMLE wrapper
# ============================================================

def make_box_prior(theta_dim_value: int, prior_bound: float = DEFAULT_PRIOR_BOUND):
    low = -float(prior_bound) * torch.ones(
        theta_dim_value,
        dtype=torch.float32,
        device=device,
    )
    high = float(prior_bound) * torch.ones(
        theta_dim_value,
        dtype=torch.float32,
        device=device,
    )

    prior = BoxUniform(low=low, high=high, device=str(device))

    prior_base = torch.distributions.Uniform(low, high)
    sbi_prior = torch.distributions.Independent(
        prior_base,
        reinterpreted_batch_ndims=1,
    )

    sbi_prior, _, prior_returns_numpy = process_prior(sbi_prior)

    return prior, sbi_prior, prior_returns_numpy


def make_sbi_simulator(sbi_prior, prior_returns_numpy):
    def simulator_wrapper(theta: torch.Tensor) -> torch.Tensor:
        generator = MixedLogitSampleGenerator(theta, n_samples=1)
        return generator.generate_samples().view(-1)

    return process_simulator(
        simulator_wrapper,
        sbi_prior,
        prior_returns_numpy,
    )


def adaimle_simulator_wrapper(theta: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
    'Adapt the simulator to the AdaIMLE batch interface.'
    epsilon = epsilon.to(device).float()
    theta = theta.to(device).float()

    eps_flat = epsilon.reshape(-1, int(MixedLogitSampleGenerator.z_dim))
    batch_size = eps_flat.shape[0]

    theta_flat = theta.reshape(-1, theta.shape[-1])

    if theta_flat.shape[0] == 1 or torch.allclose(theta_flat[0], theta_flat[-1]):
        generated = MixedLogitSampleGenerator.static_sample_model(
            theta=theta_flat[0],
            n_samples=batch_size,
            eps=eps_flat,
        )
    elif theta_flat.shape[0] == batch_size:
        res_list = []

        for i in range(batch_size):
            res = MixedLogitSampleGenerator.static_sample_model(
                theta=theta_flat[i],
                n_samples=1,
                eps=eps_flat[i:i + 1],
            )
            res_list.append(res)

        generated = torch.cat(res_list, dim=0)
    else:
        raise ValueError(
            f'theta and epsilon batch sizes do not match: theta_flat.shape={theta_flat.shape}, eps_flat.shape={eps_flat.shape}'
        )

    out_shape = list(epsilon.shape)
    out_shape[-1] = int(MixedLogitSampleGenerator.obs_dim)

    return generated.view(*out_shape)


# ============================================================

# ============================================================

@torch.no_grad()
def predict_mixed_logit_proba(
    X: torch.Tensor,
    theta: torch.Tensor,
    r_dim: int,
    n_mc: int = 512,
    seed: int = 123,
    batch_size: int = 2048,
) -> torch.Tensor:
    'Predict mixed logit proba.'
    if n_mc <= 0:
        raise ValueError(f'n_mc must be a positive integer, got {n_mc}.')
    if batch_size <= 0:
        raise ValueError(f'batch_size must be a positive integer, got {batch_size}.')

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    X = X.to(device).float()
    theta = theta.detach().to(device).float().view(-1)

    d = int(X.shape[1])
    m = int(r_dim)

    expected = 1 + m + (d + 1) * m
    if theta.numel() != expected:
        raise ValueError(
            f'Invalid theta dimension: expected {expected}, got {theta.numel()}.'
        )

    c = theta[0]
    v = theta[1:1 + m]
    A_aug = theta[1 + m:].view(d + 1, m)
    W = A_aug[:d, :]
    a = A_aug[d, :]

    
    hidden_u_mc = torch.rand(
        n_mc,
        m,
        generator=generator,
        device=device,
    )

    probs = []

    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        X_batch = X[start:end]

        hidden_prob = torch.sigmoid(X_batch @ W + a.unsqueeze(0))

        # shape: [batch, n_mc, m]
        h_mc = (
            hidden_u_mc.unsqueeze(0) < hidden_prob.unsqueeze(1)
        ).to(torch.float32)

        # shape: [batch, n_mc]
        output_logits = c + torch.einsum("bnm,m->bn", h_mc, v) / math.sqrt(float(m))
        output_prob = torch.sigmoid(output_logits)

        probs.append(output_prob.mean(dim=1))

    return torch.cat(probs, dim=0)


def evaluate_predictions(y_true: torch.Tensor, p_hat: torch.Tensor) -> dict:
    y_np = y_true.detach().cpu().numpy().astype(int)
    p_np = p_hat.detach().cpu().numpy()
    pred_np = (p_np >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_np, pred_np)),
        "f1": float(f1_score(y_np, pred_np)),
        "nll": float(
            log_loss(
                y_np,
                np.clip(p_np, 1e-7, 1.0 - 1e-7),
                labels=[0, 1],
            )
        ),
        "mean_probability": float(np.mean(p_np)),
    }

    try:
        metrics["auc"] = float(roc_auc_score(y_np, p_np))
    except ValueError:
        metrics["auc"] = float("nan")

    return metrics


def compute_classification_metrics(
    theta_hat: torch.Tensor,
    problem: MnistMixedLogitProblem,
    n_mc_predict: int = 512,
    seed: int = 123,
) -> dict:
    'Compute classification metrics.'
    p_train = predict_mixed_logit_proba(
        problem.data.X_train,
        theta_hat,
        r_dim=problem.r_dim,
        n_mc=n_mc_predict,
        seed=seed + 10,
    )

    p_test = predict_mixed_logit_proba(
        problem.data.X_test,
        theta_hat,
        r_dim=problem.r_dim,
        n_mc=n_mc_predict,
        seed=seed + 11,
    )

    train_metrics = evaluate_predictions(problem.data.y_train, p_train)
    test_metrics = evaluate_predictions(problem.data.y_test, p_test)

    return {
        "n_train": int(problem.data.X_train.shape[0]),
        "n_test": int(problem.data.X_test.shape[0]),
        "feature_dim": int(problem.feature_dim),
        "obs_dim": int(problem.obs_dim),
        "r_dim": int(problem.r_dim),
        "theta_dim": int(theta_hat.numel()),
        "z_dim": int(problem.z_dim),
        "pca_whiten": bool(problem.pca_whiten),
        "init_strategy": problem.init_strategy,
        "train_samples": problem.train_samples,
        "max_test_samples": problem.max_test_samples,
        "negative_digit": int(problem.negative_digit),
        "positive_digit": int(problem.positive_digit),
        "n_mc_predict": int(n_mc_predict),
        "theta_norm": float(torch.linalg.vector_norm(theta_hat).item()),

        "train_accuracy": train_metrics["accuracy"],
        "train_f1": train_metrics["f1"],
        "train_nll": train_metrics["nll"],
        "train_auc": train_metrics["auc"],
        "train_mean_probability": train_metrics["mean_probability"],

        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_nll": test_metrics["nll"],
        "test_auc": test_metrics["auc"],
        "test_mean_probability": test_metrics["mean_probability"],
    }


# ============================================================

# ============================================================

def build_mnist_mixed_logit_problem(
    d: int = DEFAULT_FEATURE_DIM,
    r_dim: int = DEFAULT_R_DIM,
    data_home: str | os.PathLike = "./mnist_openml_cache",
    seed: int = 123,
    train_samples: int | None = DEFAULT_TRAIN_SAMPLES,
    n_train: int | None = None,
    max_test_samples: int | None = DEFAULT_TEST_SAMPLES,
    pca_whiten: bool = True,
    init_strategy: str = "class_mean",
    prior_bound: float = DEFAULT_PRIOR_BOUND,
    negative_digit: int = DEFAULT_NEGATIVE_DIGIT,
    positive_digit: int = DEFAULT_POSITIVE_DIGIT,
) -> MnistMixedLogitProblem:
    'Build the MNIST binary stochastic feedforward model while preserving the historical public interface.'
    set_seed(seed)

    if train_samples is None and n_train is not None:
        train_samples = n_train

    data = load_mnist_1_2_pca(
        d=d,
        data_home=data_home,
        seed=seed,
        pca_whiten=pca_whiten,
        train_samples=train_samples,
        max_test_samples=max_test_samples,
        negative_digit=negative_digit,
        positive_digit=positive_digit,
    )

    MixedLogitSampleGenerator.setup(
        X_obs=data.X_train,
        r_dim=r_dim,
        prior_bound=prior_bound,
    )

    y_obs = make_observed_rows(data.X_train, data.y_train)

    init_theta = make_initial_theta(
        X_train=data.X_train,
        y_train=data.y_train,
        r_dim=r_dim,
        strategy=init_strategy,
        seed=seed,
    )

    prior, sbi_prior, prior_returns_numpy = make_box_prior(
        theta_dim_value=int(MixedLogitSampleGenerator.theta_dim),
        prior_bound=prior_bound,
    )

    sbi_simulator = make_sbi_simulator(sbi_prior, prior_returns_numpy)

    return MnistMixedLogitProblem(
        data=data,
        y_obs=y_obs,
        init_theta=init_theta.detach().to(device),
        sample_generator_cls=MixedLogitSampleGenerator,
        prior=prior,
        sbi_prior=sbi_prior,
        sbi_simulator=sbi_simulator,
        adaimle_simulator_wrapper=adaimle_simulator_wrapper,
        feature_dim=int(MixedLogitSampleGenerator.feature_dim),
        obs_dim=int(MixedLogitSampleGenerator.obs_dim),
        theta_dim=int(MixedLogitSampleGenerator.theta_dim),
        z_dim=int(MixedLogitSampleGenerator.z_dim),
        r_dim=int(r_dim),
        n_train=int(data.X_train.shape[0]),
        n_test=int(data.X_test.shape[0]),
        seed=int(seed),
        pca_whiten=bool(pca_whiten),
        init_strategy=init_strategy,
        train_samples=train_samples,
        max_test_samples=max_test_samples,
        negative_digit=int(negative_digit),
        positive_digit=int(positive_digit),
    )



build_mnist_binary_stochastic_feedforward_problem = build_mnist_mixed_logit_problem


if __name__ == "__main__":
    
    dims = compute_mnist_mixed_logit_dims(DEFAULT_FEATURE_DIM, DEFAULT_R_DIM)
    print(
        "Default BSFNN dimensions: "
        f"feature_dim={dims[0]}, obs_dim={dims[1]}, "
        f"theta_dim={dims[2]}, z_dim={dims[3]}"
    )
