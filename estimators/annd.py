
import time
import math
import numpy as np
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import minimize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================

# ============================================================

@dataclass
class OptimResult:
    theta: np.ndarray
    value: float
    success: bool
    message: str
    nit: int
    nfev: int


class _SimulationBudgetExhausted(Exception):
    pass


# ============================================================

# ============================================================

def _to_unbounded(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    'To unbounded.'
    z = []

    for xi, lo, hi in zip(x, lb, ub):
        finite_lo = np.isfinite(lo)
        finite_hi = np.isfinite(hi)

        if finite_lo and finite_hi and lo == hi:
            continue

        if finite_lo and finite_hi:
            if xi <= lo:
                z.append(-math.pi / 2.0)
            elif xi >= hi:
                z.append(math.pi / 2.0)
            else:
                scaled = 2.0 * (xi - lo) / (hi - lo) - 1.0
                scaled = float(np.clip(scaled, -1.0, 1.0))
                z.append(2.0 * math.pi + math.asin(scaled))

        elif finite_lo:
            z.append(math.sqrt(max(xi - lo, 0.0)))

        elif finite_hi:
            z.append(math.sqrt(max(hi - xi, 0.0)))

        else:
            z.append(float(xi))

    return np.asarray(z, dtype=float)


def _from_unbounded(z: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    'From unbounded.'
    x = np.zeros_like(lb, dtype=float)
    k = 0

    for i, (lo, hi) in enumerate(zip(lb, ub)):
        finite_lo = np.isfinite(lo)
        finite_hi = np.isfinite(hi)

        if finite_lo and finite_hi and lo == hi:
            x[i] = lo

        elif finite_lo and finite_hi:
            xi = (math.sin(float(z[k])) + 1.0) / 2.0
            x[i] = lo + xi * (hi - lo)
            x[i] = min(max(x[i], lo), hi)
            k += 1

        elif finite_lo:
            x[i] = lo + float(z[k]) ** 2
            k += 1

        elif finite_hi:
            x[i] = hi - float(z[k]) ** 2
            k += 1

        else:
            x[i] = float(z[k])
            k += 1

    return x


def fminsearchcon(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    xatol: float = 1e-4,
    fatol: float = 1e-4,
) -> OptimResult:
    'Fminsearchcon.'
    x0 = np.asarray(x0, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    if x0.shape != lb.shape or x0.shape != ub.shape:
        raise ValueError('x0, lb, and ub must have matching shapes.')

    x0 = np.where(np.isfinite(lb), np.maximum(x0, lb), x0)
    x0 = np.where(np.isfinite(ub), np.minimum(x0, ub), x0)

    z0 = _to_unbounded(x0, lb, ub)

    if z0.size == 0:
        value = float(fun(x0))
        return OptimResult(
            theta=x0,
            value=value,
            success=True,
            message="all parameters fixed",
            nit=0,
            nfev=1,
        )

    def wrapped(z: np.ndarray) -> float:
        theta = _from_unbounded(z, lb, ub)
        value = fun(theta)

        if not np.isfinite(value):
            return 1e100

        return float(value)

    res = minimize(
        wrapped,
        z0,
        method="Nelder-Mead",
        options={
            "xatol": xatol,
            "fatol": fatol,
            "disp": False,
        },
    )

    theta_hat = _from_unbounded(res.x, lb, ub)
    value = float(wrapped(res.x))

    return OptimResult(
        theta=theta_hat,
        value=value,
        success=bool(res.success),
        message=str(res.message),
        nit=int(res.nit),
        nfev=int(res.nfev),
    )


# ============================================================

# ============================================================

class BatchedEnsembleDiscriminator(nn.Module):
    'BatchedEnsembleDiscriminator.'

    def __init__(self, g: int, input_dim: int, num_hidden: int):
        super().__init__()

        self.g = g
        self.input_dim = input_dim
        self.num_hidden = num_hidden

        k1 = np.sqrt(1.0 / input_dim)
        self.w1 = nn.Parameter(
            torch.empty(g, num_hidden, input_dim).uniform_(-k1, k1)
        )
        self.b1 = nn.Parameter(
            torch.empty(g, num_hidden).uniform_(-k1, k1)
        )

        k2 = np.sqrt(1.0 / num_hidden)
        self.w2 = nn.Parameter(
            torch.empty(g, 1, num_hidden).uniform_(-k2, k2)
        )
        self.b2 = nn.Parameter(
            torch.empty(g, 1).uniform_(-k2, k2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        'Forward.'
        h = torch.einsum("nd,ghd->ngh", x, self.w1) + self.b1
        h = torch.tanh(h)

        logits = torch.einsum("ngh,gkh->ngk", h, self.w2) + self.b2

        return logits.squeeze(-1)


# ============================================================

# ============================================================

class AdversarialEstimator:
    'AdversarialEstimator.'

    def __init__(self, y_obs, init_theta, sample_generator_cls, train_config):
        self.y_obs = self._as_2d_tensor(y_obs).to(device)

        self.init_theta = init_theta.detach().cpu().numpy().astype(float)
        self.sample_generator_cls = sample_generator_cls

        
        self.g = train_config.get("g", 30)
        self.num_hidden = train_config.get("num_hidden", 10)
        self.reg_l2 = train_config.get("reg_l2", 0.01)

        
        self.m2 = train_config.get("m2", 512)

        
        self.xatol = train_config.get("xatol", 1e-3)
        self.fatol = train_config.get("fatol", 1e-3)
        self.n_sim = int(train_config["n_sim"])
        self.n_init_sim = int(train_config.get("n_init_sim", 0))
        if self.n_sim < self.n_init_sim:
            raise ValueError('n_sim must be at least the initialization simulation count n_init_sim.')

        
        
        
        self.S = train_config.get("S", 1)

        
        
        self.disc_optimizer = train_config.get("disc_optimizer", "adam")
        self.disc_lr = train_config.get("disc_lr", 0.01)
        self.disc_train_iter = train_config.get("disc_train_iter", 80)

        
        
        
        self.outer_optimizer = str(
            train_config.get("outer_optimizer", "nelder_mead")
        ).lower()
        self.theta_lr = float(train_config.get("theta_lr", 1e-2))
        self.theta_grad_clip = float(train_config.get("theta_grad_clip", 5.0))

        
        self.lbfgs_max_iter = train_config.get("lbfgs_max_iter", 1000)

        # Random seed
        self.seed = int(train_config.get("seed", 12345))

        # Parameter bounds
        dim_theta = self.init_theta.shape[0]
        self.bounds_lb = np.asarray(
            train_config.get("bounds_lb", [-10.0] * dim_theta),
            dtype=float,
        )
        self.bounds_ub = np.asarray(
            train_config.get("bounds_ub", [10.0] * dim_theta),
            dtype=float,
        )

        
        self.theta_history = []
        self.loss_history = []
        self.n_data = self.y_obs.shape[0]

        
        self.total_virtual_samples = self.n_init_sim

        
        self.objective_cache = {}

    @staticmethod
    def _as_2d_tensor(x) -> torch.Tensor:
        'As 2d tensor.'
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.detach().float()

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if x.ndim != 2:
            raise ValueError(
                f'Sample data must be two-dimensional (N, D), got shape={tuple(x.shape)}'
            )

        return x

    @staticmethod
    def _move_noise_to_device(eps):
        'Move noise to device.'
        if isinstance(eps, torch.Tensor):
            return eps.to(device)

        return eps

    def _reset_torch_seed(self, seed: int):
        'Reset torch seed.'
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _make_labels(self, m_fake: int) -> torch.Tensor:
        'Make labels.'
        labels = torch.cat(
            [
                torch.ones(self.n_data, 1, device=device),
                torch.zeros(m_fake, 1, device=device),
            ],
            dim=0,
        )

        labels = labels.expand(-1, self.g)

        return labels

    def _train_nnd_ensemble(
        self,
        y_fake_tensor: torch.Tensor,
        seed_offset: int = 0,
        return_model: bool = False,
    ):
        'Train nnd ensemble.'
        y_fake_tensor = self._as_2d_tensor(y_fake_tensor).to(device)

        if y_fake_tensor.shape[1] != self.y_obs.shape[1]:
            raise ValueError(
                f'Observed and simulated sample dimensions do not match: y_obs.shape={tuple(self.y_obs.shape)}, y_fake.shape={tuple(y_fake_tensor.shape)}'
            )

        inputs = torch.cat([self.y_obs, y_fake_tensor], dim=0)
        input_dim = inputs.shape[1]
        m_fake = y_fake_tensor.shape[0]

        labels = self._make_labels(m_fake)

        
        self._reset_torch_seed(self.seed + seed_offset)

        model = BatchedEnsembleDiscriminator(
            g=self.g,
            input_dim=input_dim,
            num_hidden=self.num_hidden,
        ).to(device)

        if self.disc_optimizer.lower() == "lbfgs":
            optimizer = optim.LBFGS(
                model.parameters(),
                lr=self.disc_lr,
                max_iter=self.lbfgs_max_iter,
                line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer.zero_grad()

                logits = model(inputs)

                # bce_loss = F.binary_cross_entropy_with_logits(
                #     logits,
                #     labels,
                #     reduction="none",
                # ).mean(dim=0)
                logits_real = logits[:self.n_data]
                logits_fake = logits[self.n_data:]

                loss_real = F.binary_cross_entropy_with_logits(
                    logits_real,
                    torch.ones_like(logits_real),
                    reduction="none",
                ).mean(dim=0)

                loss_fake = F.binary_cross_entropy_with_logits(
                    logits_fake,
                    torch.zeros_like(logits_fake),
                    reduction="none",
                ).mean(dim=0)

                bce_loss = loss_real + loss_fake

                l2_reg = (
                    (model.w1 ** 2).sum(dim=(1, 2))
                    + (model.b1 ** 2).sum(dim=1)
                    + (model.w2 ** 2).sum(dim=(1, 2))
                    + (model.b2 ** 2).sum(dim=1)
                )

                total_loss = (
                    bce_loss
                    + (self.reg_l2 / 2.0) * l2_reg
                ).sum()

                total_loss.backward()
                return total_loss

            optimizer.step(closure)

        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.disc_lr,
            )

            for _ in range(self.disc_train_iter):
                optimizer.zero_grad()

                logits = model(inputs)

                # bce_loss = F.binary_cross_entropy_with_logits(
                #     logits,
                #     labels,
                #     reduction="none",
                # ).mean(dim=0)
                logits_real = logits[:self.n_data]
                logits_fake = logits[self.n_data:]

                loss_real = F.binary_cross_entropy_with_logits(
                    logits_real,
                    torch.ones_like(logits_real),
                    reduction="none",
                ).mean(dim=0)

                loss_fake = F.binary_cross_entropy_with_logits(
                    logits_fake,
                    torch.zeros_like(logits_fake),
                    reduction="none",
                ).mean(dim=0)

                bce_loss = loss_real + loss_fake

                l2_reg = (
                    (model.w1 ** 2).sum(dim=(1, 2))
                    + (model.b1 ** 2).sum(dim=1)
                    + (model.w2 ** 2).sum(dim=(1, 2))
                    + (model.b2 ** 2).sum(dim=1)
                )

                total_loss = (
                    bce_loss
                    + (self.reg_l2 / 2.0) * l2_reg
                ).sum()

                total_loss.backward()
                optimizer.step()

        with torch.no_grad():
            logits_all = model(inputs)

            # D_all shape: (n_data + m_fake, g)
            D_all = torch.sigmoid(logits_all)

            
            D_avg = D_all.mean(dim=1, keepdim=True)

            D_X = D_avg[:self.n_data]
            D_Y = D_avg[self.n_data:]

            eps = 1e-8

            advn_value = (
                torch.mean(torch.log(D_X + eps))
                + torch.mean(torch.log(1.0 - D_Y + eps))
            )

        advn_value = float(advn_value.item())
        if return_model:
            return advn_value, model
        return advn_value

    def _train_with_adversarial_adam(self):
        'Train with adversarial adam.'
        start_time = time.time()

        theta = nn.Parameter(
            torch.tensor(self.init_theta, dtype=torch.float32, device=device)
        )
        theta_optimizer = optim.Adam([theta], lr=self.theta_lr)
        lb = torch.as_tensor(self.bounds_lb, dtype=torch.float32, device=device)
        ub = torch.as_tensor(self.bounds_ub, dtype=torch.float32, device=device)

        iteration = 0
        while self.total_virtual_samples < self.n_sim:
            remaining = self.n_sim - self.total_virtual_samples
            n_samples = min(self.m2, remaining)
            if n_samples <= 0:
                break

            
            self._reset_torch_seed(self.seed + 1000000 + iteration)
            eps = self.sample_generator_cls.generate_noise(n_samples=n_samples)
            eps = self._move_noise_to_device(eps)

            y_fake = self.sample_generator_cls.static_sample_model(
                theta=theta,
                n_samples=n_samples,
                eps=eps,
            )
            if not isinstance(y_fake, torch.Tensor):
                y_fake = torch.as_tensor(y_fake, dtype=torch.float32, device=device)
            else:
                y_fake = y_fake.to(device).float()
            if y_fake.ndim == 1:
                y_fake = y_fake.reshape(-1, 1)
            if y_fake.ndim != 2:
                raise ValueError(
                    f'Simulated samples must be two-dimensional (N, D), got shape={tuple(y_fake.shape)}'
                )
            self.total_virtual_samples += int(y_fake.shape[0])

            
            
            advn_value, discriminator = self._train_nnd_ensemble(
                y_fake_tensor=y_fake.detach(),
                seed_offset=iteration,
                return_model=True,
            )

            theta_before_update = theta.detach().cpu().numpy().copy()
            self.theta_history.append(theta_before_update)
            self.loss_history.append(advn_value)

            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)

            theta_optimizer.zero_grad()
            logits_fake = discriminator(y_fake)
            generator_loss = F.softplus(-logits_fake).mean()
            generator_loss.backward()

            if self.theta_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([theta], self.theta_grad_clip)

            theta_optimizer.step()
            with torch.no_grad():
                theta.copy_(torch.maximum(torch.minimum(theta, ub), lb))

            iteration += 1

        total_time = time.time() - start_time

        
        
        final_theta_np = theta.detach().cpu().numpy().copy()
        final_value = self.loss_history[-1] if self.loss_history else float("nan")

        final_theta_tensor = torch.tensor(
            final_theta_np,
            dtype=torch.float32,
            device=device,
        )

        print(
            f'--> AdvN Adam point estimation finished | objective: {final_value:.6f} | iterations: {iteration}'
        )
        print(f'--> Total simulated-sample count: {self.total_virtual_samples} models')

        return (
            final_theta_tensor,
            self.loss_history,
            self.theta_history,
            total_time,
            self.total_virtual_samples,
        )

    def train(self):
        if self.outer_optimizer in {"adam", "adversarial_adam"}:
            return self._train_with_adversarial_adam()

        start_time = time.time()

        # ====================================================
        
        # ====================================================
        fixed_eps_list = []

        for _ in range(self.S):
            eps = self.sample_generator_cls.generate_noise(
                n_samples=self.m2
            )
            eps = self._move_noise_to_device(eps)
            fixed_eps_list.append(eps)

        # ====================================================
        
        # ====================================================
        def objective(theta_np: np.ndarray) -> float:
            theta_np = np.asarray(theta_np, dtype=float)

            if self.total_virtual_samples >= self.n_sim:
                raise _SimulationBudgetExhausted

            
            key = tuple(np.round(theta_np, 10))

            if key in self.objective_cache:
                return self.objective_cache[key]

            theta_tensor = torch.tensor(
                theta_np,
                dtype=torch.float32,
                device=device,
            )

            loss_vals = []

            
            for s, fixed_eps in enumerate(fixed_eps_list):
                remaining = self.n_sim - self.total_virtual_samples
                if remaining <= 0:
                    break

                n_samples = min(self.m2, remaining)
                fixed_eps_current = fixed_eps[:n_samples]
                with torch.no_grad():
                    y_fake = self.sample_generator_cls.static_sample_model(
                        theta=theta_tensor,
                        n_samples=n_samples,
                        eps=fixed_eps_current,
                    )

                y_fake = self._as_2d_tensor(y_fake).to(device)

                self.total_virtual_samples += y_fake.shape[0]

                loss_val = self._train_nnd_ensemble(
                    y_fake_tensor=y_fake,
                    seed_offset=1000 * s,
                )

                loss_vals.append(loss_val)

            avg_loss_val = float(np.mean(loss_vals))

            self.theta_history.append(theta_np.copy())
            self.loss_history.append(avg_loss_val)

            self.objective_cache[key] = avg_loss_val

            return avg_loss_val

        # ====================================================
        
        # ====================================================
        restart_theta = self.init_theta.copy()
        res = None

        while self.total_virtual_samples < self.n_sim:
            self.objective_cache.clear()
            try:
                res = fminsearchcon(
                    fun=objective,
                    x0=restart_theta,
                    lb=self.bounds_lb,
                    ub=self.bounds_ub,
                    xatol=self.xatol,
                    fatol=self.fatol,
                )
                restart_theta = res.theta.copy()
            except _SimulationBudgetExhausted:
                break

        total_time = time.time() - start_time

        if self.loss_history:
            best_idx = int(np.argmin(self.loss_history))
            final_theta_np = self.theta_history[best_idx]
            final_value = self.loss_history[best_idx]
        else:
            final_theta_np = self.init_theta.copy()
            final_value = float("nan")

        final_theta_tensor = torch.tensor(
            final_theta_np,
            dtype=torch.float32,
            device=device,
        )

        print(
            f'--> AdvN point estimation finished | final theta: {final_theta_np} | Obj: {final_value:.6f}'
        )

        print(
            f'--> Total simulated-sample count: {self.total_virtual_samples} models'
        )

        return (
            final_theta_tensor,
            self.loss_history,
            self.theta_history,
            total_time,
            self.total_virtual_samples,
        )
