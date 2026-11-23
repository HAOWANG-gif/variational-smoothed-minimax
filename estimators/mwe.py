from __future__ import annotations
import time
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MWE:
    def __init__(
        self,
        y_obs,
        theta_init,
        sample_generator_cls,
        loss_func,
        train_params,
        lr=None,   
    ):
        # ==================================================
        
        # ==================================================
        self.y_obs = y_obs.detach().clone().to(device).float()

        if self.y_obs.dim() == 1:
            self.y_obs = self.y_obs.unsqueeze(1)

        self.theta_init = (
            theta_init
            .detach()
            .clone()
            .to(device)
            .float()
            .reshape(-1)
        )

        self.sample_generator_cls = sample_generator_cls
        self.loss_func = loss_func
        self.train_params = train_params

        self.n_obs = self.y_obs.shape[0]
        self.x_dim = self.y_obs.shape[1]
        self.theta_dim = self.theta_init.numel()

        # ==================================================
        
        # ==================================================
        self.sim_batch_size = int(
            train_params.get(
                "sim_batch_size",
                train_params.get("m", 10000)
            )
        )

        self.k_mc = int(
            train_params.get(
                "k_mc",
                train_params.get("K", 20)
            )
        )

        self.n_sim = int(train_params["n_sim"])
        self.n_init_sim = int(train_params.get("n_init_sim", 0))
        if self.n_sim < self.n_init_sim:
            raise ValueError('n_sim must be at least the initialization simulation count n_init_sim.')

        if lr is not None:
            self.learning_rate = float(lr)
        else:
            self.learning_rate = float(
                train_params.get(
                    "learning_rate",
                    train_params.get("lr", 1e-3)
                )
            )

        self.optimizer_name = str(train_params.get("optimizer", "adam")).lower()
        self.grad_clip = train_params.get("grad_clip", None)

        self.print_every = int(train_params.get("print_every", 100))
        self.verbose = bool(train_params.get("verbose", False))
        self.seed = train_params.get("seed", None)

        # ==================================================
        
        # ==================================================
        self.standardize_loss = bool(train_params.get("standardize_loss", False))

        self.obs_mean = self.y_obs.mean(dim=0, keepdim=True).detach()
        self.obs_std = (
            self.y_obs
            .std(dim=0, keepdim=True, unbiased=False)
            .clamp_min(1e-8)
            .detach()
        )

        # ==================================================
        
        # ==================================================
        
        self.param_low = train_params.get("param_low", None)
        self.param_high = train_params.get("param_high", None)

        self.param_low_np = self._to_numpy_bound(self.param_low)
        self.param_high_np = self._to_numpy_bound(self.param_high)

        self.param_low_tensor = self._to_tensor_bound(self.param_low_np)
        self.param_high_tensor = self._to_tensor_bound(self.param_high_np)

        self.project_final_theta = bool(train_params.get("project_final_theta", True))

        # ==================================================
        
        # ==================================================
        
        self.use_exact_1d_w1 = bool(train_params.get("use_exact_1d_w1", False))
        if self.use_exact_1d_w1:
            raise ValueError(
                'The Adam MWE implementation does not support use_exact_1d_w1=True because the NumPy sorting implementation is not differentiable. Set use_exact_1d_w1=False and use geomloss.SamplesLoss as loss_func.'
            )

        # ==================================================
        
        # ==================================================
        self.loss_history = []
        self.theta_history = []
        self.total_virtual_samples = self.n_init_sim
        self.fixed_eps_list = None

    # ==================================================
    # Utilities
    # ==================================================

    @staticmethod
    def _to_numpy_bound(bound):
        if bound is None:
            return None

        if isinstance(bound, torch.Tensor):
            return bound.detach().cpu().numpy().astype(np.float64).reshape(-1)

        return np.asarray(bound, dtype=np.float64).reshape(-1)

    @staticmethod
    def _to_tensor_bound(bound_np):
        if bound_np is None:
            return None

        return torch.tensor(
            bound_np,
            dtype=torch.float32,
            device=device,
        ).reshape(-1)

    def _set_seed_once(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

    def _project_theta_tensor_(self, theta):
        'Project theta tensor.'
        with torch.no_grad():
            if self.param_low_tensor is not None:
                theta.data = torch.maximum(theta.data, self.param_low_tensor)

            if self.param_high_tensor is not None:
                theta.data = torch.minimum(theta.data, self.param_high_tensor)

        return theta

    def _project_theta_np(self, theta_np):
        theta_np = np.asarray(theta_np, dtype=np.float64)

        if self.param_low_np is not None:
            theta_np = np.maximum(theta_np, self.param_low_np)

        if self.param_high_np is not None:
            theta_np = np.minimum(theta_np, self.param_high_np)

        return theta_np

    def _standardize_for_loss(self, y_true, y_sim):
        if self.standardize_loss:
            y_true_for_loss = (y_true - self.obs_mean) / self.obs_std
            y_sim_for_loss = (y_sim - self.obs_mean) / self.obs_std
        else:
            y_true_for_loss = y_true
            y_sim_for_loss = y_sim

        return y_true_for_loss, y_sim_for_loss

    def _generate_one_noise(self, n_samples, seed=None):
        'Generate one noise.'
        try:
            eps = self.sample_generator_cls.generate_noise(
                n_samples=n_samples,
                seed=seed,
            )
        except TypeError:
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)

                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            eps = self.sample_generator_cls.generate_noise(
                n_samples=n_samples,
            )

        return eps.detach().clone().to(device)

    def _make_fixed_eps_list(self):
        'Make fixed eps list.'
        eps_list = []

        base_seed = self.seed
        if base_seed is None:
            base_seed = int(torch.randint(0, 2**30, (1,)).item())

        for j in range(self.k_mc):
            eps_j = self._generate_one_noise(
                n_samples=self.sim_batch_size,
                seed=base_seed + 1009 * j,
            )
            eps_list.append(eps_j)

        return eps_list

    def _simulate_with_fixed_noise(self, theta_tensor, eps):
        'Simulate with fixed noise.'
        n_samples = eps.shape[0]
        try:
            y_sim = self.sample_generator_cls.static_sample_model(
                theta=theta_tensor,
                n_samples=n_samples,
                eps=eps,
            )
        except TypeError:
            generator = self.sample_generator_cls(
                theta_tensor,
                n_samples=n_samples,
            )
            y_sim = generator.generate_samples(eps=eps)

        y_sim = y_sim.to(device).float()

        if y_sim.dim() == 1:
            y_sim = y_sim.unsqueeze(1)

        if y_sim.shape[0] != n_samples:
            y_sim = y_sim.reshape(n_samples, -1)

        return y_sim

    def _loss_with_loss_func_tensor(self, y_sim):
        'Loss with loss func tensor.'
        y_true_for_loss, y_sim_for_loss = self._standardize_for_loss(
            y_true=self.y_obs,
            y_sim=y_sim,
        )

        loss = self.loss_func(y_true_for_loss, y_sim_for_loss)

        if not isinstance(loss, torch.Tensor):
            raise RuntimeError(
                f'The Adam implementation requires loss_func to return torch.Tensor; received {type(loss)}'
            )

        if loss.dim() > 0:
            loss = loss.mean()

        return loss

    def _compute_mewe_loss_tensor(self, theta):
        'Compute mewe loss tensor.'
        loss_terms = []

        remaining = self.n_sim - self.total_virtual_samples

        for eps in self.fixed_eps_list:
            if remaining <= 0:
                break

            n_samples = min(self.sim_batch_size, remaining)
            eps = eps[:n_samples]
            y_sim = self._simulate_with_fixed_noise(
                theta_tensor=theta,
                eps=eps,
            )

            self.total_virtual_samples += n_samples
            remaining -= n_samples

            loss_j = self._loss_with_loss_func_tensor(y_sim)
            loss_terms.append(loss_j)

        loss = torch.stack(loss_terms).mean()

        return loss

    # ==================================================
    # Main Adam training routine
    # ==================================================

    def train(self):
        'Train the estimator and return the standardized experiment outputs.'
        self._set_seed_once()

        start_time = time.time()

        # ==================================================
        
        # ==================================================
        self.fixed_eps_list = self._make_fixed_eps_list()

        # ==================================================
        
        # ==================================================
        theta = (
            self.theta_init
            .detach()
            .clone()
            .to(device)
            .float()
            .reshape(-1)
        )

        
        if self.param_low_np is not None or self.param_high_np is not None:
            theta_np = theta.detach().cpu().numpy().astype(np.float64)
            theta_np = self._project_theta_np(theta_np)
            theta = torch.tensor(
                theta_np,
                dtype=torch.float32,
                device=device,
            )

        theta = theta.requires_grad_(True)

        optimizer = torch.optim.Adam(
            [theta],
            lr=self.learning_rate,
        )

        best_loss = float("inf")
        best_theta = theta.detach().clone()
        print(
            f'\n========== Starting the Adam-MEWE / Wasserstein baseline ==========\noptimizer       = Adam\nk_mc            = {self.k_mc}\nsim_batch_size  = {self.sim_batch_size}\nn_sim           = {self.n_sim}\nlearning_rate   = {self.learning_rate}\nstandardize     = {self.standardize_loss}\nparam_low       = {self.param_low_np}\nparam_high      = {self.param_high_np}\n'
        )

        # ==================================================
        
        # ==================================================
        epoch = 0
        while self.total_virtual_samples < self.n_sim:
            epoch += 1
            optimizer.zero_grad()

            loss = self._compute_mewe_loss_tensor(theta)

            if not torch.isfinite(loss):
                print(f'⚠️ epoch={epoch}: loss is not finite; stopping training.')
                break

            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    [theta],
                    float(self.grad_clip),
                )

            optimizer.step()

            
            if self.param_low_np is not None or self.param_high_np is not None:
                self._project_theta_tensor_(theta)

            loss_val = float(loss.detach().cpu().item())
            theta_np_now = theta.detach().cpu().numpy().copy()

            self.loss_history.append(loss_val)
            self.theta_history.append(theta_np_now)

            if loss_val < best_loss:
                best_loss = loss_val
                best_theta = theta.detach().clone()

            if (
                self.verbose
                or epoch == 1
                or epoch % self.print_every == 0
                or self.total_virtual_samples == self.n_sim
            ):
                theta_str = "[" + ", ".join([f"{v:.6f}" for v in theta_np_now]) + "]"
                print(
                    f"Epoch {epoch:5d} | "
                    f"MEWE Loss = {loss_val:.8f} | "
                    f"Theta = {theta_str} | "
                    f"Virtual Samples = {self.total_virtual_samples}"
                )

        # ==================================================
        
        # ==================================================
        total_time = time.time() - start_time

        final_theta = best_theta.detach().to(device).float()

        if self.project_final_theta:
            theta_np = final_theta.detach().cpu().numpy().astype(np.float64)
            theta_np = self._project_theta_np(theta_np)
            final_theta = torch.tensor(
                theta_np,
                dtype=torch.float32,
                device=device,
            )

        self.theta_history = np.asarray(
            self.theta_history,
            dtype=np.float64,
        )

        print(
            f'\n========== Adam-MEWE / Wasserstein baseline finished ==========\nbest objective   = {best_loss:.8f}\ntime             = {total_time:.2f}s\nvirtual samples  = {self.total_virtual_samples}\nfinal theta      = {final_theta.detach().cpu().numpy()}\n'
        )

        return (
            final_theta.detach(),
            self.loss_history,
            self.theta_history,
            total_time,
            self.total_virtual_samples,
        )
