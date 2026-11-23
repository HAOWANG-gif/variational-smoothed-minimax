import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdaptiveIMLEParameterEstimator:
    def __init__(
            self,
            y_obs: torch.Tensor,
            theta_init: torch.Tensor,
            simulator_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            config: Optional[Dict] = None,
    ):
        self.y_obs = y_obs.detach().clone().float().to(device)

        if self.y_obs.dim() == 1:
            self.y_obs = self.y_obs.unsqueeze(1)

        self.theta = nn.Parameter(
            theta_init.clone().detach().float().to(device)
        )

        self.simulator_fn = simulator_fn

        self.n_samples, self.data_dim = self.y_obs.shape
        self.theta_dim = self.theta.numel()

        self.config = self._default_config()

        if config is not None:
            self.config.update(config)

        self._validate_config()
        self.n_sim = int(self.config["n_sim"])
        self.n_init_sim = int(self.config.get("n_init_sim", 0))
        if self.n_sim < self.n_init_sim:
            raise ValueError('n_sim must be at least the initialization simulation count n_init_sim.')

        self.seed = self.config.get("seed", None)
        self._set_seed_once()

        
        self.obs_mean = self.y_obs.mean(dim=0, keepdim=True).detach()
        self.obs_std = (
            self.y_obs.std(dim=0, keepdim=True, unbiased=False)
            .clamp_min(1e-8)
            .detach()
        )

        
        self.latent_dim = self.config["latent_dim"]

        self.latents = torch.randn(
            self.n_samples,
            self.latent_dim,
            device=device,
            dtype=self.y_obs.dtype,
        )

        
        self.pool_factor = self.config["pool_factor"]
        self.pool_size = int(self.n_samples * self.pool_factor)

        self.candidate_pool = torch.randn(
            self.pool_size,
            self.latent_dim,
            device=device,
            dtype=self.y_obs.dtype,
        )

        
        self.current_distances = torch.full(
            (self.n_samples,),
            float("inf"),
            device=device,
            dtype=self.y_obs.dtype,
        )

        initial_threshold = self.config["initial_threshold"]

        if initial_threshold is None:
            initial_threshold = float("inf")

        self.thresholds = torch.full(
            (self.n_samples,),
            float(initial_threshold),
            device=device,
            dtype=self.y_obs.dtype,
        )

        
        self.optimizer = torch.optim.Adam(
            [self.theta],
            lr=self.config["learning_rate"],
            weight_decay=0.0,
        )

        
        self.param_low = self.config.get("param_low", None)
        self.param_high = self.config.get("param_high", None)

        if self.param_low is not None:
            self.param_low = torch.as_tensor(
                self.param_low,
                dtype=torch.float32,
                device=device,
            )

        if self.param_high is not None:
            self.param_high = torch.as_tensor(
                self.param_high,
                dtype=torch.float32,
                device=device,
            )

        
        self.loss_history = []
        self.theta_history = []

        self.history = {
            "epoch": [],
            "loss": [],
            "recon_loss": [],
            "std_loss": [],
            "mean_distance": [],
            "mean_threshold": [],
            "update_fraction": [],
            "theta_norm": [],
            "max_theta_change": [],
            "total_virtual_samples": [],
        }

        self.total_virtual_samples = self.n_init_sim

        if self.config["verbose"]:
            self._print_init_info()

    def _default_config(self) -> Dict:
        return {
            
            "latent_dim": 1,

            
            "pool_factor": 10.0,

            # AdaIMLE curriculum
            "delta": 0.98,
            "initial_threshold": None,

            
            # "shrink_current": threshold = current_distance * (1 - change_coef)
            # "new_distance": threshold = new_distance
            "threshold_update_rule": "shrink_current",
            "change_coef": 0.15,

            
            "obs_batch_size": None,
            "learning_rate": 1e-3,
            "candidate_batch_size": 256,

            
            # "separate", "global", "none"
            "latent_search_standardization": "separate",

            
            "current_distance_standardize": False,

            
            "theta_loss_standardize": False,

            
            "refresh_candidate_pool_every": 1,

            
            
            
            "force_update_all_every": None,

            
            
            "lambda_std": 0.0,

            
            
            "std_target": "global",

            
            "grad_clip": None,

            # Parameter bounds
            "param_low": None,
            "param_high": None,

            # Random seed
            "seed": None,

            
            "verbose": True,
            "print_every": 10,
        }

    def _validate_config(self):
        assert 0 < self.config["delta"] < 1
        assert 0 < self.config["change_coef"] < 1
        assert self.config["pool_factor"] >= 1.0
        assert self.config["latent_dim"] > 0
        assert self.config["candidate_batch_size"] > 0
        assert self.config["n_sim"] >= 0

        assert self.config["latent_search_standardization"] in [
            "separate",
            "global",
            "none",
        ]

        assert self.config["threshold_update_rule"] in [
            "shrink_current",
            "new_distance",
        ]

        assert self.config["std_target"] in [
            "global",
            "batch",
        ]

    def _set_seed_once(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

    def _print_init_info(self):
        print("=" * 70)
    # ==================================================
    
    # ==================================================
    def _standardize_global(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.obs_mean) / self.obs_std

    @staticmethod
    def _standardize_self(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
        return (x - mean) / std

    def _prepare_current_distance_inputs(
            self,
            generated: torch.Tensor,
            target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config["current_distance_standardize"]:
            return self._standardize_global(generated), self._standardize_global(target)

        return generated, target

    def _prepare_theta_loss_inputs(
            self,
            generated: torch.Tensor,
            target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config["theta_loss_standardize"]:
            return self._standardize_global(generated), self._standardize_global(target)

        return generated, target

    def _project_theta_if_needed(self):
        if self.param_low is not None and self.param_high is not None:
            with torch.no_grad():
                self.theta.data = torch.max(
                    torch.min(self.theta.data, self.param_high),
                    self.param_low,
                )

    # ==================================================
    
    # ==================================================
    def _batch_simulate(
            self,
            thetas: torch.Tensor,
            latents: torch.Tensor,
    ) -> torch.Tensor:
        if thetas.dim() == 1:
            thetas = thetas.unsqueeze(0)

        if latents.dim() == 1:
            latents = latents.unsqueeze(0)

        batch_size = latents.shape[0]

        if thetas.shape[0] == 1 and batch_size > 1:
            thetas = thetas.expand(batch_size, -1)

        try:
            generated = self.simulator_fn(thetas, latents)
        except Exception:
            generated_list = []

            for i in range(batch_size):
                theta_i = thetas[i] if thetas.shape[0] > 1 else thetas[0]
                latent_i = latents[i]

                generated_i = self.simulator_fn(theta_i, latent_i)
                generated_list.append(generated_i.reshape(-1))

            generated = torch.stack(generated_list, dim=0)

        if generated.dim() == 1:
            generated = generated.unsqueeze(1)

        if generated.shape[0] != batch_size:
            generated = generated.reshape(batch_size, -1)

        if generated.shape[-1] != self.data_dim:
            generated = generated.reshape(batch_size, self.data_dim)

        return generated.float().to(device)

    def _refresh_candidate_pool(self):
        self.candidate_pool = torch.randn(
            self.pool_size,
            self.latent_dim,
            device=device,
            dtype=self.y_obs.dtype,
        )

    def _sample_observed_indices(self, max_size: Optional[int] = None) -> torch.Tensor:
        obs_batch_size = self.config["obs_batch_size"]

        if obs_batch_size is None or obs_batch_size >= self.n_samples:
            indices = torch.arange(self.n_samples, device=device)
        else:
            indices = torch.randperm(self.n_samples, device=device)[:obs_batch_size]

        if max_size is not None:
            indices = indices[:max_size]

        return indices

    # ==================================================
    # Phase 1: latent assignment
    # ==================================================
    @torch.no_grad()
    def _compute_current_distances(
            self,
            indices: torch.Tensor,
    ) -> torch.Tensor:
        latents = self.latents[indices]
        targets = self.y_obs[indices]

        generated = self._batch_simulate(
            self.theta.detach(),
            latents,
        )

        self.total_virtual_samples += latents.shape[0]

        generated_for_dist, targets_for_dist = self._prepare_current_distance_inputs(
            generated=generated,
            target=targets,
        )

        distances = torch.mean(
            (generated_for_dist - targets_for_dist) ** 2,
            dim=1,
        )

        return distances

    @torch.no_grad()
    def _find_nearest_latents(
            self,
            target_indices: torch.Tensor,
            max_simulations: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        k = len(target_indices)

        if k == 0:
            return (
                torch.empty(0, self.latent_dim, device=device),
                torch.empty(0, device=device),
            )

        target_data = self.y_obs[target_indices]
        mode = self.config["latent_search_standardization"]

        if mode == "separate":
            target_for_search = self._standardize_self(target_data)
        elif mode == "global":
            target_for_search = self._standardize_global(target_data)
        elif mode == "none":
            target_for_search = target_data
        else:
            raise ValueError(f'Unknown latent_search_standardization: {mode}')

        best_distances = torch.full(
            (k,),
            float("inf"),
            device=device,
            dtype=self.y_obs.dtype,
        )

        best_indices = torch.zeros(
            k,
            dtype=torch.long,
            device=device,
        )

        batch_size = self.config["candidate_batch_size"]
        search_size = self.pool_size
        if max_simulations is not None:
            search_size = min(search_size, max_simulations)

        n_batches = (search_size + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, search_size)

            candidate_batch = self.candidate_pool[start:end]

            candidate_data = self._batch_simulate(
                self.theta.detach(),
                candidate_batch,
            )

            self.total_virtual_samples += candidate_batch.shape[0]

            if mode == "separate":
                candidate_for_search = self._standardize_self(candidate_data)
            elif mode == "global":
                candidate_for_search = self._standardize_global(candidate_data)
            elif mode == "none":
                candidate_for_search = candidate_data
            else:
                raise ValueError(f'Unknown latent_search_standardization: {mode}')

            distances = torch.mean(
                (
                    target_for_search.unsqueeze(1)
                    - candidate_for_search.unsqueeze(0)
                ) ** 2,
                dim=2,
            )

            batch_best_dists, batch_best_indices = torch.min(
                distances,
                dim=1,
            )

            better = batch_best_dists < best_distances

            if torch.any(better):
                best_distances[better] = batch_best_dists[better]
                best_indices[better] = batch_best_indices[better] + start

        best_latents = self.candidate_pool[best_indices]

        return best_latents, best_distances

    def _phase1_latent_assignment(
            self,
            epoch: int,
            batch_indices: torch.Tensor,
    ) -> torch.Tensor:

        with torch.no_grad():
            current_distances = self._compute_current_distances(
                batch_indices
            )

            self.current_distances[batch_indices] = current_distances

            condition = (
                current_distances
                <= self.config["delta"] * self.thresholds[batch_indices]
            )

            # ==================================================
            
            # ==================================================
            force_every = self.config.get("force_update_all_every", None)

            force_update_all = (
                force_every is not None
                and force_every > 0
                and epoch % force_every == 0
            )

            if force_update_all:
                update_indices = batch_indices
            else:
                update_indices = batch_indices[condition]

            refresh_every = self.config["refresh_candidate_pool_every"]

            if len(update_indices) > 0:
                if refresh_every is not None and refresh_every > 0:
                    if epoch % refresh_every == 0:
                        self._refresh_candidate_pool()

                remaining = self.n_sim - self.total_virtual_samples
                phase2_reserve = len(batch_indices)
                candidate_budget = max(remaining - phase2_reserve, 0)

                if candidate_budget == 0:
                    return torch.empty(0, dtype=torch.long, device=device)

                new_latents, new_distances = self._find_nearest_latents(
                    update_indices,
                    max_simulations=candidate_budget,
                )

                self.latents[update_indices] = new_latents

                if self.config["threshold_update_rule"] == "new_distance":
                    self.thresholds[update_indices] = new_distances

                elif self.config["threshold_update_rule"] == "shrink_current":
                    
                    if force_update_all:
                        current_for_update = current_distances
                    else:
                        current_for_update = current_distances[condition]

                    self.thresholds[update_indices] = (
                        current_for_update * (1.0 - self.config["change_coef"])
                    )

                else:
                    raise ValueError(
                        f"Unknown threshold_update_rule: {self.config['threshold_update_rule']}"
                    )

                self.current_distances[update_indices] = new_distances

            return update_indices

    # ==================================================
    # Phase 2: theta optimization
    # ==================================================
    def _phase2_parameter_optimization(
            self,
            batch_indices: torch.Tensor,
    ) -> Tuple[float, float, float]:

        latents = self.latents[batch_indices]
        targets = self.y_obs[batch_indices]

        generated = self._batch_simulate(
            self.theta,
            latents,
        )

        self.total_virtual_samples += latents.shape[0]

        generated_for_loss, targets_for_loss = self._prepare_theta_loss_inputs(
            generated=generated,
            target=targets,
        )

        
        recon_loss = F.mse_loss(
            generated_for_loss,
            targets_for_loss,
        )

        
        lambda_std = self.config.get("lambda_std", 0.0)

        if lambda_std > 0:
            generated_std = generated.std(dim=0, unbiased=False)

            if self.config["std_target"] == "global":
                target_std = self.y_obs.std(dim=0, unbiased=False)
            elif self.config["std_target"] == "batch":
                target_std = targets.std(dim=0, unbiased=False)
            else:
                raise ValueError(f"Unknown std_target: {self.config['std_target']}")

            std_loss = torch.mean((generated_std - target_std) ** 2)
            loss = recon_loss + lambda_std * std_loss
        else:
            std_loss = torch.tensor(0.0, device=device)
            loss = recon_loss

        self.optimizer.zero_grad()
        loss.backward()

        if self.config["grad_clip"] is not None:
            torch.nn.utils.clip_grad_norm_(
                [self.theta],
                self.config["grad_clip"],
            )

        self.optimizer.step()

        self._project_theta_if_needed()

        return loss.item(), recon_loss.item(), std_loss.item()

    # ==================================================
    # Train
    # ==================================================
    def train(self, num_epochs: Optional[int] = None):
        start_time = time.time()

        prev_theta = self.theta.detach().clone()

        epoch = 0
        while self.total_virtual_samples < self.n_sim:
            epoch += 1
            epoch_start = time.time()

            remaining = self.n_sim - self.total_virtual_samples
            if remaining == 1:
                final_index = self._sample_observed_indices(max_size=1)
                self._compute_current_distances(final_index)
                break

            batch_indices = self._sample_observed_indices(
                max_size=remaining // 2
            )

            update_indices = self._phase1_latent_assignment(
                epoch=epoch,
                batch_indices=batch_indices,
            )

            loss, recon_loss, std_loss = self._phase2_parameter_optimization(
                batch_indices=batch_indices,
            )

            curr_theta = self.theta.detach()
            max_change = torch.max(torch.abs(curr_theta - prev_theta)).item()

            prev_theta = curr_theta.clone()

            self.loss_history.append(loss)
            self.theta_history.append(
                curr_theta.cpu().numpy().copy()
            )

            finite_distances = self.current_distances[
                torch.isfinite(self.current_distances)
            ]

            if len(finite_distances) > 0:
                mean_distance = finite_distances.mean().item()
            else:
                mean_distance = float("inf")

            finite_thresholds = self.thresholds[
                torch.isfinite(self.thresholds)
            ]

            if len(finite_thresholds) > 0:
                mean_threshold = finite_thresholds.mean().item()
            else:
                mean_threshold = float("inf")

            update_fraction = len(update_indices) / max(len(batch_indices), 1)

            self.history["epoch"].append(epoch)
            self.history["loss"].append(loss)
            self.history["recon_loss"].append(recon_loss)
            self.history["std_loss"].append(std_loss)
            self.history["mean_distance"].append(mean_distance)
            self.history["mean_threshold"].append(mean_threshold)
            self.history["update_fraction"].append(update_fraction)
            self.history["theta_norm"].append(torch.norm(curr_theta).item())
            self.history["max_theta_change"].append(max_change)
            self.history["total_virtual_samples"].append(self.total_virtual_samples)

            if (
                    self.config["verbose"]
                    and (
                        epoch == 1
                        or epoch % self.config["print_every"] == 0
                    )
            ):
                epoch_time = time.time() - epoch_start

                formatted_theta = "[" + ", ".join(
                    [f"{val:.4f}" for val in curr_theta.cpu().numpy()]
                ) + "]"

                print(
                    f"Epoch {epoch:5d} | "
                    f"Loss = {loss:.6f} | "
                    f"Recon = {recon_loss:.6f} | "
                    f"StdLoss = {std_loss:.6f} | "
                    f"Updated = {len(update_indices):4d}/{len(batch_indices):4d} | "
                    f"MeanDist = {mean_distance:.6f} | "
                    f"MaxDelta = {max_change:.6f} | "
                    f"Theta = {formatted_theta} | "
                    f"Virtual = {self.total_virtual_samples} | "
                    f"Time = {epoch_time:.2f}s"
                )

        total_time = time.time() - start_time

        self.theta_history = np.array(self.theta_history)

        if self.config["verbose"]:
            print(
                f'Total training time: {total_time:.2f}s | total simulated-sample count: {self.total_virtual_samples}'
            )

            print(f'Final parameter estimate: {self.theta.detach().cpu().numpy()}')

        return (
            self.theta.detach(),
            self.loss_history,
            self.theta_history,
            total_time,
            self.total_virtual_samples
        )

    def estimate_parameters(self) -> torch.Tensor:
        return self.theta.detach().clone()

    def generate_data(
            self,
            num_samples: Optional[int] = None,
            latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if num_samples is None:
            num_samples = self.n_samples
        if latents is None:
            latents = torch.randn(
                num_samples,
                self.latent_dim,
                device=device,
                dtype=self.y_obs.dtype,
            )
        with torch.no_grad():
            generated = self._batch_simulate(
                self.theta.detach(),
                latents,
            )
        return generated
    def get_training_history(self) -> Dict:
        return self.history.copy()
