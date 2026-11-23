"""The original VSM trainer with stage diagnostics and optional overall stopping.

Initialization, sample generation, objectives, optimizer updates and budget
allocation come from estimators.vsm. The minimax loop below preserves the
original operations and random-call order while counting inner iterations.
"""

import time

import torch

from estimators.vsm import AnnealedMinimaxTrainer as _OriginalTrainer
from .stopping import StageStoppingController, validate_stop_config


class AnnealedMinimaxTrainer(_OriginalTrainer):
    def __init__(self, y_obs, init_theta, sample_generator_cls, U_net, train_config):
        validate_stop_config(train_config)
        super().__init__(y_obs, init_theta, sample_generator_cls, U_net, train_config)
        self._stopping = StageStoppingController(train_config, epsilon_initial=self.epsilon_0)
        self.overall_stop_mode = self._stopping.mode
        self.stage_history = self._stopping.stage_history
        self.termination_report = self._stopping.termination_report
        self._last_inner_diagnostics = {}

    def _minimax_solver(self, current_epsilon):
        consecutive_small_steps = 0
        prev_theta = self.theta.detach().clone()
        last_loss_theta = None
        inner_iterations = 0
        inner_early_stop_hit = False
        inner_stop_reason = "minimax_iters"

        for i in range(self.minimax_iters):
            remaining = self.n_sim - self.total_virtual_samples
            if remaining <= 0:
                inner_stop_reason = "simulation_budget"
                break

            n_inner = min(self.k_inner, max(remaining - 1, 0))
            n_calls = n_inner + 1
            iteration_budget = min(remaining, self.m2 * n_calls)
            base_batch = iteration_budget // n_calls
            extra = iteration_budget - base_batch * n_calls

            self.U_psi.train()
            for inner_idx in range(n_inner):
                n_fake = base_batch + (1 if inner_idx < extra else 0)
                self.opt_psi.zero_grad()
                u_real, u_fake, _, _ = self._get_u_scores(
                    current_epsilon, detach_theta=True, n_fake=n_fake,
                )
                safe_u_fake_psi = torch.clamp(u_fake - 1.0, max=10.0)
                loss_psi = -(u_real.mean() - torch.exp(safe_u_fake_psi).mean())
                loss_psi.backward()
                self.opt_psi.step()

            theta_batch_size = base_batch + max(extra - n_inner, 0)
            self.opt_theta.zero_grad()
            u_real, u_fake, x_bar, x_tilde = self._get_u_scores(
                current_epsilon, detach_theta=False, n_fake=theta_batch_size,
            )
            abs_diff = torch.abs(x_bar.unsqueeze(1) - x_tilde.unsqueeze(0))
            min_diff_per_dim, _ = torch.min(abs_diff, dim=1)
            max_diff_per_sample, _ = torch.max(min_diff_per_dim, dim=1)
            loss_reg = torch.mean(max_diff_per_sample)
            safe_u_fake_theta = torch.clamp(u_fake - 1.0, max=10.0)
            loss_theta_main = u_real.mean() - torch.exp(safe_u_fake_theta).mean()
            loss_theta = loss_theta_main + self.lambda_reg * loss_reg
            loss_theta.backward()
            self.opt_theta.step()
            last_loss_theta = loss_theta.item()

            curr_theta = self.theta.detach()
            max_change = torch.max(torch.abs(curr_theta - prev_theta)).item()
            if max_change < self.early_stop_tol:
                consecutive_small_steps += 1
            else:
                consecutive_small_steps = 0
            prev_theta = curr_theta.clone()

            if i % 10 == 0 and n_inner > 0:
                self.loss_history_psi.append(loss_psi.item())
            inner_iterations += 1
            if consecutive_small_steps >= self.early_stop_patience:
                print(f'    [Early stop] For {self.early_stop_patience} consecutive iterations, the maximum parameter change was < {self.early_stop_tol}')
                inner_early_stop_hit = True
                inner_stop_reason = "early_stop"
                break

        if last_loss_theta is not None:
            self.loss_history_theta.append(last_loss_theta)
        if not inner_early_stop_hit and self.total_virtual_samples >= self.n_sim:
            inner_stop_reason = "simulation_budget"
        inner_iterations_saved = 0
        if inner_early_stop_hit:
            remaining = max(self.n_sim - self.total_virtual_samples, 0)
            full_iteration_cost = self.m2 * (self.k_inner + 1)
            budget_iterations_left = (remaining + full_iteration_cost - 1) // full_iteration_cost
            # A tail batch is one possible further iteration. An early-stop
            # hit at either limit saves no iterations, even if the other
            # limit still has room.
            inner_iterations_saved = min(
                max(self.minimax_iters - inner_iterations, 0), budget_iterations_left,
            )
        self._last_inner_diagnostics = {
            "inner_iterations": inner_iterations,
            "inner_early_stop_hit": inner_early_stop_hit,
            "inner_iterations_saved": inner_iterations_saved,
            "inner_stop_reason": inner_stop_reason,
        }
        return last_loss_theta

    def train(self):
        start_time = time.time()
        fallback_reason = None

        while self.total_virtual_samples < self.n_sim and not self._stopping.should_stop:
            # The raw theta at the stage start is the preceding stage endpoint.
            theta_start = self.theta.detach().cpu().tolist()
            current_loss = self._minimax_solver(self._stopping.epsilon)
            if current_loss is None:
                fallback_reason = "solver_no_progress"
                break

            current_theta_val = self.theta.detach().cpu().numpy().copy()
            self.theta_history.append(current_theta_val)
            formatted_theta = "[" + ", ".join([f"{val:.4f}" for val in current_theta_val]) + "]"
            print(f'--> Current theta: {formatted_theta}, Obj: {current_loss:.6f}')
            self._stopping.record_stage(
                theta_start=theta_start,
                theta_end=current_theta_val.tolist(),
                n_sim=self.total_virtual_samples,
                **self._last_inner_diagnostics,
            )

        self._stopping.finalize(self.total_virtual_samples, fallback_reason=fallback_reason)
        total_time = time.time() - start_time
        return self.theta, self.loss_history_theta, self.theta_history, total_time, self.total_virtual_samples
