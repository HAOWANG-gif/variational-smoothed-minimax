
import os
import sys
import time
from datetime import datetime


import numpy as np
from scipy.stats import qmc


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AnnealedMinimaxTrainer:
    def __init__(self, y_obs, init_theta, sample_generator_cls, U_net, train_config):
        
        self.y_obs = y_obs.to(device)
        self.theta = init_theta.clone().detach().to(device).requires_grad_(True)
        self.sample_generator_cls = sample_generator_cls  
        self.U_psi = U_net.to(device)  
        
        self.n_data = self.y_obs.shape[0]
        self.yasuo = train_config.get("yasuo", 1)  
        
        if self.n_data > 1:
            
            max_std = torch.std(self.y_obs, dim=0).max().item()
            
            self.epsilon_0 = max(max_std, 1e-4)
            self.epsilon_0 = self.epsilon_0/self.yasuo
            print(f'Automatically selected initial bandwidth epsilon_0 = {self.epsilon_0:.6f} (based on the largest marginal standard deviation of the observations)')
        else:
            
            self.epsilon_0 = train_config.get("epsilon_0", 1.0)
            print(f'Only one observation is available, so variance cannot be estimated; using default bandwidth epsilon_0 = {self.epsilon_0}')

        
        self.gamma = train_config.get("gamma", 0.5)
        self.n_sim = int(train_config["n_sim"])
        self.n_init_sim = int(train_config.get("n_init_sim", 0))
        if self.n_sim < self.n_init_sim:
            raise ValueError('n_sim must be at least the initialization simulation count n_init_sim.')

        
        self.m1 = train_config.get("m1", 256)  
        self.m2 = train_config.get("m2", 256)  
        self.k_inner = train_config.get("k_inner", 1)  
        self.alpha_theta = train_config.get("lr_theta", 1e-3)  
        self.alpha_psi = train_config.get("lr_psi", 5e-3)  
        self.minimax_iters = train_config.get("minimax_iters", 200)  
        self.lambda_reg = train_config.get("lambda_reg", 0.5)

        
        self.early_stop_patience = train_config.get("early_stop_patience", 1000)  
        self.early_stop_tol = train_config.get("early_stop_tol", 0.001)  

        # Training history
        self.theta_history = []  
        self.loss_history_theta = []  
        self.loss_history_psi = []  
        self.total_virtual_samples = self.n_init_sim
        # Initialize optimizers
        self.opt_psi = optim.Adam(self.U_psi.parameters(), lr=self.alpha_psi)  
        self.opt_theta = optim.Adam([self.theta], lr=self.alpha_theta)  

    def _sample_smoothed_empirical(self, epsilon, n_samples):
        'Sample smoothed empirical.'
        
        if n_samples > self.n_data:
            idx = torch.randint(0, self.n_data, (n_samples,), device=device)

        
        else:
            if not hasattr(self, "_perm") or self._perm is None or self._perm_ptr + n_samples > self.n_data:
                self._perm = torch.randperm(self.n_data, device=device)
                self._perm_ptr = 0

            idx = self._perm[self._perm_ptr: self._perm_ptr + n_samples]
            self._perm_ptr += n_samples

        
        x_base = self.y_obs[idx]

        
        noise = torch.randn_like(x_base) * epsilon

        return x_base + noise

    def _get_u_scores(self, epsilon, detach_theta=False, n_fake=None):
        current_theta = self.theta.detach() if detach_theta else self.theta
        if n_fake is None:
            n_fake = self.m2

        x_bar = self._sample_smoothed_empirical(epsilon, self.m1)
        eps = self.sample_generator_cls.generate_noise(n_samples=n_fake)
        x_tilde = self.sample_generator_cls.static_sample_model(
            theta=current_theta,
            n_samples=n_fake,
            eps=eps
        )
        self.total_virtual_samples += n_fake
        theta_real_batch = current_theta.unsqueeze(0).expand(self.m1, -1)
        theta_fake_batch = current_theta.unsqueeze(0).expand(n_fake, -1)
        u_real = self.U_psi(x_bar, theta_real_batch)
        u_fake = self.U_psi(x_tilde, theta_fake_batch)

        return u_real, u_fake, x_bar, x_tilde

    def _minimax_solver(self, current_epsilon):
        
        consecutive_small_steps = 0
        
        prev_theta = self.theta.detach().clone()
        last_loss_theta = None

        for i in range(self.minimax_iters):
            remaining = self.n_sim - self.total_virtual_samples
            if remaining <= 0:
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
                    current_epsilon,
                    detach_theta=True,
                    n_fake=n_fake,
                )
                safe_u_fake_psi = torch.clamp(u_fake - 1.0, max=10.0)
                loss_psi = -(u_real.mean() - torch.exp(safe_u_fake_psi).mean())
                loss_psi.backward()
                self.opt_psi.step()

            
            theta_batch_size = base_batch + max(extra - n_inner, 0)
            self.opt_theta.zero_grad()
            u_real, u_fake, x_bar, x_tilde = self._get_u_scores(
                current_epsilon,
                detach_theta=False,
                n_fake=theta_batch_size,
            )
            
            abs_diff = torch.abs(x_bar.unsqueeze(1) - x_tilde.unsqueeze(0))
            min_diff_per_dim, _ = torch.min(abs_diff, dim=1)
            max_diff_per_sample, _ = torch.max(min_diff_per_dim, dim=1)
            loss_reg = torch.mean(max_diff_per_sample)
            ####
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

            
            if consecutive_small_steps >= self.early_stop_patience:
                print(f'    [Early stop] For {self.early_stop_patience} consecutive iterations, the maximum parameter change was < {self.early_stop_tol}')
                break

        if last_loss_theta is not None:
            self.loss_history_theta.append(last_loss_theta)

        return last_loss_theta

    def train(self):
        start_time = time.time()
        k = 0
        epsilon_k = self.epsilon_0

        while self.total_virtual_samples < self.n_sim:
            
            current_loss = self._minimax_solver(epsilon_k)
            if current_loss is None:
                break

            current_theta_val = self.theta.detach().cpu().numpy().copy()
            self.theta_history.append(current_theta_val)
            formatted_theta = "[" + ", ".join([f"{val:.4f}" for val in current_theta_val]) + "]"
            print(f'--> Current theta: {formatted_theta}, Obj: {current_loss:.6f}')
            epsilon_k = self.gamma * epsilon_k
            k += 1
        total_time = time.time() - start_time

        # Return training results
        return self.theta, self.loss_history_theta, self.theta_history, total_time,self.total_virtual_samples
