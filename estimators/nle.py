import os
import time
import torch
import numpy as np

from sbi.inference import NLE, simulate_for_sbi
from sbi.inference.posteriors.posterior_parameters import MCMCPosteriorParameters


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_torch_load(path, map_location):
    'Safe torch load.'
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _set_global_seed(seed=None):
    'Set global seed.'
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _move_posterior_to_device(posterior, target_device):
    'Move posterior to device.'
    if hasattr(posterior, "net"):
        posterior.net.to(target_device)

    if hasattr(posterior, "posterior_estimator"):
        posterior.posterior_estimator.to(target_device)

    if hasattr(posterior, "_device"):
        posterior._device = str(target_device)

    return posterior


class AmortizedNLEEstimator:
    'AmortizedNLEEstimator.'

    def __init__(
            self,
            method_name,
            prior,
            simulator,
            n_data,
            sample_generator_cls,
            train_config
    ):
        self.method_name = method_name
        self.prior = prior
        self.simulator = simulator
        self.n_data = n_data
        self.sample_generator_cls = sample_generator_cls
        self.config = train_config

        # ==================================================
        
        # ==================================================
        self.num_simulations = train_config.get("num_simulations", 100000)
        self.batch_size = train_config.get("batch_size", 1000)
        self.simulation_batch_size = train_config.get("simulation_batch_size", 1000)

        self.density_estimator = train_config.get("density_estimator", "maf")
        self.learning_rate = train_config.get("learning_rate", 5e-4)
        self.validation_fraction = train_config.get("validation_fraction", 0.1)
        self.stop_after_epochs = train_config.get("stop_after_epochs", 20)
        self.max_num_epochs = train_config.get("max_num_epochs", 2147483647)
        self.clip_max_norm = train_config.get("clip_max_norm", 5.0)

        self.show_progress_bars = train_config.get("show_progress_bars", False)
        self.show_train_summary = train_config.get("show_train_summary", False)

        # ==================================================
        
        # ==================================================
        self.mcmc_method = train_config.get("mcmc_method", "slice_np_vectorized")
        self.num_chains = train_config.get("num_chains", 10)
        self.thin = train_config.get("thin", 5)
        self.warmup_steps = train_config.get("warmup_steps", 200)
        self.init_strategy = train_config.get("init_strategy", "proposal")

        # ==================================================
        
        # ==================================================
        self.num_posterior_samples = train_config.get("num_posterior_samples", 5000)
        self.map_batch_size = train_config.get("map_batch_size", 2048)
        self.sample_seed = train_config.get("sample_seed", None)

        # ==================================================
        # Random seed
        # ==================================================
        self.seed = train_config.get("seed", None)

        # ==================================================
        
        
        
        
        # ==================================================
        self.virtual_samples_per_simulation = train_config.get(
            "virtual_samples_per_simulation",
            1
        )
        self.total_virtual_samples = 0

        # ==================================================
        
        
        
        # ==================================================
        self.model_dir = train_config.get("model_dir", "./saved_sbi_models")
        os.makedirs(self.model_dir, exist_ok=True)

        self.save_model = train_config.get("save_model", True)

        model_tag = train_config.get(
            "model_tag",
            f"{method_name}_NLE_{self.density_estimator}_sims_{self.num_simulations}"
        )

        self.model_path = train_config.get(
            "model_path",
            os.path.join(self.model_dir, f"{model_tag}.pt")
        )

        
        self.loss_history = []
        self.theta_history = []

    # ============================================================
    
    # ============================================================

    def train_nle(self):
        'Train nle.'

        print(f'    --> No usable {self.method_name} pretrained model; generating data and training NLE...')

        _set_global_seed(self.seed)

        inferer = NLE(
            prior=self.prior,
            density_estimator=self.density_estimator,
            device=str(device),
            show_progress_bars=self.show_progress_bars
        )

        # --------------------------------------------------
        
        # theta_train: [num_simulations, theta_dim]
        # x_train:     [num_simulations, x_dim]
        # --------------------------------------------------
        theta_train, x_train = simulate_for_sbi(
            simulator=self.simulator,
            proposal=self.prior,
            num_simulations=self.num_simulations,
            simulation_batch_size=self.simulation_batch_size,
            show_progress_bar=self.show_progress_bars
        )

        
        self.total_virtual_samples += (
            self.num_simulations
            * self.virtual_samples_per_simulation
        )

        theta_train = theta_train.float()
        x_train = x_train.float()

        print(f"        theta_train.shape = {theta_train.shape}")
        print(f"        x_train.shape     = {x_train.shape}")
        print(f'        cumulative simulated-sample count = {self.total_virtual_samples}')

        # --------------------------------------------------
        
        # --------------------------------------------------
        inferer.append_simulations(
            theta_train,
            x_train
        )

        trained_density_estimator = inferer.train(
            training_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            validation_fraction=self.validation_fraction,
            stop_after_epochs=self.stop_after_epochs,
            max_num_epochs=self.max_num_epochs,
            clip_max_norm=self.clip_max_norm,
            show_train_summary=self.show_train_summary
        )

        # --------------------------------------------------
        
        # --------------------------------------------------
        mcmc_parameters = MCMCPosteriorParameters(
            method=self.mcmc_method,
            num_chains=self.num_chains,
            thin=self.thin,
            warmup_steps=self.warmup_steps,
            init_strategy=self.init_strategy,
        )

        posterior = inferer.build_posterior(
            density_estimator=trained_density_estimator,
            posterior_parameters=mcmc_parameters
        )

        return posterior

    # ============================================================
    
    # ============================================================

    def _posterior_map(self, posterior, target_x):
        'Posterior map.'

        if self.sample_seed is not None:
            _set_global_seed(self.sample_seed)

        samples = posterior.sample(
            (self.num_posterior_samples,),
            x=target_x,
            show_progress_bars=False
        ).detach().to(device)

        score_list = []

        with torch.no_grad():
            for start in range(0, samples.shape[0], self.map_batch_size):
                end = min(start + self.map_batch_size, samples.shape[0])
                batch_samples = samples[start:end]

                try:
                    if hasattr(posterior, "potential"):
                        scores = posterior.potential(
                            batch_samples,
                            x=target_x
                        )
                    else:
                        scores = posterior.log_prob(
                            batch_samples,
                            x=target_x
                        )
                except Exception:
                    try:
                        scores = posterior.log_prob(
                            batch_samples,
                            x=target_x
                        )
                    except Exception:
                        posterior_x = posterior.set_default_x(target_x)
                        scores = posterior_x.log_prob(batch_samples)

                if scores.ndim > 1:
                    scores = scores.reshape(-1)

                score_list.append(scores.detach().cpu())

        scores_all = torch.cat(score_list, dim=0)

        best_idx = torch.argmax(scores_all).item()
        map_estimate = samples[best_idx].detach().to(device)

        return map_estimate

    # ============================================================
    
    # ============================================================

    def fit(self, y_obs):
        'Fit the estimator and return the standardized experiment outputs.'

        start_time = time.time()

        
        self.total_virtual_samples = 0

        y_obs = y_obs.detach().float().to(device)

        if y_obs.ndim == 1:
            y_obs = y_obs.unsqueeze(1)

        
        #     [n_trials, x_dim]
        target_x = y_obs.to(device)

        # ==================================================
        
        # ==================================================
        model_exists = os.path.exists(self.model_path)

        if model_exists:
            print(f'    --> Found a pretrained NLE model; loading {self.model_path}')

            posterior = _safe_torch_load(
                self.model_path,
                map_location=torch.device("cpu")
            )

            posterior = _move_posterior_to_device(posterior, device)

            train_time = 0.0
            self.total_virtual_samples = 0

        else:
            train_start = time.time()

            posterior = self.train_nle()

            train_time = time.time() - train_start

            print(f'    --> NLE training completed; elapsed: {train_time:.2f}s')
            print(f'    --> NLE training simulated-sample count: {self.total_virtual_samples}')

            # --------------------------------------------------
            
            # --------------------------------------------------
            if self.save_model:
                posterior_cpu = _move_posterior_to_device(posterior, "cpu")

                torch.save(
                    posterior_cpu,
                    self.model_path
                )

                print(f'    --> NLE model saved to: {self.model_path}')

                posterior = _move_posterior_to_device(posterior_cpu, device)

        # ==================================================
        
        # ==================================================
        eval_start = time.time()

        posterior = _move_posterior_to_device(posterior, device)

        map_estimate = self._posterior_map(
            posterior=posterior,
            target_x=target_x
        )
        total_time = train_time + (time.time() - eval_start)
        return (
            map_estimate,
            self.loss_history,
            self.theta_history,
            total_time,
            self.total_virtual_samples
        )
