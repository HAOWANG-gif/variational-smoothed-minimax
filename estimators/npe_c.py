import os
import time
import torch
import numpy as np

from sbi.inference import NPE_C
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_torch_load(path, map_location):
    'Safe torch load.'
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _set_global_seed(seed=None):
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


class AmortizedNPECSetEstimator:
    def __init__(self, method_name, prior, simulator, n_data, sample_generator_cls, train_config):
        self.method_name = method_name
        self.prior = prior
        self.simulator = simulator
        self.n_data = n_data  
        self.sample_generator_cls = sample_generator_cls
        self.config = train_config

        # -----------------------------
        
        # -----------------------------
        self.num_rounds = train_config.get("num_rounds", 5)
        self.num_simulations = train_config.get("num_simulations", 20000)
        self.batch_size = train_config.get("batch_size", 64)
        self.learning_rate = train_config.get("learning_rate", 5e-4)
        self.validation_fraction = train_config.get("validation_fraction", 0.1)
        self.stop_after_epochs = train_config.get("stop_after_epochs", 20)
        self.max_num_epochs = train_config.get("max_num_epochs", 2147483647)
        self.clip_max_norm = train_config.get("clip_max_norm", 5.0)

        # -----------------------------
        
        # -----------------------------
        self.num_atoms = train_config.get("num_atoms", 10)
        self.force_first_round_loss = train_config.get("force_first_round_loss", False)
        self.discard_prior_samples = train_config.get("discard_prior_samples", False)
        self.use_combined_loss = train_config.get("use_combined_loss", False)
        self.retrain_from_scratch = train_config.get("retrain_from_scratch", False)

        
        self.exclude_invalid_x = train_config.get("exclude_invalid_x", None)

        # -----------------------------
        
        # -----------------------------
        self.density_model = train_config.get("density_model", "maf")
        self.z_score_x = train_config.get("z_score_x", "independent")
        self.density_hidden_features = train_config.get("density_hidden_features", 50)
        self.density_num_transforms = train_config.get("density_num_transforms", 5)
        self.latent_dim = train_config.get("latent_dim", 16)
        self.embedding_dim = train_config.get("embedding_dim", self.latent_dim)

        self.trial_num_hiddens = train_config.get("trial_num_hiddens", 64)
        self.trial_num_layers = train_config.get("trial_num_layers", 2)

        self.perm_num_hiddens = train_config.get("perm_num_hiddens", 32)
        self.perm_num_layers = train_config.get("perm_num_layers", 1)

        # -----------------------------
        
        # -----------------------------
        self.num_posterior_samples = train_config.get("num_posterior_samples", 5000)
        self.map_batch_size = train_config.get("map_batch_size", 2048)
        self.sample_seed = train_config.get("sample_seed", None)
        self.posterior_sampling_method = train_config.get(
            "posterior_sampling_method", "mcmc"
        )
        self.importance_oversampling_factor = int(
            train_config.get("importance_oversampling_factor", 4)
        )

        self.show_progress_bars = train_config.get("show_progress_bars", False)
        self.show_train_summary = train_config.get("show_train_summary", False)

        # -----------------------------
        # Random seed
        # -----------------------------
        self.seed = train_config.get("seed", None)
        self.total_virtual_samples = 0
        self.model_dir = train_config.get("model_dir", "./saved_sbi_models")
        os.makedirs(self.model_dir, exist_ok=True)

        self.reuse_multiround_model = train_config.get("reuse_multiround_model", False)
        self.save_model = train_config.get("save_model", True)

        model_tag = train_config.get(
            "model_tag",
            f"{method_name}_ndata_{n_data}_rounds_{self.num_rounds}_{self.density_model}_sims_{self.num_simulations}"
        )

        self.model_path = train_config.get(
            "model_path",
            os.path.join(self.model_dir, f"{model_tag}.pt")
        )

    # ============================================================
    
    # ============================================================

    def _generate_fixed_set_simulations(self, theta_train, obs_dim):
        'Generate fixed set simulations.'

        theta_train = theta_train.detach().float().to(device)
        num_simulations = theta_train.shape[0]

        
        x_train = torch.zeros(
            (num_simulations, self.n_data, obs_dim),
            dtype=torch.float32,
            device="cpu"
        )

        for i in range(num_simulations):
            current_theta = theta_train[i]

            gen = self.sample_generator_cls(
                current_theta,
                n_samples=self.n_data
            )

            xi = gen.generate_samples().detach().float().cpu()

            if xi.ndim == 1:
                xi = xi.unsqueeze(1)

            x_train[i, :, :] = xi

        
        self.total_virtual_samples += num_simulations * self.n_data

        return x_train

    # ============================================================
    
    # ============================================================

    def _build_density_estimator(self, obs_dim):
        'Build density estimator.'

        single_trial_net = FCEmbedding(
            input_dim=obs_dim,
            num_hiddens=self.trial_num_hiddens,
            num_layers=self.trial_num_layers,
            output_dim=self.latent_dim
        )

        embedding_net = PermutationInvariantEmbedding(
            single_trial_net,
            trial_net_output_dim=self.latent_dim,
            num_layers=self.perm_num_layers,
            num_hiddens=self.perm_num_hiddens,
            output_dim=self.embedding_dim
        )

        density_estimator = posterior_nn(
            model=self.density_model,
            embedding_net=embedding_net,
            z_score_x=self.z_score_x,
            hidden_features=self.density_hidden_features,
            num_transforms=self.density_num_transforms,
        )

        return density_estimator

    # ============================================================
    
    # ============================================================

    def _sample_from_posterior(self, posterior, num_samples, target_x):
        'Sample from posterior.'
        kwargs = {
            "x": target_x,
            "show_progress_bars": False,
        }
        if self.posterior_sampling_method == "importance":
            kwargs["oversampling_factor"] = self.importance_oversampling_factor

        return posterior.sample((num_samples,), **kwargs)

    def _build_posterior(self, inferer, trained_density_estimator):
        kwargs = {"sample_with": self.posterior_sampling_method}
        if self.posterior_sampling_method == "importance":
            kwargs["importance_sampling_parameters"] = {
                "method": "sir",
                "oversampling_factor": self.importance_oversampling_factor,
            }

        return inferer.build_posterior(trained_density_estimator, **kwargs)

    def train_npec(self, y_obs):
        'Train npec.'

        print(f'    --> No usable {self.method_name} pretrained model; starting NPE-C training...')

        _set_global_seed(self.seed)

        y_obs = y_obs.detach().float().to(device)

        if y_obs.ndim == 1:
            y_obs = y_obs.unsqueeze(1)

        obs_dim = y_obs.shape[-1]
        target_x = y_obs.unsqueeze(0).to(device)

        density_estimator = self._build_density_estimator(obs_dim)

        inferer = NPE_C(
            prior=self.prior,
            density_estimator=density_estimator,
            device=str(device),
            show_progress_bars=self.show_progress_bars
        )

        proposal = self.prior
        posterior = None
        trained_density_estimator = None

        for round_idx in range(self.num_rounds):
            print(f"    --> NPE-C Round {round_idx + 1}/{self.num_rounds}")

            round_seed = None
            if self.seed is not None:
                round_seed = self.seed + 10000 * round_idx

            _set_global_seed(round_seed)

            # -----------------------------
            
            # -----------------------------
            if round_idx == 0:
                theta_train = proposal.sample((self.num_simulations,))
            else:
                theta_train = self._sample_from_posterior(
                    posterior=proposal,
                    num_samples=self.num_simulations,
                    target_x=target_x,
                )

            theta_train = theta_train.detach().float().to(device)

            # -----------------------------
            
            # -----------------------------
            x_train = self._generate_fixed_set_simulations(
                theta_train=theta_train,
                obs_dim=obs_dim
            )
            theta_train_cpu = theta_train.cpu()

            print(f"        theta_train.shape = {theta_train_cpu.shape}")
            print(f"        x_train.shape     = {x_train.shape}")
            print(f'        cumulative simulated-sample count = {self.total_virtual_samples}')

            
            
            proposal_for_append = None if round_idx == 0 else proposal

            inferer.append_simulations(
                theta_train_cpu,
                x_train,
                proposal=proposal_for_append,
                exclude_invalid_x=self.exclude_invalid_x
            )

            trained_density_estimator = inferer.train(
                num_atoms=self.num_atoms,
                training_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                validation_fraction=self.validation_fraction,
                stop_after_epochs=self.stop_after_epochs,
                max_num_epochs=self.max_num_epochs,
                clip_max_norm=self.clip_max_norm,
                force_first_round_loss=self.force_first_round_loss,
                discard_prior_samples=self.discard_prior_samples,
                use_combined_loss=self.use_combined_loss,
                retrain_from_scratch=self.retrain_from_scratch,
                show_train_summary=self.show_train_summary
            )

            posterior = self._build_posterior(
                inferer=inferer,
                trained_density_estimator=trained_density_estimator,
            )
            proposal = posterior.set_default_x(target_x)

        return posterior

    # ============================================================
    
    # ============================================================

    def _posterior_map(self, posterior, target_x):
        'Posterior map.'

        if self.sample_seed is not None:
            _set_global_seed(self.sample_seed)

        samples = self._sample_from_posterior(
            posterior=posterior,
            num_samples=self.num_posterior_samples,
            target_x=target_x,
        ).detach().to(device)

        log_probs_list = []

        with torch.no_grad():
            for start in range(0, samples.shape[0], self.map_batch_size):
                end = min(start + self.map_batch_size, samples.shape[0])
                batch_samples = samples[start:end]

                try:
                    log_probs = posterior.log_prob(
                        batch_samples,
                        x=target_x
                    )
                except Exception:
                    
                    posterior_x = posterior.set_default_x(target_x)
                    log_probs = posterior_x.log_prob(batch_samples)

                if log_probs.ndim > 1:
                    log_probs = log_probs.reshape(-1)

                log_probs_list.append(log_probs.detach().cpu())

        log_probs_all = torch.cat(log_probs_list, dim=0)

        best_idx = torch.argmax(log_probs_all).item()
        map_estimate = samples[best_idx].detach().to(device)

        return map_estimate

    # ============================================================
    
    # ============================================================

    def fit(self, y_obs):
        start_time = time.time()

        
        self.total_virtual_samples = 0

        y_obs = y_obs.detach().float().to(device)

        if y_obs.ndim == 1:
            y_obs = y_obs.unsqueeze(1)

        target_x = y_obs.unsqueeze(0).to(device)

        # --------------------------------------------------------
        
        # --------------------------------------------------------
        can_reuse_model = (
            self.num_rounds == 1
            or self.reuse_multiround_model
        )

        model_exists = os.path.exists(self.model_path)

        if model_exists and can_reuse_model:
            print(f'    --> Found a pretrained model; loading {self.model_path}')

            posterior = _safe_torch_load(
                self.model_path,
                map_location=torch.device("cpu")
            )

            posterior = _move_posterior_to_device(posterior, device)

            train_time = 0.0

            
            self.total_virtual_samples = 0

        else:
            if model_exists and not can_reuse_model:
                print(
                    f'    --> Found model file {self.model_path}, but num_rounds={self.num_rounds} is a sequential NPE-C model conditioned on a specific y_obs and is not reused across observations by default.'
                )

            train_start = time.time()

            posterior = self.train_npec(y_obs)

            train_time = time.time() - train_start

            print(f'    --> NPE-C training completed; elapsed: {train_time:.2f}s')
            print(f'    --> NPE-C training simulated-sample count: {self.total_virtual_samples}')

            # ----------------------------------------------------
            
            # ----------------------------------------------------
            if self.save_model and can_reuse_model:
                posterior_cpu = _move_posterior_to_device(posterior, "cpu")

                torch.save(posterior_cpu, self.model_path)

                print(f'    --> Model saved to: {self.model_path}')

                posterior = _move_posterior_to_device(posterior_cpu, device)

        # --------------------------------------------------------
        
        # --------------------------------------------------------
        eval_start = time.time()

        posterior = _move_posterior_to_device(posterior, device)

        map_estimate = self._posterior_map(
            posterior=posterior,
            target_x=target_x
        )

        total_time = train_time + (time.time() - eval_start)

        return (
            map_estimate,
            [],
            [],
            total_time,
            self.total_virtual_samples
        )
