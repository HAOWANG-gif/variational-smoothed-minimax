import os
import time
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def find_best_initial_theta_via_nle_map(
        real_observations,
        generator_cls,
        prior,
        n_train=100000,
        n_candidates=20000,
        simulation_batch_size=1000,
        training_batch_size=1000,
        density_estimator="maf",
        model_path=None,
        retrain=False,
        candidate_batch_size=128,
        seed=None,
        map_seed=None,
        candidate_path=None,
        retrain_candidates=False,
        show_progress_bars=True,
):
    'Train or load an NLE surrogate and select the best MAP initialization from fixed prior candidates.'

    from sbi.inference import NLE, simulate_for_sbi
    from sbi.utils import BoxUniform

    start_time = time.time()

    # ==================================================
    
    # ==================================================
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ==================================================
    
    # ==================================================
    y_obs = real_observations.detach().float().to(device)

    if y_obs.ndim == 1:
        y_obs = y_obs.unsqueeze(1)

    n_data, x_dim = y_obs.shape

    # ==================================================
    
    # ==================================================
    
    # ==================================================
    try:
        prior.to(device)
    except AttributeError:
        pass
    
    theta_example = prior.sample((1,)).to(device).float()
    theta_dim = theta_example.shape[-1]
    # ==================================================
    
    
    
    # ==================================================
    def _safe_torch_load(path, map_location):
        try:
            return torch.load(
                path,
                map_location=map_location,
                weights_only=False
            )
        except TypeError:
            return torch.load(
                path,
                map_location=map_location
            )
    def simulator(theta):
        theta = theta.float().to(device)
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        xs = []
        for th in theta:
            gen = generator_cls(th, n_samples=1, seed=None)
            x = gen.generate_samples()
            if x.ndim == 1:
                x = x.unsqueeze(1)
            xs.append(x.reshape(-1))
        return torch.stack(xs, dim=0).float().to(device)

    # ==================================================
    
    # ==================================================
    if model_path is not None and os.path.exists(model_path) and not retrain:
        print(f'Loading existing NLE model: {model_path}')
        
        checkpoint = _safe_torch_load(model_path, map_location=device)
        likelihood_estimator = checkpoint["likelihood_estimator"].to(device)
        likelihood_estimator.eval()

    else:
        print('\n========== Training the NLE initialization model ==========')
        print(f"n_train = {n_train}")
        print(f"density_estimator = {density_estimator}")
        print(f'observation dimension x_dim = {x_dim}, parameter dimension theta_dim = {theta_dim}')

        inference = NLE(
            prior=prior,
            density_estimator=density_estimator,
            show_progress_bars=show_progress_bars,
            device=str(device)
        )

        theta_train, x_train = simulate_for_sbi(
            simulator=simulator,
            proposal=prior,
            num_simulations=n_train,
            simulation_batch_size=simulation_batch_size,
            show_progress_bar=show_progress_bars,
        )

        theta_train = theta_train.float().to(device)
        x_train = x_train.float().to(device)

        print(f"theta_train.shape = {theta_train.shape}")
        print(f"x_train.shape     = {x_train.shape}")

        likelihood_estimator = inference.append_simulations(
            theta_train,
            x_train
        ).train(
            training_batch_size=training_batch_size
        )

        likelihood_estimator = likelihood_estimator.to(device)
        likelihood_estimator.eval()

        if model_path is not None:
            model_dir = os.path.dirname(model_path)
            if model_dir != "":
                os.makedirs(model_dir, exist_ok=True)

            torch.save(
                {
                    "likelihood_estimator": likelihood_estimator,
                    "theta_dim": theta_dim,
                    "x_dim": x_dim,
                    "n_train": n_train,
                    "density_estimator": density_estimator,
                },
                model_path
            )
            print(f'NLE model saved to: {model_path}')

    
    likelihood_estimator.eval()

    # ==================================================
    
    # ==================================================
    @torch.no_grad()
    def score_theta_batch(theta_batch):
        'Score theta batch.'

        likelihood_estimator.eval()

        theta_batch = theta_batch.float().to(device)

        if theta_batch.ndim == 1:
            theta_batch = theta_batch.unsqueeze(0)

        B = theta_batch.shape[0]
        N = y_obs.shape[0]

        
        theta_rep = theta_batch[:, None, :].expand(
            B, N, theta_dim
        ).reshape(B * N, theta_dim)

        x_rep = y_obs[None, :, :].expand(
            B, N, x_dim
        ).reshape(B * N, x_dim)

        # --------------------------------------------------
        
        
        
        # --------------------------------------------------
        x_rep_for_logprob = x_rep.unsqueeze(0)

        log_like_each = likelihood_estimator.log_prob(
            x_rep_for_logprob,
            theta_rep
        )

        
        log_like_each = log_like_each.reshape(-1)

        
        log_like_sum = log_like_each.reshape(B, N).sum(dim=1)

        
        log_prior = prior.log_prob(theta_batch)

        return log_prior + log_like_sum

    # ==================================================
    
    # ==================================================
    print('\n========== Starting the NLE-MAP initial-value search ==========')
    print(f"n_candidates = {n_candidates}")
    print(f"n_data       = {n_data}")

    if map_seed is None:
        map_seed = seed + 99991 if seed is not None else 99991

    print(f"map_seed     = {map_seed}")

    if candidate_path is not None and os.path.exists(candidate_path) and not retrain_candidates:
        print(f'Loading existing MAP candidates: {candidate_path}')
        candidates = torch.load(candidate_path, map_location=device).float().to(device)

        if candidates.shape[0] != n_candidates:
            print(
                f'Warning: candidate_path contains {candidates.shape[0]}, which differs from current n_candidates={n_candidates} do not match.'
            )

    else:
        
        
        torch.manual_seed(map_seed)
        np.random.seed(map_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(map_seed)

        candidates = prior.sample((n_candidates,)).float().to(device)

        if candidate_path is not None:
            candidate_dir = os.path.dirname(candidate_path)
            if candidate_dir != "":
                os.makedirs(candidate_dir, exist_ok=True)

            torch.save(candidates.detach().cpu(), candidate_path)
            print(f'MAP candidate parameters saved to: {candidate_path}')

    # ==================================================
    
    # ==================================================
    best_score = -float("inf")
    best_theta = None

    for start in range(0, candidates.shape[0], candidate_batch_size):
        end = min(start + candidate_batch_size, candidates.shape[0])
        theta_batch = candidates[start:end]

        scores = score_theta_batch(theta_batch)

        batch_best_idx = torch.argmax(scores)
        batch_best_score = scores[batch_best_idx].item()

        if batch_best_score > best_score:
            best_score = batch_best_score
            best_theta = theta_batch[batch_best_idx].detach().clone()

    elapsed = time.time() - start_time


    print(f'Total elapsed time: {elapsed:.2f}s')

    return best_theta.detach().to(device)
