import os
import sys
def _early_set_cuda_visible_devices_from_argv():
    if "--gpu-id" in sys.argv:
        idx = sys.argv.index("--gpu-id")
        if idx + 1 < len(sys.argv):
            gpu_id = sys.argv[idx + 1]
            if gpu_id.lower() not in {"auto", "none", "cpu"}:
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


_early_set_cuda_visible_devices_from_argv()
import csv
import json
import time
import argparse
import subprocess
import traceback
from datetime import datetime
from pprint import pformat

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from geomloss import SamplesLoss


# ============================================================
# 2. Paths and device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################################
import os
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
CURRENT_DIR = CURRENT_FILE.parent

USER_HOME = CURRENT_DIR.parent
project_root = str(USER_HOME)

############################################################
if project_root not in sys.path:
    sys.path.append(project_root)
from estimators.mwe import MWE
from estimators.annd import AdversarialEstimator
from estimators.vsm import AnnealedMinimaxTrainer
from estimators.adaimle import AdaptiveIMLEParameterEstimator
from estimators.nle import AmortizedNLEEstimator
from estimators.nbe import AmortizedNBEEstimator, apply_nbe_training_cost_once
from estimators.npe_c import AmortizedNPECSetEstimator

from models.glm import *

from utils.initialization import find_best_initial_theta_via_nle_map
from utils.results import save_array_to_csv, ExperimentUtils


# ============================================================
# 4. Critic network used by VSM
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.LeakyReLU(negative_slope=0.2),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
        )

    def forward(self, x):
        return self.block(x) + x


class TNet(nn.Module):
    def __init__(
            self,
            x_dim=x_dim,
            theta_dim=theta_dim,
            hidden=64,
            alpha=1.0,
            num_blocks=1,
    ):
        super().__init__()
        self.alpha = alpha
        self.x_dim = x_dim
        self.theta_dim = theta_dim

        self.hyper_theta = nn.Sequential(
            spectral_norm(nn.Linear(theta_dim, hidden)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden, hidden)),
        )

        self.encoder_x = nn.Sequential(
            spectral_norm(nn.Linear(x_dim, hidden)),
            nn.LeakyReLU(0.2),
        )

        self.res_layers = nn.Sequential(
            *[ResBlock(hidden) for _ in range(num_blocks)]
        )

        self.output_layer = nn.Sequential(
            spectral_norm(nn.Linear(hidden, 1))
        )

    def forward(self, X, theta):
        x_feat = self.encoder_x(X)
        theta_feat = self.hyper_theta(theta)
        fused_feat = x_feat + theta_feat
        out = self.res_layers(fused_feat)
        return self.alpha * self.output_layer(out)


# ============================================================
# 5. Saving and serialization utilities
# ============================================================

NEED_INIT_METHODS = {"Ours", "WME_p1", "WME_p2", "ANND", "AdaIMLE"}


def make_theta_header(theta_dim):
    return [f"theta_{i}" for i in range(theta_dim)]


def make_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    if callable(obj):
        return getattr(obj, "__name__", str(obj))

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    return str(obj)


def save_experiment_config(config, root_dir):
    os.makedirs(root_dir, exist_ok=True)

    config_json_path = os.path.join(root_dir, "experiment_config.json")
    config_txt_path = os.path.join(root_dir, "experiment_config_readable.txt")

    jsonable_config = make_jsonable(config)

    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(jsonable_config, f, ensure_ascii=False, indent=4)

    with open(config_txt_path, "w", encoding="utf-8") as f:
        f.write(pformat(jsonable_config, width=120, sort_dicts=False))

    print(f'[OK] Experiment configuration saved: {config_json_path}')
    print(f'[OK] Human-readable configuration saved: {config_txt_path}')


def save_history(history, file_path):
    if history is None:
        return

    if isinstance(history, list) and len(history) == 0:
        return

    if isinstance(history, list) and all(isinstance(x, dict) for x in history):
        fieldnames = sorted(set().union(*(row.keys() for row in history)))

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

        return

    save_array_to_csv(history, file_path)


def _jsonify_for_csv(v):
    v = make_jsonable(v)

    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)

    return v


def save_records_to_csv(records, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if records is None or len(records) == 0:
        with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
            f.write("")
        return

    fieldnames = sorted(set().union(*(row.keys() for row in records)))
    preferred_order = [
        "n_data",
        "run_id",
        "method",
        "baseline_method",
        "compare_method",
        "time_used",
        "n_sim",
        "param_rmse",
        "param_mse",
        "param_mae",
        "obs_sinkhorn_norm",
        "sinkhorn_norm",
        "c2st",
    ]
    ordered = []
    for k in preferred_order:
        if k in fieldnames:
            ordered.append(k)

    for k in fieldnames:
        if k not in ordered:
            ordered.append(k)

    with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()

        for row in records:
            writer.writerow({k: _jsonify_for_csv(row.get(k, "")) for k in ordered})


def save_shard_summaries(
        shard_dir,
        params_summary,
        metrics_summary,
        mle_comparison_summary,
):
    os.makedirs(shard_dir, exist_ok=True)

    save_records_to_csv(
        params_summary,
        os.path.join(shard_dir, "params_summary.csv")
    )

    save_records_to_csv(
        metrics_summary,
        os.path.join(shard_dir, "metrics_summary.csv")
    )

    save_records_to_csv(
        mle_comparison_summary,
        os.path.join(shard_dir, "mle_comparison_summary.csv")
    )


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


# ============================================================
# 6. Configuration
# ============================================================

def build_experiment_config():
    demo_dir_name = CURRENT_DIR.name
    if demo_dir_name.startswith("demo_"):
        example_name = demo_dir_name.split("demo_", 1)[1]
    else:
        example_name = demo_dir_name
    global_seed = 42
    
    n_sim = 500000
    n_init_sim = 10000
    python_file_name = CURRENT_FILE.stem
    root_experiment_dir = os.path.join(
        str(USER_HOME),
        demo_dir_name,
        f'{python_file_name}_results'
    )
    n_runs=100
    n_data_list=[100]
    experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join(
        root_experiment_dir,
        f"Exp_{example_name}_{experiment_time}"
    )

    pretrained_root = root_experiment_dir

    methods_to_run = [
        "Ours",
        "WME_p2",
        "AdaIMLE",
        "ANND",
        "NBE",
        "MLE"
    ]

    train_configs = {
        "Ours": {
            "gamma": 0.5,
            "minimax_iters": 600,
            "lr_theta": 1e-3,
            "lr_psi": 5e-4,
            "m1": 10,
            "m2": 10,
            "lambda_reg": 0.0,
            "early_stop_patience": 50,
            "early_stop_tol": 1e-5,
            "yasuo": 1.0
        },
        "WME": {
            "sim_batch_size": 100,
            "k_mc": 5,
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "seed": global_seed + 3000,
        },
        "ANND": {
            "g": 32,
            "num_hidden": 32,
            "reg_l2": 0.00,
            "m2": 1000,
            "S": 1,
            "xatol": 1e-5,
            "fatol": 1e-5,
            "disc_optimizer": "adam",
            "disc_lr": 0.001,
            "disc_train_iter": 100,
            "bounds_lb": [-3.0] * theta_dim,
            "bounds_ub": [3.0] * theta_dim,
        },

        "AdaIMLE": {
            "latent_dim": z_dim,
            "pool_factor": 100.0,
            "delta": 0.95,
            "initial_threshold": None,
            "threshold_update_rule": "shrink_current",
            "change_coef": 0.05,
            "obs_batch_size": None,
            "learning_rate": 1e-3,
            "candidate_batch_size": 1024,
            "refresh_candidate_pool_every": 15,
            "force_update_all_every": 15,
            "latent_search_standardization": "separate",
            "theta_loss_standardize": True,
            "current_distance_standardize": True,
            "lambda_std": 0.0,
            "std_target": "global",
            "grad_clip": 5.0,
            "verbose": False,
            "print_every": 500,
        },
        "NLE": {
            "num_simulations": 500000,
            "batch_size": 1000,
            "simulation_batch_size": 1000,
            "density_estimator": "maf",
            "learning_rate": 5e-4,
            "validation_fraction": 0.1,
            "stop_after_epochs": 20,
            "clip_max_norm": 5.0,

            "mcmc_method": "slice_np_vectorized",
            "num_chains": 10,
            "thin": 5,
            "warmup_steps": 200,
            "init_strategy": "proposal",

            "num_posterior_samples": 5000,
            "map_batch_size": 2048,

            "model_dir": os.path.join(pretrained_root, "pretrained_nle_models"),
            "save_model": True,

            "seed": global_seed + 4000,
            "sample_seed": global_seed + 5000,

            "show_progress_bars": False,
            "show_train_summary": False,
        },
        "NBE": {
            "model_dir": os.path.join(pretrained_root, "pretrained_nbe_models"),
            "m_values": [1, 10,100],
            "m_cutoffs": [5,50],
            "x_dim": x_dim,
            "theta_dim": theta_dim,
            "K_train": 2250,
            "K_val": 500,
            "batch_size": 128,
            "epochs": 500,
            "target_transform": target_transform,
        },
        "NPE_C": {
            "model_dir": os.path.join(pretrained_root, "pretrained_npec_models"),
            "num_rounds": 2,
            "num_simulations": 2500,
            "batch_size": 1000,
            "density_model": "maf",
            "latent_dim": 16,
            "embedding_dim": 16,
            "trial_num_hiddens": 64,
            "trial_num_layers": 2,
            "perm_num_hiddens": 32,
            "perm_num_layers": 1,
            "z_score_x": "independent",
            "learning_rate": 5e-4,
            "validation_fraction": 0.1,
            "stop_after_epochs": 20,
            "clip_max_norm": 5.0,
            "num_atoms": 10,
            "max_num_epochs":10000,
            "use_combined_loss": True,
            "num_posterior_samples": 5000,
            "reuse_multiround_model": False,
            "save_model": False,
            "seed": global_seed + 6000,
            "sample_seed": global_seed + 7000,
            "show_progress_bars": False,
            "show_train_summary": False,
        },

        "MLE": {
            "enabled": True,
        },
    }

    init_dir = os.path.join(pretrained_root, "pretrained_nle_init")

    init_config = {
        "n_train": n_init_sim,
        "n_candidates": 50000,
        "simulation_batch_size": 64,
        "training_batch_size": 64,
        "density_estimator": "mdn",
        "model_path": os.path.join(init_dir, f"{example_name.lower()}_nle_likelihood.pt"),
        "candidate_path": os.path.join(init_dir, f"{example_name.lower()}_nle_candidates.pt"),
        "retrain": False,
        "retrain_candidates": False,
        "candidate_batch_size": 64,
        "show_progress_bars": True,
    }
    for method_key in ("Ours", "WME", "ANND", "AdaIMLE"):
        train_configs[method_key]["n_sim"] = n_sim
        train_configs[method_key]["n_init_sim"] = n_init_sim

    config = {
        "experiment": {
            "example_name": example_name,
            "global_seed": global_seed,
            "run_seed_base": 10000000,
            "manual_seed_offset": 0,
            "n_runs": n_runs,
            "n_data_list": n_data_list,
            "methods_to_run": methods_to_run,
            "device": str(device),
            "experiment_time": experiment_time,
        },
        "dimensions": {
            "x_dim": x_dim,
            "theta_dim": theta_dim,
            "theta_header": make_theta_header(theta_dim),
        },
        "paths": {
            "user_home": USER_HOME,
            "project_root": project_root,
            "root_experiment_dir": root_experiment_dir,
            "pretrained_root": pretrained_root,
            "root_dir": root_dir,
        },
        "methods": train_configs,
        "init": init_config,
        "parallel": {
            "shard_id": 0,
            "num_shards": 1,
            "skip_finished": True,
            "prewarm_shared_assets": True,
        },
    }

    return config


def apply_cli_overrides(config, args):
    if args.exp_tag:
        config["experiment"]["experiment_time"] = args.exp_tag
        config["paths"]["root_dir"] = os.path.join(
            config["paths"]["root_experiment_dir"],
            f"Exp_{config['experiment']['example_name']}_{args.exp_tag}"
        )

    if args.n_runs is not None:
        config["experiment"]["n_runs"] = int(args.n_runs)

    if args.n_data_list:
        config["experiment"]["n_data_list"] = [
            int(x.strip()) for x in args.n_data_list.split(",") if x.strip()
        ]

    if args.methods:
        config["experiment"]["methods_to_run"] = [
            x.strip() for x in args.methods.split(",") if x.strip()
        ]

    if args.manual_seed_offset is not None:
        config["experiment"]["manual_seed_offset"] = int(args.manual_seed_offset)

    config["parallel"]["shard_id"] = int(args.shard_id)
    config["parallel"]["num_shards"] = int(args.num_shards)
    config["parallel"]["skip_finished"] = not args.no_skip_finished
    config["parallel"]["prewarm_shared_assets"] = not args.no_prewarm

    config["experiment"]["device"] = str(device)

    return config


# ============================================================
# 7. Single-method execution and evaluation
# ============================================================

def normalize_estimator_output(result, estimator=None):
    'Normalize estimator output.'
    if not isinstance(result, tuple):
        raise ValueError('estimator.train()/fit() must return a tuple.')
    if len(result) == 5:
        return result
    if len(result) == 4:
        final_theta, loss_history, theta_history, time_used = result
        n_sim = getattr(estimator, "total_virtual_samples", 0)
        return final_theta, loss_history, theta_history, time_used, n_sim

    raise ValueError(f'Unexpected estimator return length: {len(result)}')


def run_estimation_method(method_name, y_obs, init_theta, config, n_data):
    print(f'\n--- Starting {method_name} method estimation ---')

    train_configs = config["methods"]
    x_dim_cfg = config["dimensions"]["x_dim"]
    theta_dim_cfg = config["dimensions"]["theta_dim"]

    if method_name == "Ours":
        if init_theta is None:
            raise ValueError('VSM requires init_theta, but no initial value was generated.')

        T_net = TNet(
            x_dim=x_dim_cfg,
            theta_dim=theta_dim_cfg,
        ).to(device)

        trainer = AnnealedMinimaxTrainer(
            y_obs=y_obs,
            init_theta=init_theta.clone().detach().to(device).requires_grad_(True),
            sample_generator_cls=SampleGenerator,
            U_net=T_net,
            train_config=train_configs["Ours"],
        )

        return normalize_estimator_output(trainer.train(), trainer)

    if method_name == "WME_p1":
        if init_theta is None:
            raise ValueError('WME_p1 requires init_theta, but no initial value was generated.')

        loss_fn = SamplesLoss(loss="sinkhorn", p=1, blur=1e-5, debias=True)

        estimator = MWE(
            y_obs=y_obs,
            theta_init=init_theta.clone().detach().to(device).requires_grad_(True),
            sample_generator_cls=SampleGenerator,
            loss_func=loss_fn,
            train_params=train_configs["WME"],
        )

        return normalize_estimator_output(estimator.train(), estimator)

    if method_name == "WME_p2":
        if init_theta is None:
            raise ValueError('WME_p2 requires init_theta, but no initial value was generated.')

        loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=1e-5, debias=True)

        estimator = MWE(
            y_obs=y_obs,
            theta_init=init_theta.clone().detach().to(device).requires_grad_(True),
            sample_generator_cls=SampleGenerator,
            loss_func=loss_fn,
            train_params=train_configs["WME"],
        )

        return normalize_estimator_output(estimator.train(), estimator)

    if method_name == "ANND":
        if init_theta is None:
            raise ValueError('ANND requires init_theta, but no initial value was generated.')

        estimator = AdversarialEstimator(
            y_obs=y_obs,
            init_theta=init_theta,
            sample_generator_cls=SampleGenerator,
            train_config=train_configs["ANND"],
        )

        return normalize_estimator_output(estimator.train(), estimator)

    if method_name == "AdaIMLE":
        if init_theta is None:
            raise ValueError('AdaIMLE requires init_theta, but no initial value was generated.')

        estimator = AdaptiveIMLEParameterEstimator(
            y_obs=y_obs,
            theta_init=init_theta.clone().detach().to(device),
            simulator_fn=adaimle_simulator_wrapper,
            config=train_configs["AdaIMLE"],
        )

        return normalize_estimator_output(estimator.train(), estimator)

    if method_name == "NBE":
        estimator = AmortizedNBEEstimator(
            method_name=method_name,
            n_data=n_data,
            sample_generator_cls=SampleGenerator,
            train_config=train_configs["NBE"],
        )

        return normalize_estimator_output(estimator.fit(y_obs), estimator)

    if method_name == "NLE":
        estimator = AmortizedNLEEstimator(
            method_name="NLE",
            prior=sbi_prior,
            simulator=sbi_simulator,
            n_data=n_data,
            sample_generator_cls=SampleGenerator,
            train_config=train_configs["NLE"],
        )

        return normalize_estimator_output(estimator.fit(y_obs), estimator)

    if method_name == "NPE_C":
        estimator = AmortizedNPECSetEstimator(
            method_name=method_name,
            prior=sbi_prior,
            simulator=sbi_simulator,
            n_data=n_data,
            sample_generator_cls=SampleGenerator,
            train_config=train_configs["NPE_C"],
        )

        return normalize_estimator_output(estimator.fit(y_obs), estimator)

    if method_name == "MLE":
        start_time = time.time()

        final_theta = compute_mle(y_obs).detach().to(device)

        loss_history = []
        theta_history = []
        time_used = time.time() - start_time
        n_sim = 0

        print(f'Final MLE parameter estimate: {final_theta.detach().cpu().numpy()}')

        return final_theta, loss_history, theta_history, time_used, n_sim

    raise ValueError(f'Unknown estimation method: {method_name}')


def evaluate_and_save_method(
        method_name,
        true_theta,
        final_theta,
        y_obs,
        time_used,
        n_sim,
        loss_history,
        theta_history,
        run_dir,
        seed,
        config,
):
    print(f'Computing {method_name} evaluation metrics...')

    method_dir = os.path.join(run_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    theta_header = config["dimensions"]["theta_header"]

    save_array_to_csv(
        final_theta,
        os.path.join(method_dir, "final_theta.csv"),
        header=theta_header
    )

    save_array_to_csv(
        np.array([time_used]),
        os.path.join(method_dir, "time_used.csv"),
        header=["time_seconds"]
    )

    save_array_to_csv(
        np.array([n_sim]),
        os.path.join(method_dir, "n_sim.csv"),
        header=["virtual_samples"]
    )

    save_history(loss_history, os.path.join(method_dir, "loss_history.csv"))
    save_history(theta_history, os.path.join(method_dir, "theta_history.csv"))

    metrics = compute_estimation_metrics(
        sample_generator_cls=SampleGenerator,
        true_theta=true_theta,
        estimate_theta=final_theta,
        y_obs=y_obs,
        device=device,
        seed=seed,
    )

    ExperimentUtils.save_dict_to_csv_fixed(
        metrics,
        os.path.join(method_dir, "evaluation_metrics.csv")
    )

    print(
        f"{method_name} completed | elapsed: {time_used:.2f}s | n_sim: {n_sim} | obs_sinkhorn_norm: {metrics.get('obs_sinkhorn_norm', 0):.4f}"
    )

    return metrics


# ============================================================
# 8. Initialization
# ============================================================

def maybe_get_initial_theta(config, y_obs, seed):
    methods_to_run = config["experiment"]["methods_to_run"]

    if not any(method in NEED_INIT_METHODS for method in methods_to_run):
        print('No selected method requires init_theta; skipping NLE-MAP initialization.')
        return None

    init_cfg = config["init"]

    init_theta = find_best_initial_theta_via_nle_map(
        real_observations=y_obs,
        generator_cls=SampleGenerator,
        prior=prior,
        n_train=init_cfg["n_train"],
        n_candidates=init_cfg["n_candidates"],
        simulation_batch_size=init_cfg["simulation_batch_size"],
        training_batch_size=init_cfg["training_batch_size"],
        density_estimator=init_cfg["density_estimator"],
        model_path=init_cfg["model_path"],
        retrain=init_cfg["retrain"],
        candidate_batch_size=init_cfg["candidate_batch_size"],
        seed=seed + 42,
        map_seed=seed + 777,
        candidate_path=init_cfg["candidate_path"],
        retrain_candidates=init_cfg["retrain_candidates"],
        show_progress_bars=init_cfg["show_progress_bars"],
    )

    return init_theta.detach().to(device).requires_grad_(True)


# ============================================================
# 9. Multi-device task partitioning
# ============================================================

def build_run_tasks(config):
    n_runs = config["experiment"]["n_runs"]
    n_data_list = config["experiment"]["n_data_list"]
    run_seed_base = config["experiment"]["run_seed_base"]
    manual_seed_offset = config["experiment"].get("manual_seed_offset", 0)

    tasks = []
    task_index = 0

    for n_data in n_data_list:
        for run_id in range(n_runs):
            current_run = run_id + 1
            seed = run_seed_base + run_id + manual_seed_offset

            tasks.append({
                "task_index": task_index,
                "n_data": int(n_data),
                "run_id": int(run_id),
                "current_run": int(current_run),
                "seed": int(seed),
            })

            task_index += 1

    return tasks


def get_shard_tasks(config):
    shard_id = config["parallel"]["shard_id"]
    num_shards = config["parallel"]["num_shards"]

    all_tasks = build_run_tasks(config)

    shard_tasks = [
        task for task in all_tasks
        if task["task_index"] % num_shards == shard_id
    ]

    return all_tasks, shard_tasks


# ============================================================
# 10. Prewarm shared resources to avoid concurrent model writes
# ============================================================

def shared_assets_flag_path(config):
    return os.path.join(config["paths"]["root_dir"], "_SHARED_ASSETS_READY.txt")


def shared_assets_fail_path(config):
    return os.path.join(config["paths"]["root_dir"], "_SHARED_ASSETS_FAILED.txt")


def shared_assets_need_prewarm(config):
    methods = config["experiment"]["methods_to_run"]

    # NLE-MAP initialization resources
    if any(method in NEED_INIT_METHODS for method in methods):
        init_cfg = config["init"]

        model_missing = not os.path.exists(init_cfg["model_path"])
        cand_missing = not os.path.exists(init_cfg["candidate_path"])

        if init_cfg.get("retrain", False):
            return True

        if init_cfg.get("retrain_candidates", False):
            return True

        if model_missing or cand_missing:
            return True

    # Pretrained NBE model
    if "NBE" in methods:
        tmp_estimator = AmortizedNBEEstimator(
            method_name="NBE",
            n_data=max(config["experiment"]["n_data_list"]),
            sample_generator_cls=SampleGenerator,
            train_config=config["methods"]["NBE"],
        )

        if not os.path.exists(tmp_estimator.model_path):
            return True

    # Pretrained NLE model
    if "NLE" in methods:
        tmp_estimator = AmortizedNLEEstimator(
            method_name="NLE",
            prior=sbi_prior,
            simulator=sbi_simulator,
            n_data=config["experiment"]["n_data_list"][0],
            sample_generator_cls=SampleGenerator,
            train_config=config["methods"]["NLE"],
        )

        if not os.path.exists(tmp_estimator.model_path):
            return True
    return False


def make_dummy_observation(config):
    seed = config["experiment"]["global_seed"] + 123456
    n_data = max(config["experiment"]["n_data_list"])

    ExperimentUtils.set_global_seed(seed)

    theta = (
        SampleGenerator
        .sample_prior(n_samples=1, seed=seed)
        .squeeze(0)
        .to(device)
    )

    y_obs = (
        SampleGenerator(theta, n_samples=n_data, seed=seed)
        .generate_samples()
        .detach()
        .to(device)
    )

    return y_obs


def prepare_shared_assets(config):

    root_dir = config["paths"]["root_dir"]
    os.makedirs(root_dir, exist_ok=True)

    flag_path = shared_assets_flag_path(config)
    fail_path = shared_assets_fail_path(config)

    if not config["parallel"].get("prewarm_shared_assets", True):
        atomic_write_text(flag_path, "skip prewarm\n")
        return

    if os.path.exists(flag_path):
        print('Shared pretrained resources are already ready; skipping prewarm.')
        return

    if not shared_assets_need_prewarm(config):
        print('Shared pretrained resources already exist; prewarming is unnecessary.')
        atomic_write_text(flag_path, "already ready\n")
        return

    print('\n========== shard 0: preparing shared pretrained resources ==========')

    try:
        y_dummy = make_dummy_observation(config)

        methods = config["experiment"]["methods_to_run"]

        if any(method in NEED_INIT_METHODS for method in methods):
            print('\n[Prewarm] checking/generating the NLE-MAP initialization model and candidates')
            _ = maybe_get_initial_theta(
                config=config,
                y_obs=y_dummy,
                seed=config["experiment"]["global_seed"] + 999,
            )

        if "NBE" in methods:
            print('\n[Prewarm] checking/training the pretrained NBE model')
            estimator = AmortizedNBEEstimator(
                method_name="NBE",
                n_data=max(config["experiment"]["n_data_list"]),
                sample_generator_cls=SampleGenerator,
                train_config=config["methods"]["NBE"],
            )

            if not os.path.exists(estimator.model_path):
                _ = estimator.fit(y_dummy)
            else:
                print(f'NBE model already exists: {estimator.model_path}')

        if "NLE" in methods:
            print('\n[Prewarm] checking/training the pretrained NLE model')
            estimator = AmortizedNLEEstimator(
                method_name="NLE",
                prior=sbi_prior,
                simulator=sbi_simulator,
                n_data=config["experiment"]["n_data_list"][0],
                sample_generator_cls=SampleGenerator,
                train_config=config["methods"]["NLE"],
            )

            if not os.path.exists(estimator.model_path):
                _ = estimator.fit(y_dummy)
            else:
                print(f'NLE model already exists: {estimator.model_path}')

        atomic_write_text(flag_path, "ready\n")

        if os.path.exists(fail_path):
            os.remove(fail_path)

        print('========== shared pretrained resources are ready ==========\n')

    except Exception:
        atomic_write_text(fail_path, traceback.format_exc())
        print(traceback.format_exc())
        raise


def wait_for_shared_assets(config, timeout_seconds=6 * 3600):
    if not config["parallel"].get("prewarm_shared_assets", True):
        return

    flag_path = shared_assets_flag_path(config)
    fail_path = shared_assets_fail_path(config)

    if os.path.exists(flag_path):
        return

    print('This shard is waiting for shard 0 to prepare shared pretrained resources...')
    start_time = time.time()

    while True:
        if os.path.exists(flag_path):
            print('Shared pretrained resources are ready; this shard is starting.')
            return

        if os.path.exists(fail_path):
            with open(fail_path, "r", encoding="utf-8") as f:
                msg = f.read()
            raise RuntimeError(f'Failed to prepare shared pretrained resources:\n{msg}')

        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(
                f'Timed out while waiting for shared pretrained resources after {timeout_seconds} seconds. Confirm that shard 0 has started or use --no-prewarm.'
            )

        time.sleep(5)


# ============================================================
# 11. Single-task execution
# ============================================================

def run_one_task(config, task):
    n_data = task["n_data"]
    current_run = task["current_run"]
    seed = task["seed"]

    methods_to_run = config["experiment"]["methods_to_run"]
    root_dir = config["paths"]["root_dir"]
    theta_header = config["dimensions"]["theta_header"]

    n_data_dir = os.path.join(root_dir, f"n_data_{n_data}")
    run_dir = os.path.join(n_data_dir, f"run_{current_run}")

    os.makedirs(run_dir, exist_ok=True)

    done_flag = os.path.join(run_dir, "_DONE.txt")
    fail_flag = os.path.join(run_dir, "_FAILED.txt")

    if config["parallel"].get("skip_finished", True) and os.path.exists(done_flag):
        print(f'Skipping completed task: n_data={n_data}, run={current_run}')
        return [], [], []

    print(
        f"\n================ "
        f"Task {task['task_index']} | "
        f"Run {current_run} | "
        f"n_data={n_data} | "
        f"seed={seed} "
        f"================"
    )

    try:
        ExperimentUtils.set_global_seed(seed)

        true_theta = (
            SampleGenerator
            .sample_prior(n_samples=1, seed=seed)
            .squeeze(0)
            .to(device)
        )

        y_obs = (
            SampleGenerator(true_theta, n_samples=n_data, seed=seed)
            .generate_samples()
            .detach()
            .to(device)
        )

        print(f'True parameter: {true_theta.detach().cpu().numpy()}')

        save_array_to_csv(
            true_theta,
            os.path.join(run_dir, "true_theta.csv"),
            header=theta_header,
        )

        save_array_to_csv(
            y_obs,
            os.path.join(run_dir, "y_obs.csv"),
        )

        init_theta = maybe_get_initial_theta(
            config=config,
            y_obs=y_obs,
            seed=seed,
        )

        if init_theta is not None:
            print(f'Initial parameter value: {init_theta}')
            save_array_to_csv(
                init_theta,
                os.path.join(run_dir, "init_theta.csv"),
                header=theta_header,
            )

        run_thetas = {}

        params_summary = []
        metrics_summary = []
        mle_comparison_summary = []

        for method_name in methods_to_run:
            try:
                final_theta, loss_history, theta_history, time_used, n_sim = run_estimation_method(
                    method_name=method_name,
                    y_obs=y_obs,
                    init_theta=init_theta,
                    config=config,
                    n_data=n_data,
                )

                final_theta = final_theta.detach().to(device)
                run_thetas[method_name] = final_theta.clone().detach()

                metrics = evaluate_and_save_method(
                    method_name=method_name,
                    true_theta=true_theta,
                    final_theta=final_theta,
                    y_obs=y_obs,
                    time_used=time_used,
                    n_sim=n_sim,
                    loss_history=loss_history,
                    theta_history=theta_history,
                    run_dir=run_dir,
                    seed=seed,
                    config=config,
                )

                params_summary.append({
                    "n_data": n_data,
                    "run_id": current_run,
                    "method": method_name,
                    "theta_values": final_theta.detach().cpu().numpy().tolist(),
                })

                record = {
                    "n_data": n_data,
                    "run_id": current_run,
                    "method": method_name,
                    "time_used": time_used,
                    "n_sim": n_sim,
                    "true_theta": true_theta.detach().cpu().numpy().tolist(),
                    "estimate_theta": final_theta.detach().cpu().numpy().tolist(),
                }
                record.update(metrics)
                metrics_summary.append(record)

            except Exception as e:
                print(f'❌ {method_name} method failed in run {current_run} failed: {repr(e)}')

                error_path = os.path.join(run_dir, f"{method_name}_ERROR.txt")
                with open(error_path, "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())

        if "MLE" in run_thetas:
            print('\nComputing discrepancies between the other methods and the MLE estimate...')

            mle_baseline_theta = run_thetas["MLE"]

            for method_name, est_theta in run_thetas.items():
                if method_name == "MLE":
                    continue

                diff_metrics = compute_estimation_metrics(
                    sample_generator_cls=SampleGenerator,
                    true_theta=mle_baseline_theta,
                    estimate_theta=est_theta,
                    y_obs=y_obs,
                    device=device,
                    seed=seed,
                )

                comp_record = {
                    "n_data": n_data,
                    "run_id": current_run,
                    "baseline_method": "MLE",
                    "compare_method": method_name,
                }

                for k, v in diff_metrics.items():
                    if isinstance(v, (int, float, list)):
                        comp_record[f"vs_MLE_{k}"] = v

                mle_comparison_summary.append(comp_record)

        atomic_write_text(done_flag, "done\n")

        if os.path.exists(fail_flag):
            os.remove(fail_flag)

        return params_summary, metrics_summary, mle_comparison_summary

    except Exception:
        atomic_write_text(fail_flag, traceback.format_exc())
        print(f'[ERROR] Task failed: n_data={n_data}, run={current_run}')
        print(traceback.format_exc())

        return [], [], []

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# 12. Worker workflow
# ============================================================

def run_worker(config):
    global_seed = config["experiment"]["global_seed"]
    root_dir = config["paths"]["root_dir"]
    methods_to_run = config["experiment"]["methods_to_run"]

    shard_id = config["parallel"]["shard_id"]
    num_shards = config["parallel"]["num_shards"]

    print(f'Compute device: {device}')
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
        print(f"current device = {torch.cuda.current_device()}")
        print(f"device name = {torch.cuda.get_device_name(torch.cuda.current_device())}")

        torch.backends.cudnn.benchmark = True

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print(f"Experiment name: {config['experiment']['example_name']}")
    print(f'Methods: {methods_to_run}')
    print(f"root_dir: {root_dir}")
    print(f"shard_id / num_shards: {shard_id} / {num_shards}")

    os.makedirs(root_dir, exist_ok=True)

    ExperimentUtils.set_global_seed(global_seed)

    # Only shard 0 writes the main configuration and prepares shared resources
    if shard_id == 0:
        save_experiment_config(config, root_dir)
        prepare_shared_assets(config)
    else:
        wait_for_shared_assets(config)

    all_tasks, shard_tasks = get_shard_tasks(config)

    print(f'Total tasks: {len(all_tasks)}')
    print(f'Tasks assigned to this shard: {len(shard_tasks)}')

    shard_dir = os.path.join(root_dir, f"_shard_{shard_id}")
    os.makedirs(shard_dir, exist_ok=True)

    params_summary_all = []
    metrics_summary_all = []
    mle_comparison_summary_all = []

    for task in shard_tasks:
        params_summary, metrics_summary, mle_comparison_summary = run_one_task(
            config=config,
            task=task,
        )

        params_summary_all.extend(params_summary)
        metrics_summary_all.extend(metrics_summary)
        mle_comparison_summary_all.extend(mle_comparison_summary)

        # Save after every task so interruptions do not lose the summary
        save_shard_summaries(
            shard_dir=shard_dir,
            params_summary=params_summary_all,
            metrics_summary=metrics_summary_all,
            mle_comparison_summary=mle_comparison_summary_all,
        )

    save_shard_summaries(
        shard_dir=shard_dir,
        params_summary=params_summary_all,
        metrics_summary=metrics_summary_all,
        mle_comparison_summary=mle_comparison_summary_all,
    )

    print(
        f'\n[OK] Worker completed: shard {shard_id}/{num_shards}, results directory: {root_dir}'
    )


# ============================================================
# 13. Merge shard results
# ============================================================

def read_csv_records(path):
    if not os.path.exists(path):
        return []

    if os.path.getsize(path) == 0:
        return []

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def merge_one_summary(root_dir, filename, out_filename):
    shard_dirs = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if d.startswith("_shard_") and os.path.isdir(os.path.join(root_dir, d))
    ]

    all_records = []

    for shard_dir in sorted(shard_dirs):
        path = os.path.join(shard_dir, filename)
        all_records.extend(read_csv_records(path))

    out_path = os.path.join(root_dir, out_filename)
    if filename == "metrics_summary.csv":
        all_records = apply_nbe_training_cost_once(all_records, root_dir)
    save_records_to_csv(all_records, out_path)

    print(f'Merge completed: {out_path} | rows={len(all_records)}')


def merge_shard_summaries(root_dir):
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f'root_dir does not exist: {root_dir}')

    merge_one_summary(
        root_dir=root_dir,
        filename="params_summary.csv",
        out_filename="params_summary_merged.csv",
    )

    merge_one_summary(
        root_dir=root_dir,
        filename="metrics_summary.csv",
        out_filename="metrics_summary_merged.csv",
    )

    merge_one_summary(
        root_dir=root_dir,
        filename="mle_comparison_summary.csv",
        out_filename="mle_comparison_summary_merged.csv",
    )

    print(f'\n[OK] All shard summaries merged: {root_dir}')


# ============================================================
# 14. Launcher for multiple workers
# ============================================================

def launch_workers(args):
    config = build_experiment_config()
    config = apply_cli_overrides(config, args)

    root_dir = config["paths"]["root_dir"]
    os.makedirs(root_dir, exist_ok=True)

    gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if len(gpu_ids) == 0:
        raise ValueError('--gpus must not be empty, for example --gpus 0,1,2,3')

    workers_per_gpu = int(args.workers_per_gpu)
    if workers_per_gpu <= 0:
        raise ValueError('--workers-per-gpu must be >= 1')

    total_workers = len(gpu_ids) * workers_per_gpu

    print(f'Preparing to launch the multi-device experiment')
    print(f"gpus = {gpu_ids}")
    print(f"workers_per_gpu = {workers_per_gpu}")
    print(f"total_workers = {total_workers}")
    print(f"root_dir = {root_dir}")

    logs_dir = os.path.join(root_dir, "_logs")
    os.makedirs(logs_dir, exist_ok=True)

    processes = []

    for worker_id in range(total_workers):
        gpu_id = gpu_ids[worker_id % len(gpu_ids)]

        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--mode", "worker",
            "--gpu-id", str(gpu_id),
            "--shard-id", str(worker_id),
            "--num-shards", str(total_workers),
            "--exp-tag", str(config["experiment"]["experiment_time"]),
        ]

        if args.n_runs is not None:
            cmd += ["--n-runs", str(args.n_runs)]

        if args.n_data_list:
            cmd += ["--n-data-list", str(args.n_data_list)]

        if args.methods:
            cmd += ["--methods", str(args.methods)]

        if args.manual_seed_offset is not None:
            cmd += ["--manual-seed-offset", str(args.manual_seed_offset)]

        if args.no_skip_finished:
            cmd += ["--no-skip-finished"]

        if args.no_prewarm:
            cmd += ["--no-prewarm"]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        log_path = os.path.join(logs_dir, f"worker_{worker_id}_gpu_{gpu_id}.log")
        log_file = open(log_path, "w", encoding="utf-8")

        print(f'Starting worker {worker_id} on GPU {gpu_id}')
        print(f'Log: {log_path}')

        p = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        processes.append((worker_id, gpu_id, p, log_file, log_path))

    failed = []

    for worker_id, gpu_id, p, log_file, log_path in processes:
        ret = p.wait()
        log_file.close()

        if ret != 0:
            failed.append((worker_id, gpu_id, ret, log_path))
            print(f'❌ worker {worker_id} on GPU {gpu_id} failed, returncode={ret}')
            print(f'   See log: {log_path}')
        else:
            print(f'✅ worker {worker_id} on GPU {gpu_id} completed')

    if len(failed) > 0:
        print('\nThe following workers failed:')
        for worker_id, gpu_id, ret, log_path in failed:
            print(f"worker={worker_id}, gpu={gpu_id}, returncode={ret}, log={log_path}")
        raise RuntimeError('Some workers failed; inspect the logs first.')

    if not args.no_merge_after:
        merge_shard_summaries(root_dir)

    print(f'\n[OK] Multi-GPU experiment completed: {root_dir}')


# ============================================================
# 15. Command-line interface
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="worker",
        choices=["worker", "launch", "merge"],
        help='worker: run one shard in this process; launch: start workers across devices; merge: merge shard results.',
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default="cpu",
        help='GPU identifiers for launch mode, for example 0,1,2,3.',
    )

    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help='Number of workers per GPU; start with 1.',
    )

    parser.add_argument(
        "--gpu-id",
        type=str,
        default="auto",
        help='GPU assigned to worker mode; normally supplied automatically by launch.',
    )

    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--exp-tag",
        type=str,
        default="",
        help='Fixed experiment-directory tag; all workers must use the same exp-tag.',
    )

    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--n-data-list",
        type=str,
        default="",
        help='For example: 10,50,100,500',
    )

    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help='For example: Ours,MLE or Ours,WME_p1,NBE,MLE',
    )

    parser.add_argument(
        "--manual-seed-offset",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--no-skip-finished",
        action="store_true",
        help='Completed runs containing _DONE.txt are skipped by default; set this flag to rerun them.',
    )

    parser.add_argument(
        "--no-prewarm",
        action="store_true",
        help='Disable shard-0 prewarming only when all pretrained models are already available.',
    )

    parser.add_argument(
        "--no-merge-after",
        action="store_true",
        help='Do not merge shard summaries automatically after launch completes.',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "launch":
        launch_workers(args)
        return

    config = build_experiment_config()
    config = apply_cli_overrides(config, args)

    if args.mode == "worker":
        run_worker(config)
        return

    if args.mode == "merge":
        merge_shard_summaries(config["paths"]["root_dir"])
        return

    raise ValueError(f'Unknown mode: {args.mode}')


if __name__ == "__main__":
    main()
