from __future__ import annotations
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
import argparse
import csv
import json
import subprocess
import time
import traceback
from datetime import datetime
from pprint import pformat
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from geomloss import SamplesLoss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path
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
current_dir = os.path.abspath(os.path.dirname(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from estimators.mwe import MWE
from estimators.annd import AdversarialEstimator
from estimators.vsm import AnnealedMinimaxTrainer
from estimators.adaimle import AdaptiveIMLEParameterEstimator
from estimators.nle import AmortizedNLEEstimator
from estimators.nbe import AmortizedNBEEstimator, apply_nbe_training_cost_once
from estimators.npe_c import AmortizedNPECSetEstimator
from utils.results import save_array_to_csv, ExperimentUtils
from utils.initialization import find_best_initial_theta_via_nle_map
from models.mnist import *
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
            hidden=128,
            alpha=1.0,
            num_blocks=2,
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
NEED_INIT_METHODS = {"Ours", "WME_p1", "WME_p2", "ANND", "AdaIMLE"}

def make_theta_header(theta_dim: int):
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

    jsonable_config = make_jsonable(config)

    config_json_path = os.path.join(root_dir, "experiment_config.json")
    config_txt_path = os.path.join(root_dir, "experiment_config_readable.txt")

    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(jsonable_config, f, ensure_ascii=False, indent=4)

    with open(config_txt_path, "w", encoding="utf-8") as f:
        f.write(pformat(jsonable_config, width=120, sort_dicts=False))

    print(f'[OK] Experiment configuration saved: {config_json_path}')


def save_history(history, file_path):
    if history is None:
        return

    if isinstance(history, list) and len(history) == 0:
        return

    if isinstance(history, list) and all(isinstance(x, dict) for x in history):
        fieldnames = sorted(set().union(*(row.keys() for row in history)))

        with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
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
        "train_samples",
        "n_train",
        "n_test",
        "run_id",
        "method",
        "time_used",
        "n_sim",
        "train_accuracy",
        "test_accuracy",
        "train_f1",
        "test_f1",
        "train_nll",
        "test_nll",
        "train_auc",
        "test_auc",
        "theta_norm",
    ]

    ordered = [k for k in preferred_order if k in fieldnames]
    ordered += [k for k in fieldnames if k not in ordered]

    with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()

        for row in records:
            writer.writerow({k: _jsonify_for_csv(row.get(k, "")) for k in ordered})


def save_shard_summaries(shard_dir, params_summary, metrics_summary):
    os.makedirs(shard_dir, exist_ok=True)

    save_records_to_csv(
        params_summary,
        os.path.join(shard_dir, "params_summary.csv"),
    )

    save_records_to_csv(
        metrics_summary,
        os.path.join(shard_dir, "metrics_summary.csv"),
    )


def atomic_write_text(path, text):
    tmp = path + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)

    os.replace(tmp, path)


# ============================================================
# Configuration
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
    n_runs = 20
    train_samples = 100
    n_data_list = [train_samples]

    experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    root_dir = os.path.join(
        root_experiment_dir,
        f"Exp_{example_name}_{experiment_time}",
    )
    pretrained_root = root_experiment_dir
    mnist_cfg = {
        "d": 20,
        "r_dim": 20,
        "data_home": "./mnist_openml_cache",
        "train_samples": train_samples,
        "max_test_samples": None,       
        "pca_whiten": True,
        "init_strategy": "class_mean",
        "prior_bound": 3.0,
        "n_mc_predict": 512,
        "negative_digit":1,
        "positive_digit":2
    }

    d = mnist_cfg["d"]
    r_dim = mnist_cfg["r_dim"]

    theta_dim = 1 + d + d * r_dim
    obs_dim = d + 1
    x_dim=obs_dim
    z_dim = 1 + r_dim + 1


    methods_to_run = [
        "Ours",
        "WME_p2",
        "AdaIMLE",
        "ANND",
        "NBE",
        "NPE_C"
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
            "disc_lr": 0.01,
            "disc_train_iter": 100,
            "outer_optimizer": "adam",
            "theta_lr": 1e-3,
            "theta_grad_clip": 5.0,
            "bounds_lb": [-3.0] * theta_dim,
            "bounds_ub": [3.0] * theta_dim,
        },
        "AdaIMLE": {
            "latent_dim": z_dim,
            "pool_factor": 1.0,
            "delta": 0.95,
            "initial_threshold": None,
            "threshold_update_rule": "shrink_current",
            "change_coef": 0.05,
            "obs_batch_size": None,
            "learning_rate": 1e-2,
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
            "m_values": [1,10,100],
            "m_cutoffs": [5,50],
            "x_dim": x_dim,
            "theta_dim": theta_dim,
            "K_train": 2250,
            "K_val": 500,
            "batch_size": 128,
            "epochs": 2000,
            "target_transform": target_transform,
            "patience":200,
            "learn1": 1e-3,
            "learn2": 1e-3,
            "LOSS": "MSE"
        },
        "NPE_C": {
            "model_dir": os.path.join(pretrained_root, "pretrained_npec_models"),
            "num_rounds": 2,
            "num_simulations": 2500,
            "batch_size": 128,
            "density_model": "maf",
            "density_hidden_features": 50,
            "density_num_transforms": 3,
            "latent_dim": 16,
            "embedding_dim": 16,
            "trial_num_hiddens": 64,
            "trial_num_layers": 2,
            "perm_num_hiddens": 32,
            "perm_num_layers": 1,
            "z_score_x": "independent",
            "learning_rate": 5e-3,
            "validation_fraction": 0.1,
            "stop_after_epochs": 20,
            "clip_max_norm": 5.0,
            "num_atoms": 10,
            "max_num_epochs": 300,
            "use_combined_loss": True,
            "num_posterior_samples": 5000,
            "posterior_sampling_method": "importance",
            "importance_oversampling_factor": 4,
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
        "model_path": os.path.join(
            init_dir,
            f"{example_name.lower()}_d{d}_r{r_dim}_train{train_samples}_nle_likelihood.pt",
        ),
        "candidate_path": os.path.join(
            init_dir,
            f"{example_name.lower()}_d{d}_r{r_dim}_train{train_samples}_nle_candidates.pt",
        ),
        "retrain": False,
        "retrain_candidates": False,
        "candidate_batch_size": 64,
        "show_progress_bars": True,
    }
    for method_key in ("Ours", "WME", "ANND", "AdaIMLE"):
        train_configs[method_key]["n_sim"] = n_sim
        train_configs[method_key]["n_init_sim"] = n_init_sim

    return {
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
        "mnist": mnist_cfg,
        "dimensions": {
            "feature_dim": d,
            "x_dim": obs_dim,
            "theta_dim": theta_dim,
            "z_dim": z_dim,
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
        },
    }


def apply_cli_overrides(config, args):
    if args.exp_tag:
        config["experiment"]["experiment_time"] = args.exp_tag
        config["paths"]["root_dir"] = os.path.join(
            config["paths"]["root_experiment_dir"],
            f"Exp_{config['experiment']['example_name']}_{args.exp_tag}",
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

    if args.d is not None:
        config["mnist"]["d"] = int(args.d)

    if args.r_dim is not None:
        config["mnist"]["r_dim"] = int(args.r_dim)

    if args.data_home:
        config["mnist"]["data_home"] = args.data_home

    if args.train_samples is not None:
        config["mnist"]["train_samples"] = int(args.train_samples)
        config["experiment"]["n_data_list"] = [int(args.train_samples)]

    if args.max_test_samples is not None:
        config["mnist"]["max_test_samples"] = (
            int(args.max_test_samples)
            if int(args.max_test_samples) > 0
            else None
        )

    if args.n_mc_predict is not None:
        config["mnist"]["n_mc_predict"] = int(args.n_mc_predict)

    if args.no_whiten:
        config["mnist"]["pca_whiten"] = False

    config["parallel"]["shard_id"] = int(args.shard_id)
    config["parallel"]["num_shards"] = int(args.num_shards)
    config["parallel"]["skip_finished"] = not args.no_skip_finished
    d = int(config["mnist"]["d"])
    r_dim = int(config["mnist"]["r_dim"])
    theta_dim = 1 + d + d * r_dim
    obs_dim = d + 1
    z_dim = 1 + r_dim + 1

    config["dimensions"]["feature_dim"] = d
    config["dimensions"]["x_dim"] = obs_dim
    config["dimensions"]["theta_dim"] = theta_dim
    config["dimensions"]["z_dim"] = z_dim
    config["dimensions"]["theta_header"] = make_theta_header(theta_dim)

    config["methods"]["ANND"]["bounds_lb"] = [-3.0] * theta_dim
    config["methods"]["ANND"]["bounds_ub"] = [3.0] * theta_dim
    config["methods"]["AdaIMLE"]["latent_dim"] = z_dim
    config["methods"]["NBE"]["x_dim"] = obs_dim
    config["methods"]["NBE"]["theta_dim"] = theta_dim
    config["methods"]["NBE"]["target_transform"] = target_transform

    config["experiment"]["device"] = str(device)
    config = refresh_init_config_paths(config)

    return config


def update_dynamic_config_from_problem(config, problem):
    config["dimensions"]["feature_dim"] = int(problem.feature_dim)
    config["dimensions"]["x_dim"] = int(problem.obs_dim)
    config["dimensions"]["theta_dim"] = int(problem.theta_dim)
    config["dimensions"]["z_dim"] = int(problem.z_dim)
    config["dimensions"]["theta_header"] = make_theta_header(problem.theta_dim)

    config["methods"]["ANND"]["bounds_lb"] = [-3.0] * problem.theta_dim
    config["methods"]["ANND"]["bounds_ub"] = [3.0] * problem.theta_dim

    config["methods"]["AdaIMLE"]["latent_dim"] = problem.z_dim

    config["methods"]["NBE"]["x_dim"] = problem.obs_dim
    config["methods"]["NBE"]["theta_dim"] = problem.theta_dim
    config["methods"]["NBE"]["target_transform"] = target_transform

    return config


# ============================================================
# Single-method execution
# ============================================================


def refresh_init_config_paths(config):
    'Refresh init config paths.'
    example_name = config["experiment"]["example_name"]
    pretrained_root = config["paths"]["pretrained_root"]
    init_dir = os.path.join(pretrained_root, "pretrained_nle_init")

    d = int(config["mnist"]["d"])
    r_dim = int(config["mnist"]["r_dim"])
    train_samples = int(config["mnist"]["train_samples"])

    config["init"]["model_path"] = os.path.join(
        init_dir,
        f"{example_name.lower()}_d{d}_r{r_dim}_train{train_samples}_nle_likelihood.pt",
    )
    config["init"]["candidate_path"] = os.path.join(
        init_dir,
        f"{example_name.lower()}_d{d}_r{r_dim}_train{train_samples}_nle_candidates.pt",
    )

    return config


def maybe_get_initial_theta(config, problem, seed):
    'Maybe get initial theta.'
    methods_to_run = config["experiment"]["methods_to_run"]

    if not any(method in NEED_INIT_METHODS for method in methods_to_run):
        print('No selected method requires init_theta; skipping NLE-MAP initialization.')
        return None

    init_cfg = config["init"]

    print('\n========== selecting the initial value with offline NLE-MAP ==========')
    print(f"init model_path     = {init_cfg['model_path']}")
    print(f"init candidate_path = {init_cfg['candidate_path']}")

    init_theta = find_best_initial_theta_via_nle_map(
        real_observations=problem.y_obs,
        generator_cls=problem.sample_generator_cls,
        prior=problem.prior,
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

    init_theta = init_theta.detach().to(device).requires_grad_(True)

    print(f'NLE-MAP initial parameter value: {init_theta.detach().cpu().numpy()}')
    print('========== NLE-MAP initialization completed ==========\n')

    return init_theta


def normalize_estimator_output(result, estimator=None):
    if not isinstance(result, tuple):
        raise ValueError('estimator.train()/fit() must return a tuple.')

    if len(result) == 5:
        return result

    if len(result) == 4:
        final_theta, loss_history, theta_history, time_used = result
        n_sim = getattr(estimator, "total_virtual_samples", 0)
        return final_theta, loss_history, theta_history, time_used, n_sim

    raise ValueError(f'Unexpected estimator return length: {len(result)}')


def run_estimation_method(method_name, problem, config):
    print(f'\n--- Starting {method_name} method estimation ---')

    y_obs = problem.y_obs
    init_theta = problem.init_theta
    sample_generator_cls = problem.sample_generator_cls
    train_configs = config["methods"]

    n_data_for_sbi = problem.n_train

    if method_name == "Ours":
        T_net = TNet(
            x_dim=problem.obs_dim,
            theta_dim=problem.theta_dim,
            hidden=256,
        ).to(device)

        trainer = AnnealedMinimaxTrainer(
            y_obs=y_obs,
            init_theta=init_theta.clone().detach().to(device).requires_grad_(True),
            sample_generator_cls=sample_generator_cls,
            U_net=T_net,
            train_config=train_configs["Ours"],
        )

        return normalize_estimator_output(trainer.train(), trainer)

    if method_name == "WME_p1":
        loss_fn = SamplesLoss(loss="sinkhorn", p=1, blur=1e-5, debias=True)

        estimator = MWE(
            y_obs=y_obs,
            theta_init=init_theta.clone().detach().to(device).requires_grad_(True),
            sample_generator_cls=sample_generator_cls,
            loss_func=loss_fn,
            train_params=train_configs["WME"],
        )

        return normalize_estimator_output(estimator.train(), estimator)

    if method_name == "WME_p2":
        loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=1e-5, debias=True)

        estimator = MWE(
            y_obs=y_obs,
            theta_init=init_theta.clone().detach().to(device).requires_grad_(True),
            sample_generator_cls=sample_generator_cls,
            loss_func=loss_fn,
            train_params=train_configs["WME"],
        )

        return normalize_estimator_output(estimator.train(), estimator)

    if method_name == "ANND":
        estimator = AdversarialEstimator(
            y_obs=y_obs,
            init_theta=init_theta,
            sample_generator_cls=sample_generator_cls,
            train_config=train_configs["ANND"],
        )

        return normalize_estimator_output(estimator.train(), estimator)

    if method_name == "AdaIMLE":
        estimator = AdaptiveIMLEParameterEstimator(
            y_obs=y_obs,
            theta_init=init_theta.clone().detach().to(device),
            simulator_fn=problem.adaimle_simulator_wrapper,
            config=train_configs["AdaIMLE"],
        )

        return normalize_estimator_output(estimator.train(), estimator)

    if method_name == "NBE":
        estimator = AmortizedNBEEstimator(
            method_name=method_name,
            n_data=n_data_for_sbi,
            sample_generator_cls=sample_generator_cls,
            train_config=train_configs["NBE"],
        )

        return normalize_estimator_output(estimator.fit(y_obs), estimator)

    if method_name == "NLE":
        estimator = AmortizedNLEEstimator(
            method_name="NLE",
            prior=problem.sbi_prior,
            simulator=problem.sbi_simulator,
            n_data=n_data_for_sbi,
            sample_generator_cls=sample_generator_cls,
            train_config=train_configs["NLE"],
        )

        return normalize_estimator_output(estimator.fit(y_obs), estimator)

    if method_name == "NPE_C":
        estimator = AmortizedNPECSetEstimator(
            method_name=method_name,
            prior=problem.sbi_prior,
            simulator=problem.sbi_simulator,
            n_data=n_data_for_sbi,
            sample_generator_cls=sample_generator_cls,
            train_config=train_configs["NPE_C"],
        )

        return normalize_estimator_output(estimator.fit(y_obs), estimator)

    raise ValueError(f'Unknown estimation method: {method_name}')


def evaluate_and_save_method(
    method_name,
    final_theta,
    problem,
    time_used,
    n_sim,
    loss_history,
    theta_history,
    run_dir,
    seed,
    config,
):
    print(f'Computing {method_name} train/test classification metrics...')

    method_dir = os.path.join(run_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    final_theta = final_theta.detach().to(device).float()

    save_array_to_csv(
        final_theta,
        os.path.join(method_dir, "final_theta.csv"),
        header=config["dimensions"]["theta_header"],
    )

    save_array_to_csv(
        np.array([time_used]),
        os.path.join(method_dir, "time_used.csv"),
        header=["time_seconds"],
    )

    save_array_to_csv(
        np.array([n_sim]),
        os.path.join(method_dir, "n_sim.csv"),
        header=["virtual_samples"],
    )

    save_history(loss_history, os.path.join(method_dir, "loss_history.csv"))
    save_history(theta_history, os.path.join(method_dir, "theta_history.csv"))

    metrics = compute_classification_metrics(
        theta_hat=final_theta,
        problem=problem,
        n_mc_predict=config["mnist"]["n_mc_predict"],
        seed=seed,
    )

    metrics["time_used"] = float(time_used)
    metrics["n_sim"] = int(n_sim)
    metrics["method"] = method_name

    ExperimentUtils.save_dict_to_csv_fixed(
        metrics,
        os.path.join(method_dir, "evaluation_metrics.csv"),
    )

    print(
        f"{method_name} completed | elapsed: {time_used:.2f}s | n_sim: {n_sim} | train_acc: {metrics.get('train_accuracy', 0):.4f} | test_acc: {metrics.get('test_accuracy', 0):.4f}"
    )

    return metrics


# ============================================================
# Task construction and execution
# ============================================================

def build_run_tasks(config):
    tasks = []
    task_index = 0

    for n_data in config["experiment"]["n_data_list"]:
        for run_id in range(config["experiment"]["n_runs"]):
            seed = (
                config["experiment"]["run_seed_base"]
                + run_id
                + config["experiment"].get("manual_seed_offset", 0)
            )

            tasks.append(
                {
                    "task_index": task_index,
                    "n_data": int(n_data),
                    "run_id": int(run_id),
                    "current_run": int(run_id + 1),
                    "seed": int(seed),
                }
            )

            task_index += 1

    return tasks


def get_shard_tasks(config):
    all_tasks = build_run_tasks(config)

    shard_id = config["parallel"]["shard_id"]
    num_shards = config["parallel"]["num_shards"]

    shard_tasks = [
        task for task in all_tasks
        if task["task_index"] % num_shards == shard_id
    ]

    return all_tasks, shard_tasks


def run_one_task(config, task):
    current_run = task["current_run"]
    seed = task["seed"]

    methods_to_run = config["experiment"]["methods_to_run"]
    root_dir = config["paths"]["root_dir"]

    train_samples = int(config["mnist"]["train_samples"])

    run_dir = os.path.join(root_dir, f"train_{train_samples}_test_full", f"run_{current_run}")
    os.makedirs(run_dir, exist_ok=True)

    done_flag = os.path.join(run_dir, "_DONE.txt")
    fail_flag = os.path.join(run_dir, "_FAILED.txt")

    if config["parallel"].get("skip_finished", True) and os.path.exists(done_flag):
        print(f'Skipping completed task: train_samples={train_samples}, run={current_run}')
        return [], []

    print(
        f"\n================ "
        f"Task {task['task_index']} | "
        f"Run {current_run} | "
        f"train={train_samples} | test=full | seed={seed} "
        f"================"
    )

    try:
        ExperimentUtils.set_global_seed(seed)

        mnist_cfg = config["mnist"]

        problem = build_mnist_mixed_logit_problem(
            d=mnist_cfg["d"],
            r_dim=mnist_cfg["r_dim"],
            data_home=mnist_cfg["data_home"],
            seed=seed,
            train_samples=mnist_cfg["train_samples"],
            max_test_samples=mnist_cfg["max_test_samples"],
            pca_whiten=mnist_cfg["pca_whiten"],
            init_strategy=mnist_cfg["init_strategy"],
            prior_bound=mnist_cfg["prior_bound"],
            negative_digit=mnist_cfg["negative_digit"],
            positive_digit=mnist_cfg["positive_digit"],
        )

        config = update_dynamic_config_from_problem(config, problem)
        config = refresh_init_config_paths(config)

        print(
            f"MNIST problem: "
            f"n_train={problem.n_train}, n_test={problem.n_test}, "
            f"d={problem.feature_dim}, r={problem.r_dim}, "
            f"obs_dim={problem.obs_dim}, theta_dim={problem.theta_dim}, z_dim={problem.z_dim}"
        )

        save_array_to_csv(problem.y_obs, os.path.join(run_dir, "y_obs.csv"))

        
        
        init_theta = maybe_get_initial_theta(
            config=config,
            problem=problem,
            seed=seed,
        )

        if init_theta is not None:
            problem.init_theta = init_theta
            save_array_to_csv(
                init_theta,
                os.path.join(run_dir, "init_theta.csv"),
                header=config["dimensions"]["theta_header"],
            )
        else:
            
            save_array_to_csv(
                problem.init_theta,
                os.path.join(run_dir, "init_theta_backup_class_mean.csv"),
                header=config["dimensions"]["theta_header"],
            )

        params_summary = []
        metrics_summary = []

        for method_name in methods_to_run:
            try:
                final_theta, loss_history, theta_history, time_used, n_sim = run_estimation_method(
                    method_name=method_name,
                    problem=problem,
                    config=config,
                )

                metrics = evaluate_and_save_method(
                    method_name=method_name,
                    final_theta=final_theta,
                    problem=problem,
                    time_used=time_used,
                    n_sim=n_sim,
                    loss_history=loss_history,
                    theta_history=theta_history,
                    run_dir=run_dir,
                    seed=seed,
                    config=config,
                )

                params_summary.append(
                    {
                        "train_samples": problem.train_samples,
                        "n_train": problem.n_train,
                        "n_test": problem.n_test,
                        "run_id": current_run,
                        "method": method_name,
                        "theta_values": final_theta.detach().cpu().numpy().tolist(),
                    }
                )

                record = {
                    "train_samples": problem.train_samples,
                    "n_train": problem.n_train,
                    "n_test": problem.n_test,
                    "run_id": current_run,
                    "method": method_name,
                    "time_used": time_used,
                    "n_sim": n_sim,
                    "estimate_theta": final_theta.detach().cpu().numpy().tolist(),
                }
                record.update(metrics)

                metrics_summary.append(record)

            except Exception as e:
                print(f'❌ {method_name} method failed in run {current_run} failed: {repr(e)}')

                with open(
                    os.path.join(run_dir, f"{method_name}_ERROR.txt"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(traceback.format_exc())

        atomic_write_text(done_flag, "done\n")

        if os.path.exists(fail_flag):
            os.remove(fail_flag)

        return params_summary, metrics_summary

    except Exception:
        atomic_write_text(fail_flag, traceback.format_exc())

        print(f'[ERROR] Task failed: train_samples={train_samples}, run={current_run}')
        print(traceback.format_exc())

        return [], []

    finally:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                print(f'[WARNING] CUDA memory cleanup failed, usually after an earlier OOM: {repr(e)}')


def run_worker(config):
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

    root_dir = config["paths"]["root_dir"]
    os.makedirs(root_dir, exist_ok=True)

    ExperimentUtils.set_global_seed(config["experiment"]["global_seed"])

    if config["parallel"]["shard_id"] == 0:
        save_experiment_config(config, root_dir)

    all_tasks, shard_tasks = get_shard_tasks(config)

    print(f"Experiment name: {config['experiment']['example_name']}")
    print(f"Methods: {config['experiment']['methods_to_run']}")
    print(f"root_dir: {root_dir}")
    print(f"shard_id / num_shards: {config['parallel']['shard_id']} / {config['parallel']['num_shards']}")
    print(f'Total tasks: {len(all_tasks)}')
    print(f'Tasks assigned to this shard: {len(shard_tasks)}')

    shard_dir = os.path.join(root_dir, f"_shard_{config['parallel']['shard_id']}")
    os.makedirs(shard_dir, exist_ok=True)

    params_summary_all = []
    metrics_summary_all = []

    for task in shard_tasks:
        params_summary, metrics_summary = run_one_task(config, task)

        params_summary_all.extend(params_summary)
        metrics_summary_all.extend(metrics_summary)

        save_shard_summaries(
            shard_dir,
            params_summary_all,
            metrics_summary_all,
        )

    save_shard_summaries(
        shard_dir,
        params_summary_all,
        metrics_summary_all,
    )

    print(
        f"\n[OK] Worker completed: shard {config['parallel']['shard_id']}/{config['parallel']['num_shards']}, results directory: {root_dir}"
    )


# ============================================================
# Merge and launch
# ============================================================

def read_csv_records(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def merge_one_summary(root_dir, filename, out_filename):
    shard_dirs = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if d.startswith("_shard_") and os.path.isdir(os.path.join(root_dir, d))
    ]

    all_records = []

    for shard_dir in sorted(shard_dirs):
        all_records.extend(
            read_csv_records(os.path.join(shard_dir, filename))
        )

    out_path = os.path.join(root_dir, out_filename)
    if filename == "metrics_summary.csv":
        all_records = apply_nbe_training_cost_once(all_records, root_dir)
    save_records_to_csv(all_records, out_path)

    print(f'Merge completed: {out_path} | rows={len(all_records)}')


def merge_shard_summaries(root_dir):
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f'root_dir does not exist: {root_dir}')

    merge_one_summary(root_dir, "params_summary.csv", "params_summary_merged.csv")
    merge_one_summary(root_dir, "metrics_summary.csv", "metrics_summary_merged.csv")

    print(f'\n[OK] All shard summaries merged: {root_dir}')


def launch_workers(args):
    config = apply_cli_overrides(build_experiment_config(), args)

    root_dir = config["paths"]["root_dir"]
    os.makedirs(root_dir, exist_ok=True)

    gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]

    if not gpu_ids:
        raise ValueError('--gpus must not be empty, for example --gpus 0,1,2,3')

    workers_per_gpu = int(args.workers_per_gpu)
    total_workers = len(gpu_ids) * workers_per_gpu

    logs_dir = os.path.join(root_dir, "_logs")
    os.makedirs(logs_dir, exist_ok=True)

    print(
        f'Preparing to launch the multi-device experiment | gpus={gpu_ids} | workers_per_gpu={workers_per_gpu} | total_workers={total_workers}'
    )

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

        for flag, value in [
            ("--n-runs", args.n_runs),
            ("--n-data-list", args.n_data_list),
            ("--methods", args.methods),
            ("--manual-seed-offset", args.manual_seed_offset),
            ("--d", args.d),
            ("--r-dim", args.r_dim),
            ("--data-home", args.data_home),
            ("--train-samples", args.train_samples),
            ("--max-test-samples", args.max_test_samples),
            ("--n-mc-predict", args.n_mc_predict),
        ]:
            if value not in [None, ""]:
                cmd += [flag, str(value)]

        if args.no_whiten:
            cmd += ["--no-whiten"]

        if args.no_skip_finished:
            cmd += ["--no-skip-finished"]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        log_path = os.path.join(
            logs_dir,
            f"worker_{worker_id}_gpu_{gpu_id}.log",
        )
        log_file = open(log_path, "w", encoding="utf-8")

        print(f'Starting worker {worker_id} on GPU {gpu_id} | log: {log_path}')

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
            print(f'❌ worker {worker_id} on GPU {gpu_id} failed, returncode={ret} | log={log_path}')
        else:
            print(f'✅ worker {worker_id} on GPU {gpu_id} completed')

    if failed:
        raise RuntimeError('Some workers failed; inspect the logs.')

    if not args.no_merge_after:
        merge_shard_summaries(root_dir)

    print(f'\n[OK] Multi-GPU experiment completed: {root_dir}')
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="launch", choices=["worker", "launch", "merge"])
    parser.add_argument("--gpus", type=str, default="cpu")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--gpu-id", type=str, default="auto")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--exp-tag", type=str, default="")

    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--n-data-list", type=str, default="")
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--manual-seed-offset", type=int, default=None)

    parser.add_argument("--d", type=int, default=None)
    parser.add_argument("--r-dim", type=int, default=None)
    parser.add_argument("--data-home", type=str, default="")

    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--n-mc-predict", type=int, default=None)

    parser.add_argument("--no-whiten", action="store_true")
    parser.add_argument("--no-skip-finished", action="store_true")
    parser.add_argument("--no-merge-after", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "launch":
        launch_workers(args)
        return

    config = apply_cli_overrides(build_experiment_config(), args)

    if args.mode == "worker":
        run_worker(config)
        return

    if args.mode == "merge":
        merge_shard_summaries(config["paths"]["root_dir"])
        return

    raise ValueError(f'Unknown mode: {args.mode}')


if __name__ == "__main__":
    main()
