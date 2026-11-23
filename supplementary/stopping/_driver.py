"""SLCP/GLM supplementary runs using the corresponding demo's n10 defaults.

The demo supplies model-specific configuration, simulator, critic and metrics.
This driver records stopping diagnostics and supports resumable shards.
Scientific dependencies are loaded only when an experiment is configured.
"""
import argparse
import csv
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
from pprint import pformat
import subprocess
import sys
import time
import traceback

from .reporting import load_run_records, save_run_records, write_csv, write_stopping_reports

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_FILE = Path(__file__).resolve()
CURRENT_DIR = CURRENT_FILE.parent
USER_HOME = CURRENT_DIR.parent
project_root = str(PROJECT_ROOT)
EXPERIMENT_TRAIN_CONFIGS = {}
DEMO = None
NEED_INIT_METHODS = {"Ours"}


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
    if DEMO is None:
        raise RuntimeError("Configure an SLCP or GLM stopping experiment first.")
    config = deepcopy(DEMO.build_experiment_config())
    example_name = EXPERIMENT_MODEL.upper()
    root_experiment_dir = CURRENT_DIR / f"{CURRENT_FILE.stem}_results"
    config["experiment"].update(example_name=example_name, methods_to_run=["Ours", "MLE"])
    config["paths"].update(
        user_home=str(USER_HOME), project_root=project_root,
        root_experiment_dir=str(root_experiment_dir),
        pretrained_root=str(root_experiment_dir),
        root_dir=str(root_experiment_dir / f"Exp_{example_name}_{config['experiment']['experiment_time']}"),
    )
    config["methods"] = {key: config["methods"][key] for key in ("Ours", "MLE")}
    config["methods"]["Ours"]["overall_stop_mode"] = "budget"
    for method, overrides in EXPERIMENT_TRAIN_CONFIGS.items():
        if method not in config["methods"]:
            raise ValueError(f"Unsupported method: {method}. Choose Ours or MLE.")
        config["methods"][method].update(overrides)
    init_dir = root_experiment_dir / "pretrained_nle_init"
    config["init"]["model_path"] = str(init_dir / f"{EXPERIMENT_MODEL}_nle_likelihood.pt")
    config["init"]["candidate_path"] = str(init_dir / f"{EXPERIMENT_MODEL}_nle_candidates.pt")
    config["ablation"] = {
        "entrypoint": str(CURRENT_FILE),
        "baseline_source": str(PROJECT_ROOT / f"demo_{example_name}" / "n10.py"),
        "train_configs_overrides": deepcopy(EXPERIMENT_TRAIN_CONFIGS),
    }
    return config


def apply_cli_overrides(config, args):
    if args.exp_tag:
        config["experiment"]["experiment_time"] = args.exp_tag
        config["paths"]["root_dir"] = os.path.join(
            config["paths"]["root_experiment_dir"],
            f"Exp_{config['experiment']['example_name']}_{args.exp_tag}"
        )

    if getattr(args, "stopping_params", ""):
        from .sensitivity import case_name, validate_case
        parameters = validate_case(json.loads(args.stopping_params))
        ours = config["methods"]["Ours"]
        if ours.get("overall_stop_mode") != "paper":
            raise ValueError("--stopping-params requires the overall-stopping experiment.")
        if parameters["n_sim"] < ours["n_init_sim"]:
            raise ValueError("n_sim must be at least n_init_sim (including initialization simulations).")
        ours.update(parameters)
        name = case_name(parameters)
        case_root = Path(config["paths"]["root_experiment_dir"]) / name
        config["paths"].update(
            root_experiment_dir=str(case_root), pretrained_root=str(case_root),
            root_dir=str(case_root / f"Exp_{config['experiment']['example_name']}_{config['experiment']['experiment_time']}"),
        )
        for key in ("model_path", "candidate_path"):
            config["init"][key] = str(case_root / "pretrained_nle_init" / Path(config["init"][key]).name)
        config["ablation"]["train_configs_overrides"].setdefault("Ours", {}).update(parameters)
        config["sensitivity"] = {"case_name": name, "parameters": parameters}

    if args.n_runs is not None:
        config["experiment"]["n_runs"] = int(args.n_runs)

    if args.n_data_list:
        config["experiment"]["n_data_list"] = [
            int(x.strip()) for x in args.n_data_list.split(",") if x.strip()
        ]

    if args.methods:
        methods = [x.strip() for x in args.methods.split(",") if x.strip()]
        if not methods or any(method not in {"Ours", "MLE"} for method in methods):
            raise ValueError("Choose Ours or MLE.")
        if len(set(methods)) != len(methods):
            raise ValueError("Duplicate methods are not allowed.")
        config["experiment"]["methods_to_run"] = methods

    if args.manual_seed_offset is not None:
        config["experiment"]["manual_seed_offset"] = int(args.manual_seed_offset)

    config["parallel"]["shard_id"] = int(args.shard_id)
    config["parallel"]["num_shards"] = int(args.num_shards)
    config["parallel"]["skip_finished"] = not args.no_skip_finished
    config["parallel"]["prewarm_shared_assets"] = not args.no_prewarm

    config["experiment"]["device"] = str(device)

    return config


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

        result = trainer.train()
        config["_vsm_run_diagnostics"] = {
            "termination_report": make_jsonable(trainer.termination_report),
            "stage_history": make_jsonable(trainer.stage_history),
        }
        return normalize_estimator_output(result, trainer)

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

    if method_name == "Ours":
        diagnostics = config["_vsm_run_diagnostics"]
        report = diagnostics["termination_report"]
        write_csv(Path(method_dir) / "stage_history.csv", diagnostics["stage_history"],
                  ("stage", "epsilon_used", "epsilon_next", "theta_change_inf", "n_sim"))
        atomic_write_text(os.path.join(method_dir, "stopping_summary.json"),
                          json.dumps(report, ensure_ascii=False, indent=2))
        metrics.update({f"vsm_{key}": value for key, value in report.items()})

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
        try:
            records = load_run_records(run_dir)
        except (OSError, ValueError) as exc:
            print(f'[WARNING] Completed task has no usable persisted records; rerunning: {exc}')
        else:
            print(f'Skipping completed task and retaining records: n_data={n_data}, run={current_run}')
            return tuple(records[key] for key in (
                "params_summary", "metrics_summary", "mle_comparison_summary"
            ))

    print(
        f"\n================ "
        f"Task {task['task_index']} | "
        f"Run {current_run} | "
        f"n_data={n_data} | "
        f"seed={seed} "
        f"================"
    )

    params_summary = []
    metrics_summary = []
    mle_comparison_summary = []
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

        method_errors = []

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
                method_errors.append(method_name)
                print(f'[ERROR] {method_name} method failed in run {current_run}: {repr(e)}')

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

        save_run_records(run_dir, {
            "params_summary": params_summary,
            "metrics_summary": metrics_summary,
            "mle_comparison_summary": mle_comparison_summary,
        })
        if method_errors:
            raise RuntimeError("Requested methods failed: " + ", ".join(method_errors))
        atomic_write_text(done_flag, "done\n")

        if os.path.exists(fail_flag):
            os.remove(fail_flag)

        return params_summary, metrics_summary, mle_comparison_summary

    except Exception:
        if os.path.exists(done_flag):
            os.remove(done_flag)
        atomic_write_text(fail_flag, traceback.format_exc())
        print(f'[ERROR] Task failed: n_data={n_data}, run={current_run}')
        print(traceback.format_exc())

        return params_summary, metrics_summary, mle_comparison_summary

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
    failed_tasks = []

    for task in shard_tasks:
        params_summary, metrics_summary, mle_comparison_summary = run_one_task(
            config=config,
            task=task,
        )

        params_summary_all.extend(params_summary)
        metrics_summary_all.extend(metrics_summary)
        mle_comparison_summary_all.extend(mle_comparison_summary)
        task_failure = os.path.join(root_dir, f"n_data_{task['n_data']}",
                                    f"run_{task['current_run']}", "_FAILED.txt")
        if os.path.exists(task_failure) or not metrics_summary:
            failed_tasks.append(task)

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

    write_stopping_reports(shard_dir, metrics_summary_all, config["experiment"]["n_runs"])
    if failed_tasks:
        raise RuntimeError(f"{len(failed_tasks)} tasks failed in shard {shard_id}; inspect _FAILED.txt")

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

    with open(os.path.join(root_dir, "experiment_config.json"), encoding="utf-8") as stream:
        saved_config = json.load(stream)
    write_stopping_reports(root_dir,
                           read_csv_records(os.path.join(root_dir, "metrics_summary_merged.csv")),
                           saved_config["experiment"]["n_runs"])

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
            str(CURRENT_FILE),
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

        if getattr(args, "stopping_params", ""):
            cmd += ["--stopping-params", args.stopping_params]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "" if gpu_id.lower() in {"cpu", "none"} else str(gpu_id)

        log_path = os.path.join(logs_dir, f"worker_{worker_id}_gpu_{gpu_id}.log")
        log_file = open(log_path, "w", encoding="utf-8")

        print(f'Starting worker {worker_id} on device {gpu_id}')
        print(f'Log: {log_path}')

        p = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
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
            print(f'[ERROR] worker {worker_id} on device {gpu_id} failed, returncode={ret}')
            print(f'   See log: {log_path}')
        else:
            print(f'[OK] worker {worker_id} on device {gpu_id} completed')

    if len(failed) > 0:
        print('\nThe following workers failed:')
        for worker_id, gpu_id, ret, log_path in failed:
            print(f"worker={worker_id}, gpu={gpu_id}, returncode={ret}, log={log_path}")
        raise RuntimeError('Some workers failed; inspect the logs first.')

    if not args.no_merge_after:
        merge_shard_summaries(root_dir)

    print(f'\n[OK] Experiment completed: {root_dir}')


# ============================================================
# 15. Command-line interface
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stopping-params", default="",
        help='Selected overall-stopping case as JSON: delta_stop, K_stop, epsilon_min, n_sim.',
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="launch",
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
        default=25,
        help='Parallel workers per device (default: 25).',
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
        help='Only Ours and MLE are supported; default: Ours,MLE',
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

    return parser


def main(args):
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


def configure_entrypoint(entrypoint, train_configs):
    """Bind the model's real n10 functions while keeping its module unchanged."""
    global CURRENT_FILE, CURRENT_DIR, USER_HOME, EXPERIMENT_TRAIN_CONFIGS
    global DEMO, EXPERIMENT_MODEL
    CURRENT_FILE = Path(entrypoint).resolve()
    CURRENT_DIR = CURRENT_FILE.parent
    USER_HOME = CURRENT_DIR.parent
    EXPERIMENT_MODEL = CURRENT_DIR.parent.name
    if EXPERIMENT_MODEL not in {"slcp", "glm"}:
        raise ValueError("Use an entry under supplementary/stopping/slcp or supplementary/stopping/glm.")
    DEMO = importlib.import_module(f"demo_{EXPERIMENT_MODEL.upper()}.n10")
    for name in ("np", "torch", "device", "SampleGenerator", "TNet", "prior",
                 "compute_mle", "compute_estimation_metrics",
                 "find_best_initial_theta_via_nle_map", "save_array_to_csv", "ExperimentUtils"):
        globals()[name] = getattr(DEMO, name)
    EXPERIMENT_TRAIN_CONFIGS = deepcopy(train_configs)


def run_experiment(entrypoint, train_configs, arguments=None):
    global AnnealedMinimaxTrainer
    args = build_parser().parse_args(arguments)
    gpu_id = args.gpu_id.lower()
    if gpu_id != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if gpu_id in {"none", "cpu"} else args.gpu_id
    configure_entrypoint(entrypoint, train_configs)
    from .vsm_trainer import AnnealedMinimaxTrainer as trainer_class
    AnnealedMinimaxTrainer = trainer_class
    main(args)


if __name__ == "__main__":
    raise SystemExit("Run a model entry under supplementary/stopping/slcp or supplementary/stopping/glm.")
