"""Configure the SLCP and MNIST variants using the main experiment drivers."""

import argparse
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output" / "supplementary" / "slcp_mnist"
GROUP_LABELS = {
    "trajectory": "SLCP",
    "mnist_width": "add2",
    "nle_initialization": "add3",
    "critic_capacity": "add4",
    "optimization": "add5",
}
SUMMARY_NAMES = ("params_summary", "metrics_summary", "mle_comparison_summary")
METHODS = ("Ours", "WME_p1", "WME_p2", "AdaIMLE", "ANND", "NBE", "NLE", "NPE_C", "MLE")


def parse_args(is_mnist):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", choices=("launch", "worker", "merge"), default="launch")
    parser.add_argument("--gpus", default="cpu", help="Comma-separated CUDA IDs or cpu, for launch mode.")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--gpu-id", default="auto", help="CUDA ID, cpu, or auto, for worker mode.")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--exp-tag", help="Reuse the same tag when resuming or merging a run.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT,
                        help="Parent folder for the experiment group and setting directories.")
    parser.add_argument("--n-runs", type=int)
    parser.add_argument("--n-data-list", default="")
    parser.add_argument("--methods", default="", help="Comma-separated estimator names.")
    parser.add_argument("--manual-seed-offset", type=int)
    parser.add_argument("--no-skip-finished", action="store_true")
    parser.add_argument("--no-merge-after", action="store_true")
    if is_mnist:
        parser.add_argument("--d", type=int)
        parser.add_argument("--r-dim", type=int)
        parser.add_argument("--data-home", default="")
        parser.add_argument("--train-samples", type=int)
        parser.add_argument("--max-test-samples", type=int)
        parser.add_argument("--n-mc-predict", type=int)
        parser.add_argument("--no-whiten", action="store_true")
    else:
        parser.add_argument("--no-prewarm", action="store_true")
    args = parser.parse_args()

    if is_mnist and any(
        value is not None and value < 1
        for value in (args.d, args.r_dim, args.train_samples, args.n_mc_predict)
    ):
        parser.error("feature dimension, hidden width, sample count, and MC count must be positive")
    if args.gpu_id.lower() not in {"auto", "cpu", "none"} and not args.gpu_id.isdecimal():
        parser.error("--gpu-id must be a CUDA device index, cpu, or auto")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("require 0 <= --shard-id < --num-shards")
    if args.workers_per_gpu < 1 or (args.n_runs is not None and args.n_runs < 1):
        parser.error("worker and repetition counts must be positive")
    if args.exp_tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.exp_tag):
        parser.error("--exp-tag must contain only letters, digits, hyphens, or underscores")
    if not args.exp_tag and (args.mode == "merge" or args.num_shards > 1):
        parser.error("a shared --exp-tag is required for merging or manual sharding")
    if args.n_data_list:
        try:
            sizes = [int(n.strip()) for n in args.n_data_list.split(",")]
            if min(sizes) < 1 or len(sizes) != len(set(sizes)):
                raise ValueError
        except ValueError:
            parser.error("--n-data-list must contain distinct, positive integers")
        if is_mnist:
            if len(sizes) != 1 or args.train_samples not in (None, sizes[0]):
                parser.error("MNIST uses one training sample count per experiment")
            args.train_samples = sizes[0]
    if args.methods:
        names = [name.strip() for name in args.methods.split(",")]
        allowed = METHODS[:-1] if is_mnist else METHODS
        if any(name not in allowed for name in names) or len(names) != len(set(names)):
            parser.error(f"--methods must select distinct names from {', '.join(allowed)}")
    args.output_root = args.output_root.expanduser().resolve()
    return args


def root_path(output_root, variant, group, exp_tag):
    root = output_root / group
    if group != "trajectory":
        root /= f"{variant.stem}_results"
    return root / f"Exp_{GROUP_LABELS[group]}_{exp_tag}"


def get_demo(is_mnist, gpu_id):
    gpu = gpu_id.lower()
    if gpu != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if gpu in {"cpu", "none"} else gpu_id
    module = "demo_MNIST.digits_1_vs_2" if is_mnist else "demo_SLCP.n10"
    demo = importlib.import_module(module)
    if gpu not in {"auto", "cpu", "none"} and demo.device.type != "cuda":
        raise RuntimeError(f"CUDA device {gpu_id} is unavailable. Use --gpu-id cpu for CPU execution.")
    return demo


def configure(demo, args, root, group, settings):
    """Use the paper protocol and apply just the selected sensitivity settings."""
    config = demo.build_experiment_config()
    config["experiment"]["example_name"] = GROUP_LABELS[group]
    config["paths"].update(
        root_experiment_dir=str(root.parent),
        pretrained_root=str(root),
        root_dir=str(root),
    )
    for method in ("NLE", "NBE", "NPE_C"):
        old_dir = Path(config["methods"][method]["model_dir"])
        config["methods"][method]["model_dir"] = str(root / old_dir.name)

    if group == "mnist_width":
        config["experiment"]["methods_to_run"] = ["Ours"]
        config["mnist"]["r_dim"] = settings["width"]
        config["mnist"]["data_home"] = str(PROJECT_ROOT / "output" / "mnist_openml_cache")
        if args.data_home:
            args.data_home = str(Path(args.data_home).expanduser().resolve())
        config = demo.apply_cli_overrides(config, args)
        d, x_dim, theta_dim, z_dim = demo.compute_mnist_mixed_logit_dims(
            config["mnist"]["d"], config["mnist"]["r_dim"]
        )
        config["dimensions"].update(
            feature_dim=d, x_dim=x_dim, theta_dim=theta_dim, z_dim=z_dim,
            theta_header=demo.make_theta_header(theta_dim),
        )
        config["methods"]["ANND"].update(bounds_lb=[-3.0] * theta_dim,
                                         bounds_ub=[3.0] * theta_dim)
        config["methods"]["NBE"]["theta_dim"] = theta_dim
        return config

    if group == "trajectory":
        config["experiment"]["methods_to_run"].remove("NPE_C")
    else:
        config["experiment"]["methods_to_run"] = ["Ours", "MLE"]
    if group == "nle_initialization":
        n_train, candidates = settings["nle"]
        config["init"].update(n_train=n_train, n_candidates=candidates)
        for method in ("Ours", "WME", "ANND", "AdaIMLE"):
            config["methods"][method]["n_init_sim"] = n_train
    if group == "critic_capacity":
        hidden, blocks = settings["critic"]
        config["critic"] = {"hidden": hidden, "num_blocks": blocks}
    if group == "optimization":
        config["methods"]["Ours"].update(settings)
    init_dir = root / "pretrained_nle_init"
    label = GROUP_LABELS[group].lower()
    config["init"].update(model_path=str(init_dir / f"{label}_nle_likelihood.pt"),
                          candidate_path=str(init_dir / f"{label}_nle_candidates.pt"))
    return demo.apply_cli_overrides(config, args)


def atomic_json(path, payload):
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def config_identity(config):
    experiment = {key: value for key, value in config["experiment"].items() if key != "device"}
    return {
        "experiment": experiment,
        "dimensions": config["dimensions"],
        "mnist": config.get("mnist"),
        "critic": config.get("critic"),
        "methods": config["methods"],
        "init": config["init"],
        "num_shards": config["parallel"]["num_shards"],
    }


def ensure_config(demo, config, root):
    path = root / "experiment_config.json"
    current = demo.make_jsonable(config)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if config_identity(saved) != config_identity(current):
            raise ValueError("The existing experiment configuration differs. Use a new --exp-tag.")
    elif config["parallel"]["shard_id"] == 0:
        atomic_json(path, current)


def run_dir_for(root, task, is_mnist):
    size, run = task["n_data"], task["current_run"]
    data_dir = f"train_{size}_test_full" if is_mnist else f"n_data_{size}"
    return root / data_dir / f"run_{run}"


def read_run(run_dir, expected_methods, is_mnist):
    if not (run_dir / "_DONE.txt").is_file():
        raise FileNotFoundError(f"Incomplete run: {run_dir}")
    records = json.loads((run_dir / "run_records.json").read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != (2 if is_mnist else 3):
        raise ValueError(f"Incomplete summary records: {run_dir}")
    for rows in records[:2]:
        if sorted(row["method"] for row in rows) != sorted(expected_methods):
            raise ValueError(f"Incomplete method results: {run_dir}")
    if not is_mnist:
        comparisons = sorted(set(expected_methods) - {"MLE"}) if "MLE" in expected_methods else []
        if sorted(row["compare_method"] for row in records[2]) != comparisons:
            raise ValueError(f"Incomplete MLE comparisons: {run_dir}")
    return records


def run_one(demo, config, task, is_mnist, skip_finished):
    run_dir = run_dir_for(Path(config["paths"]["root_dir"]), task, is_mnist)
    expected_methods = config["experiment"]["methods_to_run"]
    if skip_finished:
        try:
            records = read_run(run_dir, expected_methods, is_mnist)
        except (OSError, ValueError, TypeError, KeyError):
            pass
        else:
            print(f"Reusing completed run: {run_dir}")
            return records

    done = run_dir / "_DONE.txt"
    done.unlink(missing_ok=True)
    (run_dir / "run_records.json").unlink(missing_ok=True)
    records = demo.run_one_task(config, task)
    produced_methods = {row["method"] for row in records[1]}
    missing = set(expected_methods) - produced_methods
    if missing:
        done.unlink(missing_ok=True)
        raise RuntimeError(f"Run {run_dir} failed for {', '.join(sorted(missing))}. See error logs.")
    atomic_json(run_dir / "run_records.json", demo.make_jsonable(list(records)))
    for method in expected_methods:
        (run_dir / f"{method}_ERROR.txt").unlink(missing_ok=True)
    return records


def write_records(demo, destination, records):
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    try:
        demo.save_records_to_csv(records, str(temporary))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_summaries(demo, root, groups, is_mnist, *, merged=False):
    names = SUMMARY_NAMES[:2] if is_mnist else SUMMARY_NAMES
    for name, rows in zip(names, groups):
        suffix = "_merged" if merged else ""
        if merged and name == "metrics_summary":
            # The helper searches relative to the parent of the supplied directory.
            # Limit it to this run tag, where its own checkpoint is stored.
            rows = demo.apply_nbe_training_cost_once(rows, root / "_shard_0")
        write_records(demo, root / f"{name}{suffix}.csv", rows)


def wait_for_mnist_assets(root):
    deadline = time.monotonic() + 6 * 3600
    ready = root / "_SHARED_ASSETS_READY.txt"
    failure = root / "_SHARED_ASSETS_FAILED.txt"
    while not ready.is_file():
        if failure.is_file():
            raise RuntimeError(f"Shard 0 could not complete its first MNIST task: {failure}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for shard 0 to prepare MNIST assets: {root}")
        time.sleep(2)


def worker(demo, config, root, is_mnist):
    root.mkdir(parents=True, exist_ok=True)
    shard = config["parallel"]["shard_id"]
    skip_finished = config["parallel"]["skip_finished"]
    ensure_config(demo, config, root)
    demo.ExperimentUtils.set_global_seed(config["experiment"]["global_seed"])
    _, tasks = demo.get_shard_tasks(config)
    shard_dir = root / f"_shard_{shard}"
    shard_dir.mkdir(exist_ok=True)
    summaries = [[], []] if is_mnist else [[], [], []]
    failed = []
    ready = root / "_SHARED_ASSETS_READY.txt"
    failure = root / "_SHARED_ASSETS_FAILED.txt"
    if shard == 0:
        failure.unlink(missing_ok=True)
        if not skip_finished:
            for name in SUMMARY_NAMES:
                (root / f"{name}_merged.csv").unlink(missing_ok=True)
        if not is_mnist:
            demo.prepare_shared_assets(config)
    elif is_mnist:
        wait_for_mnist_assets(root)
    else:
        demo.wait_for_shared_assets(config)
    ensure_config(demo, config, root)

    for index, task in enumerate(tasks):
        try:
            for summary, rows in zip(summaries, run_one(demo, config, task, is_mnist, skip_finished)):
                summary.extend(rows)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            if is_mnist and shard == 0 and index == 0 and not ready.exists():
                failure.write_text(str(exc), encoding="utf-8")
                raise
            failed.append(task["current_run"])
        else:
            if is_mnist and shard == 0 and index == 0:
                ready.touch()
        save_summaries(demo, shard_dir, summaries, is_mnist)
    save_summaries(demo, shard_dir, summaries, is_mnist)
    print(f"Results for shard {shard}: {root}")
    if failed:
        raise RuntimeError(f"{len(failed)} tasks failed in shard {shard}. Retry with the same tag.")


def merge(demo, root, is_mnist):
    config = json.loads((root / "experiment_config.json").read_text(encoding="utf-8"))
    summaries = [[], []] if is_mnist else [[], [], []]
    for task in demo.build_run_tasks(config):
        run_dir = run_dir_for(root, task, is_mnist)
        records = read_run(run_dir, config["experiment"]["methods_to_run"], is_mnist)
        for summary, rows in zip(summaries, records):
            summary.extend(rows)
    save_summaries(demo, root, summaries, is_mnist, merged=True)
    print(f"Merged results: {root}")


def worker_options(args):
    process_options = {
        "mode", "gpus", "workers_per_gpu", "gpu_id", "shard_id",
        "num_shards", "exp_tag", "no_merge_after",
    }
    options = []
    for name, value in vars(args).items():
        if name in process_options or value is None or value is False or value == "":
            continue
        flag = "--" + name.replace("_", "-")
        if value is True:
            options.append(flag)
        else:
            options.extend((flag, str(value)))
    return options


def launch(args, variant, root, is_mnist):
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must list at least one device")
    for gpu in gpu_ids:
        if gpu.lower() not in {"auto", "cpu", "none"} and not gpu.isdecimal():
            raise ValueError(f"Invalid GPU ID: {gpu}")
    count = len(gpu_ids) * args.workers_per_gpu
    logs = root / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    (root / "_SHARED_ASSETS_FAILED.txt").unlink(missing_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    options = worker_options(args)
    processes = []
    try:
        for shard in range(count):
            gpu = gpu_ids[shard % len(gpu_ids)]
            log = logs / f"worker_{shard}_gpu_{gpu}.log"
            command = [
                sys.executable, "-u", str(variant), *options,
                "--mode", "worker", "--gpu-id", gpu, "--num-shards", str(count),
                "--shard-id", str(shard), "--exp-tag", args.exp_tag,
            ]
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            processes.append((process, log))
            print(f"Started worker {shard} on {gpu}. Log: {log}", flush=True)
        for process, log in processes:
            if process.wait() != 0:
                raise RuntimeError(f"Worker failed. See {log}")
    finally:
        for process, _ in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
    if not args.no_merge_after:
        merge(get_demo(is_mnist, "cpu"), root, is_mnist)


def run(variant, group, **settings):
    """Entry point shared by the individual supplementary settings."""
    variant = Path(variant).resolve()
    is_mnist = group == "mnist_width"
    args = parse_args(is_mnist)
    args.exp_tag = args.exp_tag or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = root_path(args.output_root, variant, group, args.exp_tag)
    if args.mode == "launch":
        launch(args, variant, root, is_mnist)
    elif args.mode == "merge":
        merge(get_demo(is_mnist, "cpu"), root, is_mnist)
    else:
        demo = get_demo(is_mnist, args.gpu_id)
        worker(demo, configure(demo, args, root, group, settings), root, is_mnist)
