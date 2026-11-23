"""Summarize the additional SLCP and MNIST experiments from merged CSVs."""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import postprocess_mnist_results as mnist_post
import postprocess_numeric_results as numeric_post


DEFAULT_ROOT = PROJECT_ROOT / "output" / "supplementary" / "slcp_mnist"
LEGACY_GROUPS = {
    "result_1": "trajectory",
    "result_2": "mnist_width",
    "result_3": "nle_initialization",
    "result_4": "critic_capacity",
    "result_5": "optimization",
}
GROUP_NAMES = set(LEGACY_GROUPS.values())
MLE_RENAME = {
    "compare_method": "method",
    "vs_MLE_estimate_theta_processed": "estimate_theta_processed",
    "vs_MLE_true_theta_processed": "true_theta_processed",
    "vs_MLE_estimate_theta": "estimate_theta",
    "vs_MLE_true_theta": "true_theta",
}
IDENTITY_COLUMNS = [
    "result_group", "setting", "experiment", "model", "comparison", "n_data",
    "method", "num_runs", "configured_runs", "run_count_matches_config",
]
DETAIL_FILES = {
    "metrics_summary_by_method.csv", "test_metrics_summary_by_method.csv",
    "clipped_row_metrics.csv", "clipped_summary_by_method.csv",
    "clipped_core_summary_by_method.csv", "failed_rows.csv",
}


def natural_key(path: Path) -> list:
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", path.as_posix())]


def is_source(path: Path, group: Path) -> bool:
    return not any(part.lower().startswith(("_shard", "postprocessed"))
                   for part in path.relative_to(group).parts)


def discover_experiments(group: Path) -> list[Path]:
    # Configs and shards identify experiments with missing merged CSVs.
    found = {p.parent for p in group.rglob("metrics_summary_merged.csv")
             if is_source(p, group)}
    found.update(p.parent for p in group.rglob("experiment_config.json")
                 if is_source(p, group))
    found.update(p.parent.parent for p in group.rglob("metrics_summary.csv")
                 if p.parent.name.startswith("_shard")
                 and is_source(p.parent.parent, group))
    return sorted(found, key=natural_key)


def missing_setting_records(group: Path, experiments: list[Path]) -> list[dict]:
    """Record settings that have yet to produce a result directory."""
    source_dir = SCRIPT_DIR / LEGACY_GROUPS.get(group.name, group.name)
    if source_dir.name == "trajectory":
        return []
    expected = {p.stem for p in source_dir.glob("*.py")
                if p.name != "main.py" and not p.name.startswith("_")}
    present = {setting_name(p, group) for p in experiments}
    return [{"result_group": group.name, "setting": setting, "status": "warning",
             "message": "No result directory for experiment entry: "
                        + (source_dir / f"{setting}.py").as_posix()}
            for setting in sorted(expected - present)]


def setting_name(experiment: Path, group: Path) -> str:
    if LEGACY_GROUPS.get(group.name, group.name) == "trajectory":
        return "trajectory"
    relative = experiment.relative_to(group)
    return relative.parts[0].removesuffix("_results") if relative.parts else experiment.name


def read_config(experiment: Path) -> dict:
    path = experiment / "experiment_config.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def identify_model(group: Path, experiment: Path, config: dict) -> str:
    group_name = LEGACY_GROUPS.get(group.name, group.name)
    if ("mnist" in config or group_name == "mnist_width"
            or mnist_post.infer_task(experiment) != "unknown"):
        return "MNIST"
    model = numeric_post.infer_model_name(experiment)
    if model:
        return model
    example = config.get("experiment", {}).get("example_name")
    if example == "add2":
        return "MNIST"
    if example in {"add3", "add4", "add5"} or group_name in GROUP_NAMES:
        return "SLCP"
    raise ValueError("Cannot identify model; no known experiment name/configuration")


def setting_metadata(group: Path, experiment: Path, config: dict, model: str) -> dict:
    relative = experiment.relative_to(group)
    setting = setting_name(experiment, group)
    exp = config.get("experiment", {})
    dims = config.get("dimensions", {})
    mnist = config.get("mnist", {})
    init = config.get("init", {})
    ours = config.get("methods", {}).get("Ours", {})
    notes = [] if config else ["experiment_config.json missing; unknown values left blank"]
    metadata = {
        "result_group": group.name,
        "setting": setting,
        "experiment": relative.as_posix(),
        "model": model,
        "configured_runs": exp.get("n_runs"),
        "feature_dim": mnist.get("d", dims.get("feature_dim")),
        "hidden_dim": mnist.get("r_dim"),
        "theta_dim": dims.get("theta_dim"),
        "nle_train": init.get("n_train"),
        "prior_candidates": init.get("n_candidates"),
        "ours_n_sim_budget": ours.get("n_sim"),
        "ours_n_init_sim": ours.get("n_init_sim"),
        "source_config": (experiment / "experiment_config.json").as_posix() if config else "",
    }
    for key in ("gamma", "minimax_iters", "lr_theta", "lr_psi", "m1", "m2",
                "k_inner", "lambda_reg", "early_stop_patience", "early_stop_tol", "yasuo"):
        metadata[f"ours_{key}"] = ours.get(key)
    if ours and "k_inner" not in ours:
        metadata["ours_k_inner"] = 1
        notes.append("ours_k_inner=1: estimators/vsm.py default, not saved in config")
    critic = re.fullmatch(r"critic_hidden_(\d+)_blocks_(\d+)", setting)
    if critic:
        metadata["critic_hidden"] = int(critic[1])
        metadata["critic_blocks"] = int(critic[2])
        notes.append("critic capacity: result folder name (not saved in config)")
    metadata["metadata_notes"] = "; ".join(notes)
    return metadata


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write in the destination directory for atomic replacement.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def clear_generated_details(output_dir: Path) -> None:
    """Clear generated detail files before recomputing them."""
    boundary = output_dir.resolve()
    directories = [output_dir, *output_dir.glob("n_data_*")]
    for directory in directories:
        if not directory.is_dir():
            continue
        for name in DETAIL_FILES:
            path = directory / name
            if path.exists():
                if not path.resolve().is_relative_to(boundary):
                    raise ValueError(f"Output path escapes postprocessing directory: {path}")
                path.unlink()


def ordered_summary(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=IDENTITY_COLUMNS)
    result = pd.concat(frames, ignore_index=True, sort=False)
    first = [c for c in IDENTITY_COLUMNS if c in result]
    return result[first + [c for c in result if c not in first]]


def process_comparison(input_csv: Path, output_dir: Path, metadata: dict,
                       config: dict, comparison: str) -> tuple[list[pd.DataFrame], list[dict]]:
    """Run the shared metric code separately for each sample size."""
    clear_generated_details(output_dir)
    raw = pd.read_csv(input_csv, encoding="utf-8-sig")
    method_column = "compare_method" if comparison == "VSMLE" else "method"
    if raw.empty:
        raise ValueError(f"CSV has no data rows: {input_csv.name}")
    if method_column not in raw or raw[method_column].isna().any():
        raise ValueError(f"Missing/empty method column: {method_column}")
    if "run_id" not in raw or raw["run_id"].isna().any():
        raise ValueError("Missing/empty run_id column")
    size_column = "train_samples" if metadata["model"] == "MNIST" else "n_data"
    if size_column not in raw or raw[size_column].isna().any():
        raise ValueError(f"Missing/empty sample-size column: {size_column}")
    keys = [size_column, method_column, "run_id"]
    if raw.duplicated(keys).any():
        raise ValueError(f"Duplicate rows for {keys}; refusing to double-count runs")

    frames, index = [], []
    batches = list(raw.groupby(size_column, dropna=False, sort=True))
    for size, batch in batches:
        n_data = int(size)
        if n_data != size or n_data <= 0:
            raise ValueError(f"Invalid sample size: {size}")
        destination = output_dir if len(batches) == 1 else output_dir / f"n_data_{n_data}"
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="vsm_postprocess_") as temporary:
            # The shared metric functions accept a separate file for each sample size.
            source = input_csv
            if len(batches) > 1:
                source = Path(temporary) / input_csv.name
                batch.to_csv(source, index=False, encoding="utf-8-sig")
            if metadata["model"] == "MNIST":
                summary = mnist_post.summarize_mnist_csv(source)
                write_csv(summary, destination / "metrics_summary_by_method.csv")
                write_csv(mnist_post.core_test_summary(summary),
                          destination / "test_metrics_summary_by_method.csv")
                rows, failed = len(batch), 0
            else:
                rows, failed, summary = numeric_post.process_one_csv(
                    source, destination, numeric_post.load_model_spec(metadata["model"]),
                    rename_columns=MLE_RENAME if comparison == "VSMLE" else None,
                )

        messages = []
        if failed:
            messages.append(f"{failed} invalid parameter rows; see failed_rows.csv")
        expected = config.get("experiment", {}).get("n_runs")
        if summary is not None and not summary.empty:
            tagged = summary.copy()
            for key, value in metadata.items():
                tagged[key] = value
            tagged["comparison"] = comparison
            tagged["n_data"] = n_data
            tagged["source_csv"] = input_csv.as_posix()
            tagged["run_count_matches_config"] = (
                tagged["num_runs"].eq(expected) if expected is not None else pd.NA
            )
            if expected is not None and not tagged["num_runs"].eq(expected).all():
                messages.append(f"Actual valid run count differs from configured n_runs={expected}")
            planned = set(config.get("experiment", {}).get("methods_to_run", []))
            if comparison == "VSMLE":
                planned.discard("MLE")
            missing = planned - set(summary["method"])
            if missing:
                messages.append("Missing configured methods: " + ", ".join(sorted(missing)))
            frames.append(tagged)
        else:
            messages.append("No valid summary rows")
        index.append({
            **{k: metadata[k] for k in ("result_group", "setting", "experiment", "model")},
            "comparison": comparison, "n_data": n_data, "input_csv": input_csv.as_posix(),
            "output_dir": destination.as_posix(), "rows": rows, "failed_rows": failed,
            "status": "failed" if failed or summary is None else ("warning" if messages else "ok"),
            "message": "; ".join(messages),
        })
    return frames, index


def run(root: Path) -> int:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Result directory does not exist: {root}")
    group_names = GROUP_NAMES | LEGACY_GROUPS.keys()
    groups = [root] if root.name in group_names else sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name in group_names),
        key=natural_key,
    )
    if not groups:
        raise ValueError(f"No SLCP or MNIST experiment groups found in {root}")
    all_frames, all_index = [], []
    for group in groups:
        frames, index = [], []
        experiments = discover_experiments(group)
        missing = missing_setting_records(group, experiments)
        index.extend(missing)
        print(f"[{group.name}] {len(experiments)} experiment directories")
        for record in missing:
            print(f"  [WARNING] {record['setting']}: no result directory")
        for experiment in experiments:
            identity = {"result_group": group.name,
                        "experiment": experiment.relative_to(group).as_posix()}
            try:
                config = read_config(experiment)
                model = identify_model(group, experiment, config)
                metadata = setting_metadata(group, experiment, config, model)
            except Exception as exc:
                for suffix in ("", "VSTRUE", "VSMLE"):
                    clear_generated_details(experiment / "postprocessed" / suffix)
                index.append({**identity, "status": "failed", "message": str(exc)})
                continue
            comparisons = [("TEST" if model == "MNIST" else "VSTRUE",
                            experiment / "metrics_summary_merged.csv")]
            if model != "MNIST":
                mle = experiment / "mle_comparison_summary_merged.csv"
                if mle.exists() and mle.stat().st_size > 3:
                    comparisons.append(("VSMLE", mle))
                else:
                    clear_generated_details(experiment / "postprocessed" / "VSMLE")
                    index.append({**identity, "comparison": "VSMLE", "status": "skipped",
                                  "message": "No non-empty MLE comparison CSV"})
            for comparison, source in comparisons:
                destination = experiment / "postprocessed"
                if model != "MNIST":
                    destination /= comparison
                try:
                    summaries, records = process_comparison(
                        source, destination, metadata, config, comparison,
                    )
                    frames.extend(summaries)
                    index.extend(records)
                    counts = [f"{r['n_data']}: {r['rows']} rows, {r['failed_rows']} failed"
                              for r in records]
                    print(f"  {metadata['setting']} / {experiment.name} / {comparison}: "
                          + "; ".join(counts))
                except Exception as exc:
                    index.append({**identity, "comparison": comparison,
                                  "input_csv": source.as_posix(), "status": "failed",
                                  "message": str(exc)})
                    print(f"  [ERROR] {identity['experiment']} / {comparison}: {exc}")
        if not experiments:
            index.append({"result_group": group.name, "status": "skipped",
                          "message": "No experiments found (empty supplementary experiment)"})
        summary = ordered_summary(frames)
        write_csv(summary, group / f"{group.name}_summary.csv")
        write_csv(pd.DataFrame(index), group / "postprocess_index.csv")
        all_frames.extend(frames)
        all_index.extend(index)
        print(f"  Saved {group.name}_summary.csv: {len(summary)} summary rows")
    # A single-group invocation updates that group's tables only.
    if groups != [root]:
        write_csv(ordered_summary(all_frames), root / "all_results_summary.csv")
        write_csv(pd.DataFrame(all_index), root / "postprocess_index.csv")
    failures = sum(row["status"] == "failed" for row in all_index)
    warnings = sum(row["status"] == "warning" for row in all_index)
    print(f"Completed: {len(groups)} groups; {failures} failed items; {warnings} warnings.")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="A results root or a single experiment group. Default: %(default)s")
    args = parser.parse_args()
    try:
        return run(args.root)
    except (OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
