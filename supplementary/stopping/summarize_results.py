"""Summarize overall-stopping runs from their saved CSVs and configuration.

The default input is the selected model's paper_stopping_results directory.
The summary contains one row per experiment directory and observation count.
It uses the numeric postprocessor for prior clipping, error metrics and
sample standard deviations. It requires NumPy and pandas.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if not (PROJECT_ROOT / "postprocess_numeric_results.py").is_file():
    raise FileNotFoundError(
        "Upload postprocess_numeric_results.py to the project root: "
        f"{PROJECT_ROOT / 'postprocess_numeric_results.py'}"
    )
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import postprocess_numeric_results as numeric_postprocess
from supplementary.stopping.reporting import REASONS, summarize_stopping_records


ERROR_METRICS = ("param_mae", "param_mse", "param_nrmse")
COMPARISONS = ("ours_vs_true", "ours_vs_mle", "mle_vs_true")
COUNTS = ("theta_convergence_count", "epsilon_min_count", "simulation_budget_count",
          "reason_theta_convergence_count", "reason_epsilon_min_count", "reason_simulation_budget_count")
MLE_RENAME = {
    "compare_method": "method",
    "vs_MLE_estimate_theta_processed": "estimate_theta_processed",
    "vs_MLE_true_theta_processed": "true_theta_processed",
    "vs_MLE_estimate_theta": "estimate_theta",
    "vs_MLE_true_theta": "true_theta",
}
SUMMARY_COLUMNS = [
    "case_name", "experiment_tag", "model", "n_data",
    "delta_stop", "K_stop", "epsilon_min", "n_sim_budget", "n_init_sim",
    "configured_runs", "ours_runs", "mle_runs", "ours_vs_mle_runs", "status", "notes",
    *[f"{comparison}_{metric}_{stat}" for comparison in COMPARISONS
      for metric in ERROR_METRICS for stat in ("mean", "std")],
    "n_sim_mean", "n_sim_std", *COUNTS,
    "ours_clipped_runs", "mle_clipped_runs", "true_clipped_runs",
    "metrics_source", "mle_comparison_source", "experiment_dir",
]


def _discover_experiments(root: Path):
    """Visit only experiment metadata, pruning caches, logs and run payloads."""
    for current, directories, files in os.walk(root):
        if "experiment_config.json" in files or "metrics_summary_merged.csv" in files:
            yield Path(current)
            directories[:] = []
            continue
        directories[:] = sorted(name for name in directories if not (
            name.startswith(("pretrained", "_", "run_"))
            or name in {"sbi-logs", "postprocessed", "__pycache__"}
        ))


def _read_nonempty(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return None
    return frame if not frame.empty else None


def _validate_records(frame: pd.DataFrame, label: str):
    frame = frame.copy()
    for column in ("n_data", "run_id", "method"):
        if column not in frame:
            raise ValueError(f"{label}: missing {column}")
    for column in ("n_data", "run_id"):
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all() or (values <= 0).any() or (values % 1 != 0).any():
            raise ValueError(f"{label}: invalid {column}")
        frame[column] = values.astype(int)
    duplicate = frame.duplicated(["n_data", "run_id", "method"], keep=False)
    if duplicate.any():
        raise ValueError(f"{label}: duplicate (n_data, run_id, method) records")
    return frame


def _read_metrics(directory: Path):
    frame = _read_nonempty(directory / "metrics_summary_merged.csv")
    source = "metrics_summary_merged.csv"
    if frame is None:
        shards = [_read_nonempty(path) for path in sorted(directory.glob("_shard_*/metrics_summary.csv"))]
        shards = [part for part in shards if part is not None]
        if not shards:
            raise FileNotFoundError("No nonempty metrics_summary_merged.csv or shard metrics summaries")
        frame = pd.concat(shards, ignore_index=True)
        source = "_shard_*/metrics_summary.csv"
    return _validate_records(frame, source), source


def _mle_comparison(directory: Path, metrics: pd.DataFrame):
    frame = _read_nonempty(directory / "mle_comparison_summary_merged.csv")
    if frame is not None:
        frame = frame.rename(columns=MLE_RENAME)
        if "baseline_method" in frame:
            frame = frame.loc[frame.baseline_method == "MLE"]
        return _validate_records(frame, "MLE comparison"), "mle_comparison_summary_merged.csv"

    # Rebuild only the input vectors, without implementing any error formula.
    # Processed columns remain preferred by the original numeric postprocessor.
    ours = metrics.loc[metrics.method == "Ours"]
    mle = metrics.loc[metrics.method == "MLE"].set_index(["n_data", "run_id"])
    rows = []
    for _, estimate in ours.iterrows():
        key = (estimate.n_data, estimate.run_id)
        if key not in mle.index:
            continue
        reference = mle.loc[key]
        row = {"n_data": key[0], "run_id": key[1], "method": "Ours", "baseline_method": "MLE"}
        for suffix in ("", "_processed"):
            column = f"estimate_theta{suffix}"
            if column in metrics:
                row[column] = estimate[column]
                row[f"true_theta{suffix}"] = reference[column]
        rows.append(row)
    frame = pd.DataFrame(rows, columns=["n_data", "run_id", "method", "baseline_method",
        "estimate_theta", "true_theta", "estimate_theta_processed", "true_theta_processed"])
    return _validate_records(frame, "paired MLE comparison"), "paired Ours/MLE vectors"


def _recompute(frame: pd.DataFrame, spec):
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for _, source in frame.iterrows():
        try:
            row = numeric_postprocess.compute_one_row_metrics(source, spec)
            if not np.isfinite([row[key] for key in ERROR_METRICS]).all():
                raise ValueError("Nonfinite recomputed parameter error")
            rows.append(row)
        except Exception as exc:
            raise ValueError(f"Invalid {source.get('method')} run {source.get('run_id')}: {exc}") from exc
    return pd.DataFrame(rows)


def _add_error_summary(output, row_metrics, method, prefix, spec):
    selected = row_metrics.loc[row_metrics.method == method] if not row_metrics.empty else row_metrics
    if selected.empty:
        return
    summary = numeric_postprocess.summarize_by_method(selected, spec.theta_dim).iloc[0]
    for metric in ERROR_METRICS:
        output[f"{prefix}_{metric}_mean"] = summary[f"{metric}_mu"]
        output[f"{prefix}_{metric}_std"] = summary[f"{metric}_std"]


def _metadata(directory, config, model):
    experiment = config.get("experiment", {})
    ours = config.get("methods", {}).get("Ours", {})
    sensitivity = config.get("sensitivity", {})
    parameters = dict(ours, **sensitivity.get("parameters", {}))
    return {
        "case_name": sensitivity.get("case_name", directory.parent.name),
        "experiment_tag": experiment.get("experiment_time", directory.name),
        "model": model, "experiment_dir": str(directory.resolve()),
        "delta_stop": parameters.get("delta_stop"), "K_stop": parameters.get("K_stop"),
        "epsilon_min": parameters.get("epsilon_min"), "n_sim_budget": parameters.get("n_sim"),
        "n_init_sim": parameters.get("n_init_sim"), "configured_runs": experiment.get("n_runs"),
        "status": "ok", "notes": "",
    }


def _summarize_group(output, metrics, comparisons, spec, directory):
    ours = metrics.loc[metrics.method == "Ours"].copy()
    mle = metrics.loc[metrics.method == "MLE"]
    paired = comparisons.loc[comparisons.method == "Ours"]
    output.update(ours_runs=len(ours), mle_runs=len(mle), ours_vs_mle_runs=len(paired))
    if ours.empty:
        raise ValueError("No Ours records for this observation count")
    if not set(paired.run_id).issubset(set(ours.run_id) & set(mle.run_id)):
        raise ValueError("MLE comparison has no matching Ours/MLE run")

    true_metrics = _recompute(metrics.loc[metrics.method.isin(["Ours", "MLE"])], spec)
    mle_metrics = _recompute(paired, spec)
    for method, prefix in (("Ours", "ours_vs_true"), ("MLE", "mle_vs_true")):
        _add_error_summary(output, true_metrics, method, prefix, spec)
    _add_error_summary(output, mle_metrics, "Ours", "ours_vs_mle", spec)
    for method, column in (("Ours", "ours_clipped_runs"), ("MLE", "mle_clipped_runs")):
        selected = true_metrics.loc[true_metrics.method == method]
        output[column] = int(selected.estimate_was_clipped.sum())
    output["true_clipped_runs"] = int(true_metrics.loc[true_metrics.method == "Ours", "true_was_clipped"].sum())

    simulations = pd.to_numeric(ours["n_sim"], errors="raise")
    if not np.isfinite(simulations).all() or (simulations < 0).any():
        raise ValueError("Ours n_sim must be finite and nonnegative")
    output["n_sim_mean"] = float(simulations.mean())
    output["n_sim_std"] = float(simulations.std(ddof=1)) if len(simulations) > 1 else 0.0
    required = ("vsm_stop_reason", "vsm_stop_theta", "vsm_stop_epsilon", "vsm_stop_budget")
    for key in required:
        if key not in ours or ours[key].isna().any():
            raise ValueError(f"Missing stopping diagnostic {key}; counts cannot be inferred as zero")
    if not ours.vsm_stop_reason.isin(REASONS).all():
        raise ValueError("Invalid final stopping reason")
    stop_summary = summarize_stopping_records(ours.to_dict("records"))[0]
    output.update({key: stop_summary[key] for key in COUNTS})

    issues = []
    configured = output.get("configured_runs")
    if configured is not None:
        expected = set(range(1, int(configured) + 1))
        for label, frame in (("Ours", ours), ("MLE", mle), ("Ours vs MLE", paired)):
            if set(frame.run_id) != expected:
                issues.append(f"{label}: {len(frame)}/{configured} expected runs or mismatched run IDs")
    elif set(ours.run_id) != set(mle.run_id) or set(ours.run_id) != set(paired.run_id):
        issues.append("Ours/MLE run IDs do not match")
    failures = sum((directory / f"n_data_{output['n_data']}" / f"run_{run_id}" / "_FAILED.txt").is_file()
                   for run_id in ours.run_id)
    if failures:
        issues.append(f"{failures} runs have _FAILED.txt")
    if issues:
        output.update(status="incomplete", notes="; ".join(issues))


def summarize_results(results_root=None, output_csv=None, model="SLCP", exp_tag=None):
    """Return/write the large summary table using the original numeric functions.

    Incomplete or invalid runs appear with a status and any available statistics.
    Parameters come from experiment_config.json in each result directory.
    """
    model = model.upper()
    default_root = SCRIPT_DIR / model.lower() / "overall_stopping" / "paper_stopping_results"
    root = Path(results_root) if results_root is not None else default_root
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Result directory does not exist: {root}")
    destination = Path(output_csv) if output_csv is not None else root / "stopping_summary.csv"
    destination = destination.expanduser().resolve()
    spec = numeric_postprocess.load_model_spec(model)
    rows = []
    for directory in sorted(_discover_experiments(root)):
        config, metadata = {}, {"case_name": directory.parent.name, "experiment_dir": str(directory), "model": model}
        try:
            with (directory / "experiment_config.json").open(encoding="utf-8-sig") as stream:
                config = json.load(stream)
            metadata = _metadata(directory, config, model)
            if exp_tag is not None and str(metadata["experiment_tag"]) != str(exp_tag):
                continue
            actual_model = str(config.get("experiment", {}).get("example_name", model)).upper()
            if actual_model in {"SLCP", "GLM"} and actual_model != model:
                raise ValueError(f"Result model is {actual_model}; rerun with --model {actual_model}")
            metrics, source = _read_metrics(directory)
            comparisons, mle_source = _mle_comparison(directory, metrics)
            metadata.update(metrics_source=source, mle_comparison_source=mle_source)
            configured_sizes = config.get("experiment", {}).get("n_data_list", [])
            sizes = sorted(set(map(int, configured_sizes)) | set(metrics.n_data))
        except Exception as exc:
            # A selected tag must not include broken metadata from another
            # conventional Exp_MODEL_tag directory. Otherwise retain the error.
            prefix = f"Exp_{model}_"
            if (exp_tag is not None and directory.name.startswith(prefix)
                    and directory.name[len(prefix):] != str(exp_tag)):
                continue
            rows.append(dict(metadata, status="error", notes=f"{type(exc).__name__}: {exc}"))
            print(f"[ERROR] {directory}: {exc}")
            continue
        for n_data in sizes:
            output = dict(metadata, n_data=n_data)
            try:
                _summarize_group(output, metrics.loc[metrics.n_data == n_data],
                                 comparisons.loc[comparisons.n_data == n_data], spec, directory)
            except Exception as exc:
                output.update(status="error", notes=f"{type(exc).__name__}: {exc}")
            rows.append(output)
            print(f"[{output['status'].upper()}] {output['case_name']} / {output['experiment_tag']} / n_data={n_data}")
    if not rows:
        raise FileNotFoundError(f"No matching experiment result directories found under {root}")
    summary = pd.DataFrame(rows).reindex(columns=SUMMARY_COLUMNS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination, index=False, encoding="utf-8-sig")
    print(f"\nSummary written: {destination} ({len(summary)} rows)")
    print("Error metrics: postprocess_numeric_results.py; prior-clipped, sample std ddof=1.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, help="Run directory or model results root.")
    parser.add_argument("--output", type=Path, help="CSV path (default: stopping_summary.csv in the results root).")
    parser.add_argument("--model", choices=("SLCP", "GLM"), default="SLCP")
    parser.add_argument("--exp-tag", default=None, help="Optional original experiment tag, e.g. V3")
    args = parser.parse_args()
    try:
        summary = summarize_results(args.root, args.output, args.model, args.exp_tag)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Summary failed: {exc}\n")
    problems = int((summary.status != "ok").sum())
    if problems:
        print(f"{problems} rows are incomplete or invalid; inspect status and notes in the CSV.")
    return int(problems > 0)


if __name__ == "__main__":
    raise SystemExit(main())
