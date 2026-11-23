from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# Project discovery
# =============================================================================

def find_project_root(start: Path) -> Path:
    """Find the vsm_supplementary root from the script location."""
    start = start.resolve()
    candidates = [start] + list(start.parents)
    for candidate in candidates:
        if (candidate / "demo_MNIST").is_dir() and (candidate / "models").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate the vsm_supplementary project root. "
        "Place this script in the project root (recommended) or a subdirectory of it."
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)
MNIST_ROOT = PROJECT_ROOT / "demo_MNIST"


# =============================================================================
# Result discovery
# =============================================================================

def infer_task(path: Path) -> str:
    """Infer the MNIST binary task from the current result-directory names."""
    text = "/".join(p.lower() for p in path.parts)
    if "digits_1_vs_2" in text:
        return "1_vs_2"
    if "digits_4_vs_9" in text:
        return "4_vs_9"
    return "unknown"


def discover_mnist_experiments(search_root: Path) -> List[Tuple[Path, str]]:
    found: List[Tuple[Path, str]] = []

    for csv_path in search_root.rglob("metrics_summary_merged.csv"):
        lower_parts = [p.lower() for p in csv_path.parts]
        if "postprocessed" in lower_parts:
            continue
        found.append((csv_path.parent, infer_task(csv_path)))

    unique = {(str(d.resolve()), task): (d, task) for d, task in found}
    return sorted(unique.values(), key=lambda x: (x[1], str(x[0]).lower()))


# =============================================================================
# Summary calculation
# =============================================================================

METRIC_COLUMNS = [
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
    "train_mean_probability",
    "test_mean_probability",
]


def summarize_mnist_csv(input_csv: Path) -> pd.DataFrame:
    """Summarize one merged MNIST result CSV by method using mean and sample std."""
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)
    if input_csv.stat().st_size == 0:
        raise ValueError(f"Input CSV is empty: {input_csv}")

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    if "method" not in df.columns:
        raise ValueError(f"Missing 'method' column in {input_csv}")

    metric_cols = [c for c in METRIC_COLUMNS if c in df.columns]
    if not metric_cols:
        raise ValueError(
            f"None of the expected MNIST metric columns were found in {input_csv}."
        )

    summary_rows: List[Dict[str, object]] = []

    for method, group in df.groupby("method", dropna=False):
        result: Dict[str, object] = {
            "method": method,
            "num_runs": int(len(group)),
        }

        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            result[f"{col}_mu"] = float(values.mean())
            std_val = values.std(ddof=1)
            result[f"{col}_std"] = 0.0 if pd.isna(std_val) else float(std_val)

        summary_rows.append(result)

    summary_df = pd.DataFrame(summary_rows)

    # Put the highest mean test accuracy first, matching the old post-processing script.
    if not summary_df.empty and "test_accuracy_mu" in summary_df.columns:
        summary_df = summary_df.sort_values(
            by="test_accuracy_mu",
            ascending=False,
        )

    return summary_df.reset_index(drop=True)


def core_test_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return the paper-facing test metrics while preserving method and run count."""
    preferred = [
        "method",
        "num_runs",
        "test_accuracy_mu",
        "test_accuracy_std",
        "test_f1_mu",
        "test_f1_std",
        "test_auc_mu",
        "test_auc_std",
        "test_nll_mu",
        "test_nll_std",
        "time_used_mu",
        "time_used_std",
        "n_sim_mu",
        "n_sim_std",
    ]
    cols = [c for c in preferred if c in summary_df.columns]
    return summary_df[cols].copy()


# =============================================================================
# Main driver
# =============================================================================

def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def run(search_root: Path) -> None:
    experiments = discover_mnist_experiments(search_root)

    print("=" * 88)
    print("MNIST-result post-processing")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Search root  : {search_root}")
    print(f"Experiments  : {len(experiments)}")
    print("=" * 88)

    if not experiments:
        print("No MNIST metrics_summary_merged.csv files were found.")
        print("Run the MNIST experiments first, then run this script again.")
        return

    all_summary_frames: List[pd.DataFrame] = []
    index_rows: List[Dict[str, object]] = []

    for experiment_dir, task in experiments:
        input_csv = experiment_dir / "metrics_summary_merged.csv"
        output_dir = experiment_dir / "postprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "-" * 88)
        print(f"Task       : {task}")
        print(f"Experiment : {relative_to_project(experiment_dir)}")

        try:
            summary_df = summarize_mnist_csv(input_csv)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            index_rows.append(
                {
                    "task": task,
                    "experiment_dir": relative_to_project(experiment_dir),
                    "input_csv": relative_to_project(input_csv),
                    "status": "failed",
                    "error": repr(exc),
                }
            )
            continue

        full_output = output_dir / "metrics_summary_by_method.csv"
        core_output = output_dir / "test_metrics_summary_by_method.csv"

        summary_df.to_csv(full_output, index=False, encoding="utf-8-sig")
        core_test_summary(summary_df).to_csv(
            core_output,
            index=False,
            encoding="utf-8-sig",
        )

        tagged = summary_df.copy()
        tagged.insert(0, "task", task)
        tagged.insert(1, "experiment_dir", relative_to_project(experiment_dir))
        all_summary_frames.append(tagged)

        run_counts = summary_df[["method", "num_runs"]].to_dict("records")
        print(f"  Run counts : {run_counts}")
        print(f"  Full output: {relative_to_project(full_output)}")
        print(f"  Core output: {relative_to_project(core_output)}")

        index_rows.append(
            {
                "task": task,
                "experiment_dir": relative_to_project(experiment_dir),
                "input_csv": relative_to_project(input_csv),
                "status": "ok",
                "error": "",
            }
        )

    if all_summary_frames:
        combined_df = pd.concat(all_summary_frames, ignore_index=True, sort=False)
        combined_path = PROJECT_ROOT / "postprocessed_mnist_summary_all.csv"
        combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"\nCombined MNIST summary: {combined_path}")

    index_df = pd.DataFrame(index_rows)
    index_path = PROJECT_ROOT / "postprocess_mnist_index.csv"
    index_df.to_csv(index_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 88)
    print("MNIST post-processing completed.")
    print(f"Index written to: {index_path}")
    print("=" * 88)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically summarize merged MNIST experiment results in the "
            "current vsm_supplementary repository."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(MNIST_ROOT),
        help="Directory to scan recursively. Default: demo_MNIST.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scan_root = Path(args.root).expanduser().resolve()
    if not scan_root.exists():
        sys.exit(f"Search root does not exist: {scan_root}")
    run(scan_root)
