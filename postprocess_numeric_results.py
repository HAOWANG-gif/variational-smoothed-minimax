from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Project discovery
# =============================================================================

def find_project_root(start: Path) -> Path:
    """Find the vsm_supplementary root from the script location."""
    start = start.resolve()
    candidates = [start] + list(start.parents)
    for candidate in candidates:
        if (candidate / "models").is_dir() and (candidate / "demo_SLCP").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate the vsm_supplementary project root. "
        "Place this script in the project root (recommended) or a subdirectory of it."
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)


# =============================================================================
# Model metadata read directly from source files
# =============================================================================

MODEL_FILES: Dict[str, Path] = {
    "SLCP": PROJECT_ROOT / "models" / "slcp.py",
    "GLM": PROJECT_ROOT / "models" / "glm.py",
    "gk": PROJECT_ROOT / "models" / "gk.py",
    "BGK": PROJECT_ROOT / "models" / "bgk.py",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    theta_dim: int
    lower: np.ndarray
    upper: np.ndarray


def _call_name(node: ast.AST) -> str:
    """Return a dotted name for a Call/Attribute/Name node when possible."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _call_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def _number(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_number(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _number(node.operand)
    raise ValueError(f"Unsupported numeric AST node: {ast.dump(node, include_attributes=False)}")


def _literal_vector(node: ast.AST) -> np.ndarray:
    if isinstance(node, (ast.List, ast.Tuple)):
        return np.asarray([_number(x) for x in node.elts], dtype=float)
    raise ValueError("Expected a literal list/tuple of numeric values.")


def _extract_theta_dim(tree: ast.AST, source_file: Path) -> int:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "theta_dim":
                    if value is None:
                        break
                    dim = int(_number(value))
                    if dim <= 0:
                        raise ValueError(f"Invalid theta_dim={dim} in {source_file}")
                    return dim
    raise ValueError(f"Could not find theta_dim in {source_file}")


def _find_assignment_to_attribute(tree: ast.AST, attr_name: str) -> Optional[ast.AST]:
    """Find an assignment such as cls._prior_low = ... anywhere in the module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == attr_name:
                    return node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Attribute) and target.attr == attr_name:
                return node.value
    return None


def _evaluate_vector_expr(node: ast.AST, tree: ast.AST, theta_dim: int) -> np.ndarray:
    """Evaluate the simple vector expressions used for priors in models/*.py."""
    if isinstance(node, (ast.List, ast.Tuple)):
        return _literal_vector(node)

    if isinstance(node, ast.Call):
        name = _call_name(node.func)

        # torch.tensor([..])
        if name.endswith("tensor"):
            if not node.args:
                raise ValueError("torch.tensor call has no positional data argument.")
            return _literal_vector(node.args[0])

        # torch.full((theta_dim,), fill_value, ...)
        if name.endswith("full"):
            if len(node.args) < 2:
                raise ValueError("torch.full call is missing shape/fill arguments.")
            fill_value = _number(node.args[1])
            return np.full(theta_dim, fill_value, dtype=float)

    # GLM uses low=SampleGenerator._prior_low and high=SampleGenerator._prior_high.
    if isinstance(node, ast.Attribute) and node.attr in {"_prior_low", "_prior_high"}:
        assigned = _find_assignment_to_attribute(tree, node.attr)
        if assigned is None:
            raise ValueError(f"Could not resolve {node.attr} in model source.")
        return _evaluate_vector_expr(assigned, tree, theta_dim)

    raise ValueError(
        "Unsupported prior-bound expression: "
        + ast.dump(node, include_attributes=False)
    )


def _extract_prior_bounds(tree: ast.AST, theta_dim: int, source_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "prior" for t in node.targets):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        if not _call_name(node.value.func).endswith("BoxUniform"):
            continue

        kwargs = {kw.arg: kw.value for kw in node.value.keywords if kw.arg is not None}
        if "low" not in kwargs or "high" not in kwargs:
            continue

        lower = _evaluate_vector_expr(kwargs["low"], tree, theta_dim)
        upper = _evaluate_vector_expr(kwargs["high"], tree, theta_dim)
        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)

        if lower.size == 1:
            lower = np.repeat(lower.item(), theta_dim)
        if upper.size == 1:
            upper = np.repeat(upper.item(), theta_dim)

        if lower.size != theta_dim or upper.size != theta_dim:
            raise ValueError(
                f"Prior-bound dimension mismatch in {source_file}: "
                f"theta_dim={theta_dim}, low={lower.size}, high={upper.size}"
            )
        if np.any(upper <= lower):
            raise ValueError(f"Invalid prior bounds in {source_file}: upper <= lower")

        return lower, upper

    raise ValueError(f"Could not find prior = BoxUniform(...) in {source_file}")


def load_model_spec(model_name: str) -> ModelSpec:
    source_file = MODEL_FILES[model_name]
    source = source_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_file))
    theta_dim = _extract_theta_dim(tree, source_file)
    lower, upper = _extract_prior_bounds(tree, theta_dim, source_file)
    return ModelSpec(model_name, theta_dim, lower, upper)


# =============================================================================
# Result discovery
# =============================================================================

def _parts_lower(path: Path) -> List[str]:
    return [p.lower() for p in path.parts]


def infer_model_name(path: Path) -> Optional[str]:
    """Infer which model generated a result from the current repository layout."""
    parts = _parts_lower(path)

    if "demo_slcp" in parts:
        return "SLCP"
    if "demo_glm" in parts:
        return "GLM"
    if "demo_gk" in parts:
        return "gk"
    if "demo_bgk" in parts:
        return "BGK"

    for parent, child in zip(parts, parts[1:]):
        if parent in {"stopping", "view3"}:
            if child == "slcp":
                return "SLCP"
            if child == "glm":
                return "GLM"
        if parent == "slcp_mnist" and child in {
            "trajectory", "nle_initialization", "critic_capacity", "optimization",
        }:
            return "SLCP"
        if parent == "supplementary" and child in {
            "initial_bandwidth", "annealing_factor", "simulation_budget",
        }:
            return "SLCP"
        if parent == "view1" and child in {"add3", "add4", "add5"}:
            return "SLCP"

    return None


def discover_numeric_experiments(search_root: Path) -> List[Tuple[Path, str]]:
    """
    Find directories containing metrics_summary_merged.csv for non-MNIST experiments.
    Returns (experiment_dir, model_name).
    """
    found: List[Tuple[Path, str]] = []

    for csv_path in search_root.rglob("metrics_summary_merged.csv"):
        parts = _parts_lower(csv_path)
        if "demo_mnist" in parts:
            continue
        if "postprocessed" in parts:
            continue

        model_name = infer_model_name(csv_path)
        if model_name is None:
            continue

        found.append((csv_path.parent, model_name))

    # Deduplicate and sort for deterministic processing.
    unique = {(str(d.resolve()), m): (d, m) for d, m in found}
    return sorted(unique.values(), key=lambda x: str(x[0]).lower())


# =============================================================================
# CSV parsing and metrics
# =============================================================================

def parse_vector_cell(cell) -> np.ndarray:
    """Parse a vector stored as JSON/Python-list text into a 1D float array."""
    if isinstance(cell, np.ndarray):
        return cell.astype(float, copy=False).reshape(-1)
    if isinstance(cell, (list, tuple)):
        return np.asarray(cell, dtype=float).reshape(-1)

    if cell is None:
        raise ValueError("Empty vector field.")
    if isinstance(cell, float) and np.isnan(cell):
        raise ValueError("Empty vector field (NaN).")

    text = str(cell).strip()
    if not text:
        raise ValueError("Empty vector field.")

    try:
        return np.asarray(json.loads(text), dtype=float).reshape(-1)
    except Exception:
        pass

    try:
        return np.asarray(ast.literal_eval(text), dtype=float).reshape(-1)
    except Exception as exc:
        raise ValueError(f"Could not parse vector field: {text[:200]}") from exc


def _pick_vector(row: pd.Series, preferred: Sequence[str], label: str) -> np.ndarray:
    for col in preferred:
        if col not in row.index:
            continue
        value = row[col]
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return parse_vector_cell(value)
    raise ValueError(f"Could not find a usable {label} vector. Tried columns: {list(preferred)}")


def compute_one_row_metrics(row: pd.Series, spec: ModelSpec) -> Dict[str, object]:
    """Clip estimate/reference to the model prior and compute parameter-space errors."""
    estimate = _pick_vector(
        row,
        ["estimate_theta_processed", "estimate_theta"],
        "estimate",
    )
    reference = _pick_vector(
        row,
        ["true_theta_processed", "true_theta"],
        "reference",
    )

    if estimate.size != spec.theta_dim:
        raise ValueError(
            f"Estimate dimension {estimate.size} does not match theta_dim={spec.theta_dim}."
        )
    if reference.size != spec.theta_dim:
        raise ValueError(
            f"Reference dimension {reference.size} does not match theta_dim={spec.theta_dim}."
        )

    est_outside = (estimate < spec.lower) | (estimate > spec.upper)
    ref_outside = (reference < spec.lower) | (reference > spec.upper)

    estimate_clip = np.clip(estimate, spec.lower, spec.upper)
    reference_clip = np.clip(reference, spec.lower, spec.upper)

    diff = estimate_clip - reference_clip
    abs_diff = np.abs(diff)
    width = spec.upper - spec.lower

    result: Dict[str, object] = {}

    for key in [
        "n_data",
        "run_id",
        "method",
        "baseline_method",
        "time_used",
        "n_sim",
    ]:
        if key in row.index:
            result[key] = row[key]

    result["estimate_clipped_dims"] = int(est_outside.sum())
    result["true_clipped_dims"] = int(ref_outside.sum())
    result["estimate_was_clipped"] = int(est_outside.any())
    result["true_was_clipped"] = int(ref_outside.any())

    result["param_mae"] = float(np.mean(abs_diff))
    result["param_mse"] = float(np.mean(diff ** 2))
    result["param_nrmse"] = float(np.sqrt(np.mean((diff / width) ** 2)))

    for j in range(spec.theta_dim):
        result[f"theta{j + 1}_mae"] = float(abs_diff[j])

    result["estimate_theta_clipped"] = json.dumps(estimate_clip.tolist())
    result["true_theta_clipped"] = json.dumps(reference_clip.tolist())

    return result


def summarize_by_method(row_metrics_df: pd.DataFrame, theta_dim: int) -> pd.DataFrame:
    metric_cols = [
        "param_mae",
        "param_mse",
        "param_nrmse",
        "estimate_clipped_dims",
        "true_clipped_dims",
        "estimate_was_clipped",
        "true_was_clipped",
    ]
    metric_cols.extend(f"theta{j + 1}_mae" for j in range(theta_dim))
    metric_cols = [c for c in metric_cols if c in row_metrics_df.columns]

    summary_rows: List[Dict[str, object]] = []

    for method, group in row_metrics_df.groupby("method", dropna=False):
        out: Dict[str, object] = {
            "method": method,
            "num_runs": int(len(group)),
        }

        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            out[f"{col}_mu"] = float(values.mean())
            std_val = values.std(ddof=1)
            out[f"{col}_std"] = 0.0 if pd.isna(std_val) else float(std_val)

        summary_rows.append(out)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty and "param_nrmse_mu" in summary_df.columns:
        summary_df = summary_df.sort_values("param_nrmse_mu", ascending=True)

    return summary_df.reset_index(drop=True)


def process_one_csv(
    input_csv: Path,
    output_dir: Path,
    spec: ModelSpec,
    rename_columns: Optional[Dict[str, str]] = None,
) -> Tuple[int, int, Optional[pd.DataFrame]]:
    """Process one merged CSV and write row-level and method-level summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        return 0, 0, None
    if input_csv.stat().st_size == 0:
        print(f"  [SKIP] Empty file: {input_csv.name}")
        return 0, 0, None

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    if rename_columns:
        df = df.rename(columns=rename_columns)

    if "method" not in df.columns:
        raise ValueError(f"Missing 'method' column in {input_csv}")

    row_metrics: List[Dict[str, object]] = []
    failed_rows: List[Dict[str, object]] = []

    for idx, row in df.iterrows():
        try:
            row_metrics.append(compute_one_row_metrics(row, spec))
        except Exception as exc:
            failed_rows.append(
                {
                    "row_index": int(idx),
                    "method": row.get("method", ""),
                    "error": repr(exc),
                }
            )

    row_df = pd.DataFrame(row_metrics)
    row_path = output_dir / "clipped_row_metrics.csv"
    row_df.to_csv(row_path, index=False, encoding="utf-8-sig")

    failed_path = output_dir / "failed_rows.csv"
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(failed_path, index=False, encoding="utf-8-sig")
    elif failed_path.exists():
        failed_path.unlink()

    if row_df.empty:
        print(f"  [WARN] No rows could be processed from {input_csv.name}")
        return len(df), len(failed_rows), None

    summary_df = summarize_by_method(row_df, spec.theta_dim)
    summary_df.to_csv(
        output_dir / "clipped_summary_by_method.csv",
        index=False,
        encoding="utf-8-sig",
    )

    core_cols = [
        "method",
        "num_runs",
        "param_mae_mu",
        "param_mae_std",
        "param_mse_mu",
        "param_mse_std",
        "param_nrmse_mu",
        "param_nrmse_std",
        "estimate_clipped_dims_mu",
        "estimate_clipped_dims_std",
        "estimate_was_clipped_mu",
        "estimate_was_clipped_std",
    ]
    core_cols = [c for c in core_cols if c in summary_df.columns]
    summary_df[core_cols].to_csv(
        output_dir / "clipped_core_summary_by_method.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return len(df), len(failed_rows), summary_df


# =============================================================================
# Main driver
# =============================================================================

def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def run(search_root: Path) -> None:
    experiments = discover_numeric_experiments(search_root)

    print("=" * 88)
    print("Numerical-result post-processing")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Search root  : {search_root}")
    print(f"Experiments  : {len(experiments)}")
    print("=" * 88)

    if not experiments:
        print("No non-MNIST metrics_summary_merged.csv files were found.")
        print("Run the experiments first, then run this script again.")
        return

    spec_cache: Dict[str, ModelSpec] = {}
    index_rows: List[Dict[str, object]] = []

    for experiment_dir, model_name in experiments:
        if model_name not in spec_cache:
            spec_cache[model_name] = load_model_spec(model_name)
        spec = spec_cache[model_name]

        print("\n" + "-" * 88)
        print(f"Experiment : {relative_to_project(experiment_dir)}")
        print(f"Model      : {model_name}")
        print(f"theta_dim  : {spec.theta_dim}")
        print(f"prior low  : {spec.lower.tolist()}")
        print(f"prior high : {spec.upper.tolist()}")

        post_root = experiment_dir / "postprocessed"

        # A. Relative to the true parameters.
        true_csv = experiment_dir / "metrics_summary_merged.csv"
        true_rows, true_failed, true_summary = process_one_csv(
            input_csv=true_csv,
            output_dir=post_root / "VSTRUE",
            spec=spec,
        )
        print(
            f"  VSTRUE: rows={true_rows}, failed={true_failed}, "
            f"output={relative_to_project(post_root / 'VSTRUE')}"
        )

        index_rows.append(
            {
                "model": model_name,
                "comparison": "VSTRUE",
                "experiment_dir": relative_to_project(experiment_dir),
                "input_csv": relative_to_project(true_csv),
                "rows": true_rows,
                "failed_rows": true_failed,
            }
        )

        # B. Relative to MLE, when the experiment produced this file.
        mle_csv = experiment_dir / "mle_comparison_summary_merged.csv"
        if mle_csv.exists() and mle_csv.stat().st_size > 0:
            rename_columns_for_mle = {
                "compare_method": "method",
                "vs_MLE_estimate_theta_processed": "estimate_theta_processed",
                "vs_MLE_true_theta_processed": "true_theta_processed",
                "vs_MLE_estimate_theta": "estimate_theta",
                "vs_MLE_true_theta": "true_theta",
            }

            mle_rows, mle_failed, mle_summary = process_one_csv(
                input_csv=mle_csv,
                output_dir=post_root / "VSMLE",
                spec=spec,
                rename_columns=rename_columns_for_mle,
            )
            print(
                f"  VSMLE : rows={mle_rows}, failed={mle_failed}, "
                f"output={relative_to_project(post_root / 'VSMLE')}"
            )

            index_rows.append(
                {
                    "model": model_name,
                    "comparison": "VSMLE",
                    "experiment_dir": relative_to_project(experiment_dir),
                    "input_csv": relative_to_project(mle_csv),
                    "rows": mle_rows,
                    "failed_rows": mle_failed,
                }
            )
        else:
            print("  VSMLE : no non-empty mle_comparison_summary_merged.csv; skipped")

    index_df = pd.DataFrame(index_rows)
    index_path = PROJECT_ROOT / "postprocess_numeric_index.csv"
    index_df.to_csv(index_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 88)
    print("Numerical post-processing completed.")
    print(f"Index written to: {index_path}")
    print("=" * 88)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically post-process merged numerical experiment results in the "
            "current vsm_supplementary repository."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(PROJECT_ROOT),
        help="Directory to scan recursively. Default: project root.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scan_root = Path(args.root).expanduser().resolve()
    if not scan_root.exists():
        sys.exit(f"Search root does not exist: {scan_root}")
    run(scan_root)
