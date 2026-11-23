"""Compare SLCP initialization and VSM annealing stages with the per-run MLE."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
METRICS = ("MAE", "MSE", "NRMSE")
OUTPUT_FILES = (
    "run_status.csv", "per_run_metrics.csv", "iteration_summary.csv", "theta_history_vs_mle.xlsx",
    "metrics_mean_std.png", "metrics_mean_std.pdf", "metrics_distribution.png", "metrics_distribution.pdf",
)


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from postprocess_numeric_results import compute_one_row_metrics, load_model_spec  # noqa: E402


def read_theta_matrix(path: Path, theta_dim: int = 5) -> np.ndarray:
    """Accept a headerless history or a final estimate with theta column names."""
    frame = pd.read_csv(path, header=None, encoding="utf-8-sig", skip_blank_lines=False)
    if frame.shape[1] != theta_dim:
        raise ValueError(f"{path.name}: expected {theta_dim} parameter columns, got {frame.shape[1]}")
    expected = [f"theta_{i}" for i in range(theta_dim)]
    if [str(x).strip() for x in frame.iloc[0]] == expected:
        frame = frame.iloc[1:]
    if frame.empty:
        raise ValueError(f"{path.name}: no parameter rows")
    # Keep invalid rows in place to preserve the stage numbers.
    return frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def compute_history_metrics(history: np.ndarray, mle: np.ndarray, run_id: int, spec,
                            initial_theta: np.ndarray | None = None) -> pd.DataFrame:
    """Compute per-stage errors while preserving the original stage numbers."""
    history = np.asarray(history, dtype=float)
    mle = np.asarray(mle, dtype=float).reshape(-1)
    if history.ndim != 2 or history.shape[1] != spec.theta_dim:
        raise ValueError("History shape does not match SLCP parameter dimension")
    if mle.shape != (spec.theta_dim,) or not np.isfinite(mle).all():
        raise ValueError("MLE must contain one finite parameter vector")
    reference = mle.copy()
    reference[[2, 3]] = np.abs(reference[[2, 3]])
    points = [(outer_k + 1, theta) for outer_k, theta in enumerate(history)]
    if initial_theta is not None:
        initial_theta = np.asarray(initial_theta, dtype=float).reshape(-1)
        if initial_theta.shape != (spec.theta_dim,):
            raise ValueError("Initial estimate dimension does not match SLCP parameter dimension")
        points.insert(0, (0, initial_theta))
    rows = []
    for iteration, theta in points:
        valid = bool(np.isfinite(theta).all())
        point_type = "initial" if iteration == 0 else "stage"
        record = {"run_id": run_id, "iteration": iteration,
                  "outer_k": iteration - 1 if iteration > 0 else pd.NA,
                  "point_type": point_type, "valid": valid,
                  "error": "" if valid else f"non-finite/non-numeric {point_type} parameter"}
        record.update({f"theta_raw_{j}": value for j, value in enumerate(theta)})
        record.update({f"mle_raw_{j}": value for j, value in enumerate(mle)})
        for name in METRICS:
            record[name] = np.nan
        if valid:
            estimate = theta.copy()
            # The signs of the third and fourth SLCP parameters are unidentifiable.
            estimate[[2, 3]] = np.abs(estimate[[2, 3]])
            metrics = compute_one_row_metrics(pd.Series({
                "estimate_theta_processed": estimate.tolist(),
                "true_theta_processed": reference.tolist(),
            }), spec)
            for name, original in zip(METRICS, ("param_mae", "param_mse", "param_nrmse")):
                record[name] = metrics[original]
            for prefix, vector in (
                ("theta_used", json.loads(metrics["estimate_theta_clipped"])),
                ("mle_used", json.loads(metrics["true_theta_clipped"])),
            ):
                record.update({f"{prefix}_{j}": value for j, value in enumerate(vector)})
            record["theta_clipped_dims"] = metrics["estimate_clipped_dims"]
            record["mle_clipped_dims"] = metrics["true_clipped_dims"]
        rows.append(record)
    frame = pd.DataFrame(rows)
    front = ["run_id", "iteration", "outer_k", "point_type", "valid", *METRICS, "error"]
    frame = frame.reindex(columns=front + [c for c in frame if c not in front])
    frame["outer_k"] = frame["outer_k"].astype("Int64")
    return frame


def summarize_iterations(details: pd.DataFrame) -> pd.DataFrame:
    if details.duplicated(["run_id", "iteration"]).any():
        raise ValueError("Duplicate run_id/iteration; refusing to count a run twice")
    valid = details.loc[details["valid"]].copy()
    valid = valid.loc[np.isfinite(valid[list(METRICS)]).all(axis=1)]
    if valid.empty:
        raise ValueError("No valid iteration metrics to summarize")
    grouped = valid.groupby("iteration", sort=True)
    summary = grouped[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "n_runs", grouped["run_id"].nunique())
    summary = summary.reset_index()
    summary.insert(1, "outer_k", (summary["iteration"] - 1).where(summary["iteration"].gt(0)).astype("Int64"))
    summary.insert(2, "point_type", np.where(summary["iteration"].eq(0), "initial", "stage"))
    # Sample standard deviations are undefined for a single observation.
    return summary


def collect_runs(input_dir: Path, spec, expected_runs: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dirs = [p for p in input_dir.iterdir() if p.is_dir() and re.fullmatch(r"run_\d+", p.name)]
    run_dirs.sort(key=lambda p: int(p.name.split("_")[-1]))
    ids = [int(p.name.split("_")[-1]) for p in run_dirs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate numeric run IDs, e.g. run_1 and run_01")
    frames, records = [], []
    for run_dir, run_id in zip(run_dirs, ids):
        history_path = run_dir / "Ours" / "theta_history.csv"
        mle_path = run_dir / "MLE" / "final_theta.csv"
        initial_path = run_dir / "init_theta.csv"
        record = {"run_id": run_id, "status": "ok", "history_rows": 0,
                  "initial_rows": 0, "initial_status": "unavailable", "total_rows": 0,
                  "valid_rows": 0, "invalid_rows": 0, "message": "",
                  "history_csv": history_path.as_posix(), "mle_csv": mle_path.as_posix(),
                  "initial_csv": initial_path.as_posix()}
        try:
            mle = read_theta_matrix(mle_path, spec.theta_dim)
            if mle.shape[0] != 1:
                raise ValueError("MLE/final_theta.csv must contain exactly one estimate")
            messages = []
            history = np.empty((0, spec.theta_dim))
            try:
                history = read_theta_matrix(history_path, spec.theta_dim)
            except Exception as exc:
                messages.append(f"History unavailable: {exc}")
            initial = None
            try:
                initial_matrix = read_theta_matrix(initial_path, spec.theta_dim)
                if initial_matrix.shape[0] != 1:
                    raise ValueError("init_theta.csv must contain exactly one initial estimate")
                initial = initial_matrix[0]
                record["initial_status"] = "ok" if np.isfinite(initial).all() else "invalid"
            except Exception as exc:
                record["initial_status"] = "missing" if not initial_path.exists() else "invalid"
                messages.append(f"Initial estimate unavailable: {exc}")
            if len(history) == 0 and initial is None:
                raise ValueError("; ".join(messages))
            frame = compute_history_metrics(history, mle[0], run_id, spec, initial_theta=initial)
            record.update(history_rows=len(history), initial_rows=int(initial is not None),
                          total_rows=len(frame), valid_rows=int(frame["valid"].sum()),
                          invalid_rows=int((~frame["valid"]).sum()))
            if record["invalid_rows"]:
                messages.append("Invalid parameter rows omitted from aggregates; see per-run metrics")
            if messages:
                record.update(status="partial" if record["valid_rows"] else "failed", message="; ".join(messages))
            frames.append(frame)
        except Exception as exc:
            record.update(status="failed", message=str(exc))
        records.append(record)
    if expected_runs is not None:
        for missing in sorted(set(range(1, expected_runs + 1)) - set(ids)):
            records.append({"run_id": missing, "status": "missing", "history_rows": 0,
                            "valid_rows": 0, "invalid_rows": 0, "message": "run directory missing"})
    audit = pd.DataFrame(records)
    if not audit.empty:
        audit = audit.sort_values("run_id").reset_index(drop=True)
    details = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return details, audit


def make_plots(details: pd.DataFrame, summary: pd.DataFrame, output: Path, title: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#236B9C", "#C66A24", "#32836A")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    valid = details.loc[details["valid"]]
    x = summary["iteration"].to_numpy()
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for j, (ax, metric, color) in enumerate(zip(axes, METRICS, colors)):
        mean = summary[f"{metric}_mean"].to_numpy()
        std = summary[f"{metric}_std"].to_numpy()
        # A missing stage breaks the line.
        axis_x = np.arange(int(x.min()), int(x.max()) + 1)
        mean = pd.Series(mean, index=x).reindex(axis_x).to_numpy()
        std = pd.Series(std, index=x).reindex(axis_x).to_numpy()
        ax.plot(axis_x, mean, color=color, linewidth=2, label="Mean")
        ax.fill_between(axis_x, mean - std, mean + std, color=color, alpha=.20, label="Mean +/- 1 SD")
        if 0 in x:
            ax.scatter([0], [mean[0]], color=color, s=26, zorder=3)
        ax.set(title=metric, xlabel="Annealing stage (0 = initialization)", ylabel=f"{metric} vs MLE", ylim=(0, None))
        ax.grid(alpha=.20)
        if j == 0:
            ax.legend(frameon=False, fontsize=9)
    figure.suptitle(title + "\nMean and sample SD across available runs", fontsize=14)
    saved = []
    for extension in ("png", "pdf"):
        path = output / f"metrics_mean_std.{extension}"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        saved.append(path)
    plt.close(figure)

    # Use up to seven observed stages, including the first and last.
    selected = x[np.unique(np.linspace(0, len(x) - 1, min(7, len(x))).round().astype(int))]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, metric, color in zip(axes, METRICS, colors):
        batches = [valid.loc[valid["iteration"].eq(stage), metric].to_numpy() for stage in selected]
        boxes = ax.boxplot(batches, patch_artist=True, showfliers=True, widths=.6,
                           medianprops={"color": "#19232D", "linewidth": 1.4},
                           flierprops={"marker": ".", "markersize": 3, "alpha": .4})
        for box in boxes["boxes"]:
            box.set(facecolor=color, alpha=.35)
        ax.set_xticks(range(1, len(selected) + 1),
                      [f"{'0 (init)' if stage == 0 else stage}\n(n={len(batch)})" for stage, batch in zip(selected, batches)], fontsize=8)
        ax.set(title=metric, xlabel="Selected stages (0 = initialization)", ylabel=f"{metric} vs MLE", ylim=(0, None))
        ax.grid(axis="y", alpha=.20)
    figure.suptitle(title + "\nDistribution across available runs (box: Q1-Q3; whiskers: 1.5 IQR)", fontsize=13)
    for extension in ("png", "pdf"):
        path = output / f"metrics_distribution.{extension}"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        saved.append(path)
    plt.close(figure)
    return saved


def write_workbook(path: Path, summary: pd.DataFrame, details: pd.DataFrame,
                   audit: pd.DataFrame, notes: list[tuple[str, object]]) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    workbook = Workbook()
    workbook.remove(workbook.active)
    sheets = [("Stage summary", summary), ("Per-run metrics", details), ("Run status", audit),
              ("Metric notes", pd.DataFrame(notes, columns=["item", "description"]))]
    for title, frame in sheets:
        if len(frame) > 1_048_575:
            raise ValueError(f"{title}: exceeds Excel row limit; full data are in the CSV output")
        sheet = workbook.create_sheet(title)
        sheet.append(list(frame.columns))
        for row in frame.itertuples(index=False, name=None):
            values = []
            for value in row:
                if value is None or (not isinstance(value, str) and pd.isna(value)):
                    value = None
                elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    value = str(value)
                elif isinstance(value, np.generic):
                    value = value.item()
                values.append(value)
            sheet.append(values)
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        sheet.sheet_view.showGridLines = False
        for cell in sheet[1]:
            cell.fill = PatternFill("solid", fgColor="244A64")
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(vertical="center")
        sheet.row_dimensions[1].height = 25
        for j, column in enumerate(frame.columns, 1):
            letter = sheet.cell(1, j).column_letter
            width = 20 if column in {"iteration", "outer_k", "n_runs", "run_id", "valid", "status"} else 24
            if column in {"history_csv", "mle_csv", "initial_csv", "message", "error"}:
                width = 55
            if column == "description":
                width = 105
            if column == "item":
                width = 30
            sheet.column_dimensions[letter].width = width
            for cells in sheet.iter_rows(min_row=2, min_col=j, max_col=j):
                cell = cells[0]
                if isinstance(cell.value, float):
                    cell.number_format = "0.000000" if column != "epsilon_over_epsilon0" else "0.000E+00"
                if column in {"description", "message", "error", "history_csv", "mle_csv", "initial_csv"}:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
        if title == "Metric notes":
            for row_number in range(2, sheet.max_row + 1):
                sheet.row_dimensions[row_number].height = 42
        if title == "Run status":
            for row_number in range(2, sheet.max_row + 1):
                sheet.row_dimensions[row_number].height = 65
    workbook.save(path)


def run(input_dir: Path, output_dir: Path) -> int:
    input_dir, output_dir = input_dir.expanduser().resolve(), output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clear generated files so a failed run leaves no stale summary.
    for name in OUTPUT_FILES:
        path = output_dir / name
        if path.exists():
            if not path.resolve().is_relative_to(output_dir):
                raise ValueError(f"Output path escapes the specified output directory: {path}")
            path.unlink()
    config_path = input_dir.parent / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.exists() else {}
    expected_runs = config.get("experiment", {}).get("n_runs")
    gamma = config.get("methods", {}).get("Ours", {}).get("gamma")
    if gamma is not None and (not np.isfinite(gamma) or gamma <= 0):
        raise ValueError("Configured gamma must be finite and positive")
    spec = load_model_spec("SLCP")
    details, audit = collect_runs(input_dir, spec, expected_runs)
    audit.to_csv(output_dir / "run_status.csv", index=False, encoding="utf-8-sig")
    if details.empty:
        raise ValueError("No usable Ours/MLE pairs; see run_status.csv")
    summary = summarize_iterations(details)
    if gamma is not None:
        for frame in (details, summary):
            outer_k = frame["outer_k"].to_numpy(dtype=float, na_value=np.nan)
            frame.insert(3, "epsilon_over_epsilon0", float(gamma) ** outer_k)
    details.to_csv(output_dir / "per_run_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "iteration_summary.csv", index=False, encoding="utf-8-sig")
    n_valid_runs = details.loc[details["valid"], "run_id"].nunique()
    notes = [
        ("input_dir", input_dir.as_posix()), ("configuration", config_path.as_posix()),
        ("configured_runs", expected_runs if expected_runs is not None else "not recorded"),
        ("actual_valid_runs", n_valid_runs), ("gamma", gamma if gamma is not None else "not recorded"),
        ("reference", "Each run uses its own MLE/final_theta.csv."),
        ("initial_point", "Iteration 0 uses init_theta.csv from the same run."),
        ("history_rows", "The first headerless row in theta_history.csv records the estimate after stage 1."),
        ("iteration", "For completed stages, outer_k = iteration - 1. The initialization has a blank outer_k."),
        ("epsilon_over_epsilon0", "For stages, this is gamma**outer_k. The absolute initial bandwidth can vary by run."),
        ("parameter_transform", "The third and fourth components of both estimates are transformed with abs()."),
        ("clipping", f"Both estimates are clipped to the prior bounds: {spec.lower.tolist()} through {spec.upper.tolist()}."),
        ("MAE", "Mean absolute error across parameters for the same run and stage."),
        ("MSE", "Mean squared error across parameters for the same run and stage."),
        ("NRMSE", "Root mean squared error after dividing each parameter error by its prior width."),
        ("mean/std", "Valid runs at each stage have equal weight. Sample standard deviations use ddof=1."),
        ("missing_iterations", "The aggregation uses available, valid rows at each original stage number."),
        ("missing_initial", "The run audit records missing or invalid initial estimates. Valid stages still contribute."),
        ("plots", "The mean plot band is one sample standard deviation. Box plots use up to seven observed stages."),
        ("changing_run_count", "Stage summaries can use different sets of runs. See n_runs for each stage."),
    ]
    write_workbook(output_dir / "theta_history_vs_mle.xlsx", summary, details, audit, notes)
    make_plots(details, summary, output_dir, f"Ours vs MLE | SLCP | {input_dir.name} | {n_valid_runs} runs")
    errors = int(audit["status"].isin(["failed", "missing", "partial"]).sum())
    print(f"Valid runs: {n_valid_runs}; initial rows: {int(details.iteration.eq(0).sum())}; "
          f"history rows: {int(details.iteration.gt(0).sum())}; valid rows: {int(details['valid'].sum())}")
    print(f"Plotted points: {int(summary.iteration.min())}..{int(summary.iteration.max())}; "
          f"available runs: {int(summary.n_runs.min())}..{int(summary.n_runs.max())}")
    print(f"Runs needing attention: {errors}; output: {output_dir}")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="n_data_* directory containing run_* folders")
    parser.add_argument("--output-dir", type=Path, help="Destination for the workbook, CSVs and plots")
    args = parser.parse_args()
    try:
        return run(args.input_dir, args.output_dir or args.input_dir / "theta_history_analysis")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
