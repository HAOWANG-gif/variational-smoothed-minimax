"""Persist completed runs and summarize VSM stopping criteria (standard library only)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile


RECORD_KEYS = ("params_summary", "metrics_summary", "mle_comparison_summary")
REASONS = ("theta_convergence", "epsilon_min", "simulation_budget")


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                     dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer(stream)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_run_records(run_dir: Path | str, records: dict) -> None:
    if any(not isinstance(records.get(key), list) for key in RECORD_KEYS):
        raise ValueError("Run records must contain the three summary lists")
    _atomic_write(Path(run_dir) / "run_records.json",
                  lambda stream: json.dump(records, stream, ensure_ascii=False, indent=2))


def load_run_records(run_dir: Path | str) -> dict:
    path = Path(run_dir) / "run_records.json"
    with path.open(encoding="utf-8-sig") as stream:
        records = json.load(stream)
    if not isinstance(records, dict) or any(
        not isinstance(records.get(key), list) for key in RECORD_KEYS
    ):
        raise ValueError(f"Incomplete completed-run records: {path}")
    return records


def _truth(value) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0"}:
        return True
    if text in {"false", "0", "0.0"}:
        return False
    raise ValueError(f"Invalid stopping flag: {value!r}")


def _number(row: dict, key: str, default=0) -> float:
    value = row.get(key, default)
    return float(default if value is None or value == "" else value)


def summarize_stopping_records(records: list[dict]) -> list[dict]:
    groups: dict[int, list[dict]] = {}
    seen = set()
    for row in records:
        if row.get("method") != "Ours":
            continue
        key = (int(row["n_data"]), int(row["run_id"]))
        if key in seen:
            raise ValueError(f"Duplicate VSM result for n_data/run_id={key}")
        seen.add(key)
        if row.get("vsm_stop_reason") not in REASONS:
            raise ValueError(f"Missing or invalid VSM stopping reason for {key}")
        groups.setdefault(key[0], []).append(row)

    summaries = []
    for n_data, rows in sorted(groups.items()):
        result = {"n_data": n_data, "num_runs": len(rows)}
        for reason, flag in zip(REASONS, ("theta", "epsilon", "budget")):
            result[f"{reason}_count"] = sum(_truth(row.get(f"vsm_stop_{flag}")) for row in rows)
            result[f"reason_{reason}_count"] = sum(row["vsm_stop_reason"] == reason for row in rows)
        hits = [_number(row, "vsm_inner_early_stop_count") for row in rows]
        result["inner_early_stop_runs"] = sum(value > 0 for value in hits)
        result["inner_early_stop_hits_total"] = int(sum(hits))
        result["inner_iterations_saved_total"] = int(sum(
            _number(row, "vsm_inner_iterations_saved_total") for row in rows
        ))
        for source, label in (("vsm_n_stages", "n_stages"), ("n_sim", "n_sim")):
            values = [_number(row, source) for row in rows]
            result[f"{label}_mean"] = sum(values) / len(values)
            result[f"{label}_min"] = min(values)
            result[f"{label}_max"] = max(values)
        summaries.append(result)
    return summaries


def write_csv(path: Path | str, rows: list[dict], default_fields=()) -> None:
    keys = list(default_fields)
    keys.extend(key for key in sorted(set().union(*(row.keys() for row in rows))) if key not in keys)
    def write(stream):
        output = csv.DictWriter(stream, fieldnames=keys)
        output.writeheader()
        output.writerows(rows)
    _atomic_write(Path(path), write)


def write_stopping_reports(root_dir: Path | str, records: list[dict], expected_runs: int) -> None:
    root_dir = Path(root_dir)
    summaries = summarize_stopping_records(records)
    for row in summaries:
        row["configured_runs"] = expected_runs
        row["run_count_matches_config"] = row["num_runs"] == expected_runs
    individual = []
    for row in records:
        if row.get("method") == "Ours":
            individual.append({key: value for key, value in row.items()
                               if key in {"n_data", "run_id", "method", "time_used", "n_sim"}
                               or key.startswith("vsm_")})
    write_csv(root_dir / "vsm_stopping_runs.csv", individual,
              ("n_data", "run_id", "method", "n_sim", "vsm_stop_reason"))
    write_csv(root_dir / "vsm_stopping_counts.csv", summaries,
              ("n_data", "num_runs", "configured_runs", "run_count_matches_config"))
