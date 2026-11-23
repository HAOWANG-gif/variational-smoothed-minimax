"""Validated Cartesian sweeps for the SLCP and GLM overall-stopping parameters."""
import argparse
from datetime import datetime
from itertools import product
import json
import math
from numbers import Integral, Real
from pathlib import Path
import subprocess
import sys


PARAMETERS = ("delta_stop", "K_stop", "epsilon_min", "n_sim")
DEFAULT_CONFIG = Path(__file__).resolve().with_name("overall_stopping_grid.json")


def _validate_value(key, value):
    integer = key in {"K_stop", "n_sim"}
    kind = Integral if integer else Real
    if (isinstance(value, bool) or not isinstance(value, kind)
            or not math.isfinite(value) or value <= 0):
        description = "a positive integer" if integer else "a finite positive number"
        raise ValueError(f"{key} must be {description}.")
    return int(value) if integer else float(value)


def validate_case(parameters):
    if not isinstance(parameters, dict) or set(parameters) != set(PARAMETERS):
        raise ValueError(f"Stopping parameters must contain exactly {PARAMETERS}.")
    return {key: _validate_value(key, parameters[key]) for key in PARAMETERS}


def case_name(parameters):
    case = validate_case(parameters)
    delta = str(case["delta_stop"]).replace(".", "p")
    epsilon = str(case["epsilon_min"]).replace(".", "p")
    return f"d{delta}_k{case['K_stop']}_e{epsilon}_n{case['n_sim']}"


def expand_grid(grid):
    if not isinstance(grid, dict) or set(grid) != set(PARAMETERS):
        raise ValueError(f"Config must contain exactly the four lists {PARAMETERS}.")
    values = []
    for key in PARAMETERS:
        if not isinstance(grid[key], list) or not grid[key]:
            raise ValueError(f"{key} must be a nonempty list.")
        candidates = [_validate_value(key, value) for value in grid[key]]
        if len(set(candidates)) != len(candidates):
            raise ValueError(f"Duplicate values in {key}.")
        values.append(candidates)
    return [dict(zip(PARAMETERS, combination)) for combination in product(*values)]


def load_grid(path):
    with Path(path).expanduser().open(encoding="utf-8-sig") as stream:
        grid = json.load(stream)
    expand_grid(grid)
    return grid


def _add_grid_arguments(parser):
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="JSON file specifying the grid of stopping parameters.")
    parser.add_argument("--list-configs", action="store_true",
                        help="List validated parameter combinations without running the experiment.")


def run_sensitivity(entrypoint, arguments=None):
    """Run all cases sequentially, or execute one immutable selected case.

    A selected case never rereads the grid. Its JSON is forwarded unchanged
    through the ordinary launcher to each worker, including during resume.
    """
    entrypoint = Path(entrypoint).resolve()
    parser = argparse.ArgumentParser(add_help=False)
    _add_grid_arguments(parser)
    parser.add_argument("--stopping-params", default="")
    parser.add_argument("--exp-tag", default="")
    parser.add_argument("--mode", choices=("launch", "worker", "merge"), default="launch")
    args, forwarded = parser.parse_known_args(sys.argv[1:] if arguments is None else arguments)

    if "--help" in forwarded or "-h" in forwarded:
        from ._driver import build_parser

        help_parser = build_parser()
        _add_grid_arguments(help_parser)
        help_parser.print_help()
        return 0

    if args.stopping_params:
        selected = validate_case(json.loads(args.stopping_params))
        if args.list_configs:
            print(case_name(selected), json.dumps(selected, sort_keys=True))
            return 0
        if args.mode == "merge" and not args.exp_tag:
            raise ValueError("--mode merge requires the original --exp-tag.")
        from ._driver import run_experiment
        driver_args = ["--stopping-params", args.stopping_params,
                       "--mode", args.mode, *forwarded]
        if args.exp_tag:
            driver_args.extend(["--exp-tag", args.exp_tag])
        run_experiment(entrypoint, {"Ours": {"overall_stop_mode": "paper"}}, driver_args)
        return 0

    cases = expand_grid(load_grid(args.config))
    print(f"{entrypoint.parent.parent.name}: {len(cases)} stopping-parameter combinations", flush=True)
    if args.list_configs:
        for case in cases:
            print(case_name(case), json.dumps(case, sort_keys=True))
        return 0
    if args.mode == "merge" and not args.exp_tag:
        raise ValueError("--mode merge requires the original --exp-tag.")
    # Every combination in this invocation uses one tag, including timestamp defaults.
    tag = args.exp_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Experiment tag: {tag}", flush=True)
    failures = []
    for index, case in enumerate(cases, 1):
        name = case_name(case)
        print(f"\n[{index}/{len(cases)}] {name}", flush=True)
        command = [sys.executable, "-u", str(entrypoint),
                   "--stopping-params", json.dumps(case, sort_keys=True),
                   "--mode", args.mode, "--exp-tag", tag, *forwarded]
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures.append((name, result.returncode))
    if failures:
        print(f"Failed sensitivity cases: {failures}", flush=True)
    return int(bool(failures))
