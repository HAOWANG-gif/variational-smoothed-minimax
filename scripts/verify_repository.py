#!/usr/bin/env python3
"""Statically verify the VSM source tree and experiment entry points."""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TENSORS = {
    "models/design_matrix.pt",
    "models/stimulus_I.pt",
    "models/stimulus_t.pt",
}
SLCP_ABLATIONS = {
    "supplementary/initial_bandwidth/c_epsilon_0p0001.py",
    "supplementary/initial_bandwidth/c_epsilon_0p001.py",
    "supplementary/initial_bandwidth/c_epsilon_0p01.py",
    "supplementary/initial_bandwidth/c_epsilon_0p1.py",
    "supplementary/initial_bandwidth/c_epsilon_10.py",
    "supplementary/annealing_factor/gamma_0p1.py",
    "supplementary/annealing_factor/gamma_0p3.py",
    "supplementary/annealing_factor/gamma_0p7.py",
    "supplementary/annealing_factor/gamma_0p9.py",
    "supplementary/simulation_budget/budget_5k.py",
    "supplementary/simulation_budget/budget_50k.py",
}
SLCP_MNIST = {
    "supplementary/slcp_mnist/summarize_results.py",
    "supplementary/slcp_mnist/trajectory/run.py",
    "supplementary/slcp_mnist/trajectory/summarize_history.py",
    "supplementary/slcp_mnist/mnist_width/main.py",
    "supplementary/slcp_mnist/mnist_width/digits_1_vs_2_hidden_5.py",
    "supplementary/slcp_mnist/mnist_width/digits_1_vs_2_hidden_10.py",
    "supplementary/slcp_mnist/mnist_width/digits_1_vs_2_hidden_50.py",
    "supplementary/slcp_mnist/mnist_width/digits_1_vs_2_hidden_100.py",
    "supplementary/slcp_mnist/nle_initialization/main.py",
    "supplementary/slcp_mnist/nle_initialization/nle_1000_candidates_5000.py",
    "supplementary/slcp_mnist/nle_initialization/nle_5000_candidates_25000.py",
    "supplementary/slcp_mnist/nle_initialization/nle_50000_candidates_100000.py",
    "supplementary/slcp_mnist/nle_initialization/nle_100000_candidates_100000.py",
    "supplementary/slcp_mnist/critic_capacity/main.py",
    "supplementary/slcp_mnist/critic_capacity/critic_hidden_32_blocks_1.py",
    "supplementary/slcp_mnist/critic_capacity/critic_hidden_128_blocks_2.py",
    "supplementary/slcp_mnist/critic_capacity/critic_hidden_256_blocks_3.py",
    "supplementary/slcp_mnist/optimization/main.py",
    "supplementary/slcp_mnist/optimization/lr_low.py",
    "supplementary/slcp_mnist/optimization/lr_high.py",
    "supplementary/slcp_mnist/optimization/m1_5_m2_5.py",
    "supplementary/slcp_mnist/optimization/m1_20_m2_20.py",
    "supplementary/slcp_mnist/optimization/k_inner_10.py",
}
STOPPING = {
    f"supplementary/stopping/{model}/{entrypoint}"
    for model in ("slcp", "glm")
    for entrypoint in (
        "main.py",
        "fixed_bandwidth/main.py",
        "fixed_bandwidth/gamma_1_yasuo_1.py",
        "fixed_bandwidth/gamma_1_yasuo_100.py",
        "fixed_bandwidth/gamma_1_yasuo_10000.py",
        "stage_patience/main.py",
        "stage_patience/patience_300.py",
        "stage_patience/patience_600.py",
        "overall_stopping/paper_stopping.py",
    )
} | {"supplementary/stopping/summarize_results.py"}
STOPPING_SUPPORT = {
    "supplementary/stopping/overall_stopping_grid.json",
    "supplementary/stopping/tests/test_stopping_summary.py",
}
NON_MNIST_MAIN = {
    "demo_SLCP/n10.py", "demo_SLCP/n100.py",
    "demo_GLM/n10.py", "demo_GLM/n100.py",
    "demo_gk/n10.py", "demo_gk/n100.py",
    "demo_BGK/n10.py", "demo_BGK/n100.py",
}
MNIST_MAIN = {
    "demo_MNIST/digits_1_vs_2.py",
    "demo_MNIST/digits_4_vs_9.py",
}
LOCAL_ROOTS = {
    "demo_BGK", "demo_GLM", "demo_MNIST", "demo_SLCP", "demo_gk",
    "estimators", "models", "utils", "supplementary",
    "postprocess_mnist_results", "postprocess_numeric_results",
    "view1", "view3",
}
EXCLUDED_DIRECTORIES = {
    ".git", ".idea", ".venv", ".vscode", "__pycache__", "venv", "env", "ENV",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "build", "dist",
    "mnist_openml_cache", "output", "tmp", "results", "结果", "postprocessed", "_logs", "sbi-logs",
    "theta_history_analysis",
}
LEGACY_RESULT_DIRECTORIES = {
    "view1",
    *(f"demo_GLM/{name}" for name in ("1_1", "1_2", "1_3", "2_1", "2_2", "3_1")),
    *(f"demo_SLCP/{name}" for name in ("1_1", "1_100", "1_10000", "2_300", "2_600", "3_all", "3_buget")),
}
EXCLUDED_PREFIXES = ("Exp_", "_shard_", "pretrained_")
EXCLUDED_SUFFIXES = ("_results", ".egg-info")
FORBIDDEN_FILE_TYPES = {".zip", ".pyc", ".pyo"}
CHECKED_FILE_TYPES = FORBIDDEN_FILE_TYPES | {".py", ".sh"}


def files() -> list[Path]:
    source = []
    for directory, children, names in os.walk(ROOT):
        parent = Path(directory)
        children[:] = [
            child for child in children
            if child not in EXCLUDED_DIRECTORIES
            and not child.startswith(EXCLUDED_PREFIXES)
            and not child.endswith(EXCLUDED_SUFFIXES)
            and (parent / child).relative_to(ROOT).as_posix() not in LEGACY_RESULT_DIRECTORIES
        ]
        source.extend(parent / name for name in names if Path(name).suffix.lower() in CHECKED_FILE_TYPES)
    return sorted(source)


def python_trees(paths: list[Path]) -> dict[str, ast.Module]:
    trees = {}
    for path in paths:
        if path.suffix != ".py":
            continue
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=rel, type_comments=True)
        compile(text, rel, "exec", dont_inherit=True)
        trees[rel] = tree
    return trees


def local_module_exists(module: str) -> bool:
    p = ROOT.joinpath(*module.split("."))
    return p.with_suffix(".py").is_file() or p.is_dir()


def verify_imports(trees: dict[str, ast.Module]) -> int:
    checked = 0
    for rel, tree in trees.items():
        package = Path(rel).parts[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(package):
                        raise RuntimeError(f"Relative import outside the repository in {rel}")
                    package_root = ".".join(package[:len(package) - node.level + 1])
                    names = [f"{package_root}.{node.module}" if node.module else package_root]
                else:
                    names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            for name in names:
                if name and name.split(".", 1)[0] in LOCAL_ROOTS:
                    checked += 1
                    if not local_module_exists(name):
                        raise RuntimeError(f"Unresolved local import in {rel}: {name}")
    return checked


def first_run_default(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"\bn_runs\s*=\s*(\d+)\b", text)
    if not m:
        raise RuntimeError(f"Could not locate n_runs default in {path.relative_to(ROOT)}")
    return int(m.group(1))


def main() -> None:
    all_files = files()
    trees = python_trees(all_files)

    supplementary = SLCP_ABLATIONS | SLCP_MNIST | STOPPING
    required = REQUIRED_TENSORS | supplementary | STOPPING_SUPPORT | NON_MNIST_MAIN | MNIST_MAIN
    for rel in sorted(required):
        if not (ROOT / rel).is_file():
            raise RuntimeError(f"Missing required file: {rel}")

    for rel in NON_MNIST_MAIN | SLCP_ABLATIONS:
        got = first_run_default(ROOT / rel)
        if got != 100:
            raise RuntimeError(f"Expected n_runs=100 in {rel}, found {got}")
    for rel in MNIST_MAIN:
        got = first_run_default(ROOT / rel)
        if got != 20:
            raise RuntimeError(f"Expected n_runs=20 in {rel}, found {got}")

    imports = verify_imports(trees)

    bash = shutil.which("bash")
    shell_files = [p for p in all_files if p.suffix == ".sh"]
    if bash:
        for path in shell_files:
            subprocess.run([bash, "-n", path.relative_to(ROOT).as_posix()], cwd=ROOT, check=True)

    forbidden = [p for p in all_files if p.suffix.lower() in FORBIDDEN_FILE_TYPES]
    if forbidden:
        raise RuntimeError("Forbidden generated artifacts: " + ", ".join(str(p.relative_to(ROOT)) for p in forbidden))

    print("Repository verification PASSED")
    print(f"Python files parsed: {len(trees)}")
    print(f"Local imports checked: {imports}")
    print(f"Supplementary entry points verified: {len(supplementary)}")
    print(f"Required tensor assets verified: {len(REQUIRED_TENSORS)}")
    print("Replication defaults: 100 for main simulation studies and SLCP ablations, 20 for main MNIST studies")
    print(f"Shell scripts checked: {len(shell_files)}" if bash else "Shell syntax check skipped: Bash unavailable")


if __name__ == "__main__":
    main()
