"""Run experiment entries sequentially, forwarding their worker/launch/merge arguments."""
from pathlib import Path
import subprocess
import sys


def run_batch(directory, arguments=None):
    directory = Path(directory).resolve()
    scripts = sorted(path for path in directory.glob("*.py")
                     if path.name not in {"main.py", "baseline.py"}
                     and not path.name.startswith("_"))
    if not scripts:
        raise ValueError(f"No experiment entries in {directory}")
    failures = []
    for index, script in enumerate(scripts, 1):
        print(f"\n[{index}/{len(scripts)}] {script}", flush=True)
        result = subprocess.run([sys.executable, "-u", str(script),
                                 *(sys.argv[1:] if arguments is None else arguments)],
                                check=False)
        if result.returncode:
            failures.append((script.name, result.returncode))
    if failures:
        print(f"Failed experiment processes: {failures}", flush=True)
    return int(bool(failures))
