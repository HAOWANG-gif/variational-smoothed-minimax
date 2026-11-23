"""Run the settings in one supplementary experiment group."""

from pathlib import Path
import re
import subprocess
import sys


def run_batch(current_file):
    directory = Path(current_file).resolve().parent
    scripts = sorted(
        (path for path in directory.glob("*.py")
         if path.name != "main.py" and not path.name.startswith("_")),
        key=lambda path: [int(part) if part.isdecimal() else part
                          for part in re.split(r"(\d+)", path.name)],
    )
    failures = []
    for index, script in enumerate(scripts, start=1):
        print(f"[{index}/{len(scripts)}] {script.name}", flush=True)
        try:
            result = subprocess.run([sys.executable, "-u", str(script), *sys.argv[1:]], check=False)
        except OSError as error:
            failures.append((script.name, str(error)))
        else:
            if result.returncode:
                failures.append((script.name, f"exit code {result.returncode}"))
    for name, reason in failures:
        print(f"{name} failed ({reason})", file=sys.stderr)
    return bool(failures)
