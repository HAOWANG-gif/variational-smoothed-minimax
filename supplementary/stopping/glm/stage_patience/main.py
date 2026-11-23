"""Run the two inner-stage patience settings."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from supplementary.stopping.batch import run_batch

if __name__ == "__main__":
    raise SystemExit(run_batch(Path(__file__).resolve().parent))
