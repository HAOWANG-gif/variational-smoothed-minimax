"""Run the SLCP overall-stopping sensitivity grid."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from supplementary.stopping.sensitivity import run_sensitivity

if __name__ == "__main__":
    entrypoint = Path(__file__).parent / "overall_stopping" / "paper_stopping.py"
    raise SystemExit(run_sensitivity(entrypoint))
