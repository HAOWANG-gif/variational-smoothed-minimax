"""GLM stopping sensitivity: all cases from the shared overall-stopping grid."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from supplementary.stopping.sensitivity import run_sensitivity

if __name__ == "__main__":
    raise SystemExit(run_sensitivity(__file__))
