"""Stage-internal early-stopping patience of 300; other n10 settings unchanged."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from supplementary.stopping._driver import run_experiment

train_configs = {"Ours": {"early_stop_patience": 300, "overall_stop_mode": "budget"}}

if __name__ == "__main__":
    run_experiment(__file__, train_configs)
