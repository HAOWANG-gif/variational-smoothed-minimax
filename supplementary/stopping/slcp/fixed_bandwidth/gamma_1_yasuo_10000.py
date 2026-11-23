"""Fixed bandwidth at 1/10000 of the default initial bandwidth."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from supplementary.stopping._driver import run_experiment

train_configs = {"Ours": {"gamma": 1.0, "yasuo": 10000, "overall_stop_mode": "budget"}}

if __name__ == "__main__":
    run_experiment(__file__, train_configs)
