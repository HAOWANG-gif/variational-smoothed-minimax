"""Run all settings for the nle initialization study."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from supplementary.slcp_mnist._batch import run_batch


if __name__ == "__main__":
    try:
        raise SystemExit(run_batch(__file__))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
