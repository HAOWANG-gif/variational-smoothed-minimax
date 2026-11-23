"""Run the nle initialization supplementary experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from supplementary.slcp_mnist._runner import run


if __name__ == "__main__":
    run(__file__, "nle_initialization", nle=(5_000, 25_000))
