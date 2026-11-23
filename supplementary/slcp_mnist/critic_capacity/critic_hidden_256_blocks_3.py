"""Run the critic capacity supplementary experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from supplementary.slcp_mnist._runner import run


if __name__ == "__main__":
    run(__file__, "critic_capacity", critic=(256, 3))
