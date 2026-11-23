# Variational Smoothed Minimax (VSM)

Research code for **Maximum Likelihood Estimation without Likelihoods: a Variational Smoothed Minimax Approach**.

VSM estimates parameters of a simulator when its likelihood is unavailable. The implementation smooths the empirical distribution and uses a critic to estimate the divergence from the simulator distribution during minimax optimization. The experiment outputs label VSM as `Ours`.

[Paper-to-code map](docs/PAPER_TO_CODE.md) · [Supplementary experiments](docs/SUPPLEMENTARY_EXPERIMENTS.md) · [Reproducibility](docs/REPRODUCIBILITY.md)

## Install

Python 3.12 is the recommended version. Create the environment with Conda or install the training dependencies in a virtual environment.

```bash
conda env create -f environment.yml
conda activate vsm
```

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` or use `.\.venv\Scripts\python.exe` directly. For GPU runs, use a PyTorch build compatible with your CUDA environment. To analyze saved outputs on a machine without the training stack, install `requirements-analysis.txt`.

## Main experiments

| Experiment | Entry points | Default repetitions |
| --- | --- | ---: |
| SLCP | [10 observations](demo_SLCP/n10.py), [100 observations](demo_SLCP/n100.py) | 100 each |
| Bernoulli GLM | [10 observations](demo_GLM/n10.py), [100 observations](demo_GLM/n100.py) | 100 each |
| Univariate g-and-k | [10 observations](demo_gk/n10.py), [100 observations](demo_gk/n100.py) | 100 each |
| Bivariate g-and-k | [10 observations](demo_BGK/n10.py), [100 observations](demo_BGK/n100.py) | 100 each |
| MNIST binary classification | [1 vs 2](demo_MNIST/digits_1_vs_2.py), [4 vs 9](demo_MNIST/digits_4_vs_9.py) | 20 per task |

Run commands from the project root. For a one-repetition CPU check using the SLCP maximum-likelihood reference, use a fresh experiment tag.

```bash
python demo_SLCP/n10.py --mode worker --gpu-id cpu --shard-id 0 --num-shards 1 --methods MLE --n-runs 1 --exp-tag slcp-smoke
python demo_SLCP/n10.py --mode merge --exp-tag slcp-smoke
```

For the full study, use the entry point's default methods and replication count. To use a GPU, select its index with `--gpu-id` in worker mode. See the [SLURM launcher](scripts/slurm/README.md) for cluster submission.

The MNIST loader accepts a local `.npz` containing `X` and `y` via the `MNIST_LOCAL_NPZ` environment variable. It can also load and cache MNIST from OpenML.

## Supplementary experiments

| Study | Code |
| --- | --- |
| Initial bandwidth, annealing factor, and simulation budget | [SLCP ablation entry points](docs/SUPPLEMENTARY_EXPERIMENTS.md#slcp-ablation-drivers) |
| SLCP trajectories, MNIST hidden width, NLE initialization, critic capacity, and optimization settings | [Additional SLCP and MNIST experiments](supplementary/slcp_mnist/README.md) |
| Fixed bandwidth, within-stage patience, and overall stopping in SLCP and GLM | [SLCP and GLM stopping experiments](supplementary/stopping/README.md) |

The source includes the fixed GLM tensors. Experiment runs generate their own results and checkpoints. The supplementary guides explain where to find and summarize them.

## Verify

```bash
python scripts/verify_repository.py
python supplementary/stopping/slcp/main.py --list-configs
python supplementary/stopping/glm/main.py --list-configs
```

See [Reproducibility](docs/REPRODUCIBILITY.md) for the analysis tests and the distinction between a quick check and a full experimental run.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite the software. Cite the associated paper when using the method or discussing the experiments.

Usage terms are in [LICENSE](LICENSE).
