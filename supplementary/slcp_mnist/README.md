# Additional SLCP and MNIST experiments

These studies use the [SLCP](../../demo_SLCP/n10.py) and
[MNIST 1 vs 2](../../demo_MNIST/digits_1_vs_2.py) experiment drivers. Each
entry point applies the settings for one study and saves its effective
configuration with the results.

| Study | Entry point | Settings | Default runs |
| --- | --- | --- | ---: |
| SLCP parameter trajectories | [trajectory/run.py](trajectory/run.py) | Parameter estimates across VSM annealing stages with a per-run MLE reference | 100 |
| MNIST classifier width | [mnist_width/main.py](mnist_width/main.py) | Hidden widths 5, 10, 50, and 100 | 20 per width |
| SLCP NLE initialization | [nle_initialization/main.py](nle_initialization/main.py) | Training simulations and prior candidates of (1,000, 5,000), (5,000, 25,000), (50,000, 100,000), and (100,000, 100,000) | 100 per setting |
| SLCP critic capacity | [critic_capacity/main.py](critic_capacity/main.py) | Hidden units and residual blocks of (32, 1), (128, 2), and (256, 3) | 100 per setting |
| SLCP optimization | [optimization/main.py](optimization/main.py) | Learning-rate pairs, paired simulation batch sizes, and 10 inner critic updates | 100 per setting |

The SLCP studies use 10 observations per run. The trajectory study runs VSM
(`Ours`), WME, AdaIMLE, ANND, NBE, and MLE. The NLE initialization, critic
capacity, and optimization studies run VSM and MLE. The MNIST width study
runs VSM with 100 training images split between the two digits and the
full eligible test set.

## Run the experiments

Install the dependencies from the [repository README](../../README.md#install)
and run commands from the repository root. Use a group's `main.py` to
run all its settings or select an individual file in that directory.

```bash
python supplementary/slcp_mnist/trajectory/run.py --gpus cpu --exp-tag trajectory
python supplementary/slcp_mnist/mnist_width/main.py --gpus cpu --exp-tag width
python supplementary/slcp_mnist/nle_initialization/main.py --gpus cpu --exp-tag initialization
python supplementary/slcp_mnist/critic_capacity/critic_hidden_128_blocks_2.py --gpus cpu --exp-tag critic
python supplementary/slcp_mnist/optimization/lr_low.py --gpus cpu --exp-tag lr-low
```

Use `--gpus 0,1` to launch one worker on each of two CUDA devices.
`--mode launch` is the default and merges the results after all workers
finish. A direct `--mode worker` invocation writes one shard. Run
`--mode merge` with the same entry point, `--exp-tag`, and output root
once all shards have finished. To resume, repeat the original launch or
worker command with the same tag and settings.

The MNIST loader can read an existing `mnist_784_local.npz` containing
`X` and `y` through the `MNIST_LOCAL_NPZ` environment variable. Otherwise,
the default download cache is `output/mnist_openml_cache/`.
`--data-home` selects a different cache.

## Find and summarize results

Outputs are written to `output/supplementary/slcp_mnist/` by default.
Select another output location with `--output-root`.

- Trajectory runs go to `trajectory/Exp_SLCP_<tag>/n_data_10/run_<i>/`.
  Each run includes the initial parameters, the VSM parameter history,
  and the final MLE estimate.
- The other studies use `<study>/<setting>_results/Exp_<label>_<tag>/`.
  Each experiment directory contains `experiment_config.json`, the
  individual runs, and the merged metrics and parameters. SLCP studies
  with a baseline and at least one comparison method also record metrics
  relative to MLE.

Install the [analysis dependencies](../../requirements-analysis.txt) to
summarize the default output tree with
[`summarize_results.py`](summarize_results.py). Use `--root` for a different
results location, a single study, or historical `result_2` through `result_5`
directories. To analyze an individual SLCP
trajectory experiment, pass its `n_data_10` directory to
[`trajectory/summarize_history.py`](trajectory/summarize_history.py).

```bash
python supplementary/slcp_mnist/summarize_results.py
python supplementary/slcp_mnist/trajectory/summarize_history.py --input-dir output/supplementary/slcp_mnist/trajectory/Exp_SLCP_trajectory/n_data_10
```

The batch summary writes one row per setting, experiment tag, sample size,
method, and reference. SLCP rows report prior-clipped MAE, MSE, and NRMSE
against the true parameters and, when available, against the MLE. MNIST rows
report the classification metrics. The output includes an index with
missing settings, invalid rows, and differences from the configured run
count.

The trajectory summary pairs each run's initialization and completed
annealing stages with its own MLE. It writes CSVs, a workbook, and PNG and
PDF plots to `theta_history_analysis/` under the specified sample directory.
Use `--output-dir` to write the report elsewhere.
