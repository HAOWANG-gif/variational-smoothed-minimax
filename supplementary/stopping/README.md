# SLCP and GLM stopping experiments

These supplementary experiments use the simulators, priors, critic networks,
initialization and evaluation from the [SLCP](../../demo_SLCP/n10.py) and
[GLM](../../demo_GLM/n10.py) demos. Each setting defaults to 100 repetitions
with 10 observations per repetition. The drivers run the VSM estimator
(`Ours` in CSVs) followed by `MLE` to preserve the random-number sequence.
The GLM simulator returns 100-dimensional Bernoulli observations with a
10-dimensional parameter. SLCP uses 8-dimensional observations with a
5-dimensional parameter.

| Experiment | Entry under either model directory | Override to the model demo |
| --- | --- | --- |
| Fixed bandwidth | `fixed_bandwidth/main.py` | Sets `gamma=1` and tries `yasuo` values of 1, 100 and 10000. |
| Within-stage patience | `stage_patience/main.py` | Tries `early_stop_patience` values of 300 and 600. |
| Overall stopping | `main.py` or `overall_stopping/paper_stopping.py` | Sweeps `delta_stop`, `K_stop`, `epsilon_min` and `n_sim`. |

The fixed-bandwidth runs freeze the bandwidth at 1, 1/100 or 1/10000 of
the default initial bandwidth. With the default 10 observations, the
initial bandwidth before this scaling is the largest marginal sample
standard deviation, with a floor of `1e-4`. Fixed-bandwidth and within-stage
patience runs use the simulation budget as the overall stopping criterion.
At the 600-iteration stage limit, a patience-600 hit on the last iteration
saves zero iterations. Reports count patience hits and iterations saved
separately.

## Inspect the overall-stopping grid

From the repository root, list the cases without initializing a model.

```bash
python supplementary/stopping/slcp/main.py --list-configs
python supplementary/stopping/glm/main.py --list-configs
```

The [shared grid](overall_stopping_grid.json) contains 27 cases per model.
To run a subset, pass `--config path/to/grid.json` with nonempty lists for the
same four keys. Existing `add3_config.json` grid files can be used as-is.
`delta_stop` and `epsilon_min` take finite positive numbers, and `K_stop` and
`n_sim` take positive integers. Each list must contain distinct values.
`n_sim` includes 10000 initialization simulations, so its minimum is 10000.
The checked-in grid tests three values each for `delta_stop`, `K_stop` and
`epsilon_min` at a total budget of 500000 simulations.

## Run an experiment

Install the [project dependencies](../../requirements.txt), then run from the
repository root. Both model entry points run the overall-stopping grid by
default. A full grid is computationally expensive.

```bash
python supplementary/stopping/slcp/main.py --mode launch --gpus cpu --workers-per-gpu 25 --exp-tag V3
python supplementary/stopping/glm/main.py --mode launch --gpus cpu --workers-per-gpu 25 --exp-tag V3
python supplementary/stopping/glm/stage_patience/main.py --mode launch --gpus cpu --exp-tag V3
python supplementary/stopping/slcp/fixed_bandwidth/gamma_1_yasuo_100.py --mode launch --gpus cpu --exp-tag V3
```

On Slurm, use the [shared CPU launcher](../../scripts/slurm/README.md).
The commands below request 50 CPU cores per model for 25 workers using two
threads each. Add your cluster's account, partition and environment settings
at submission.

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
sbatch --cpus-per-task=50 scripts/slurm/cpu_job.sh supplementary/stopping/slcp/main.py --gpus cpu --exp-tag V3
sbatch --cpus-per-task=50 scripts/slurm/cpu_job.sh supplementary/stopping/glm/main.py --gpus cpu --exp-tag V3
```

Use the same `--exp-tag`, grid and shard count to resume an interrupted
launch. Each case is named by its parameter values, so changing grid order
does not change the location of completed cases. To run a shard and merge it
manually, use the original experiment tag and grid.

```bash
python supplementary/stopping/glm/main.py --mode worker --gpu-id cpu --shard-id 0 --num-shards 1 --exp-tag V3
python supplementary/stopping/glm/main.py --mode merge --exp-tag V3
```

## Outputs and reporting

Each setting writes to its own `*_results/` directory. For the overall grid,
the path under each model directory is
`overall_stopping/paper_stopping_results/<case>/Exp_<MODEL>_<tag>/`.
Each case has a separate NLE-MAP initialization cache. Saved
`experiment_config.json` files record the exact case and model overrides.
The driver uses completed `run_records.json` files when resuming or rebuilding
shard summaries.

To resume an earlier overall-stopping run, copy its `paper_stopping_results/`
directory into the matching model's `overall_stopping/` directory before
running with the original grid, experiment tag and shard count. Use the same
software environment and device type when reusing initialization caches.

At the end of each stage, the trainer compares the raw parameter vector to
its value at the start of the stage using the infinity norm. It counts
consecutive changes at most `delta_stop`, resets that count after a larger
change and multiplies the bandwidth by `gamma`. Overall stopping checks
whether `K_stop` consecutive stages met that threshold, the updated bandwidth
is at most `epsilon_min`, or the simulation budget is exhausted. The final
simulation batch is reduced when necessary to stay within the budget.

| File | Contents |
| --- | --- |
| `Ours/stage_history.csv` | Stage-by-stage parameter changes, bandwidth and inner stopping diagnostics. |
| `Ours/stopping_summary.json` | Termination details for one repetition. |
| `vsm_stopping_runs.csv` | One row per repetition. |
| `vsm_stopping_counts.csv` | Counts by observation size. Condition flags may overlap. Final reasons are exclusive and prioritize parameter stability, bandwidth, then budget. |

To summarize an overall-stopping grid, use the model's default results
directory or pass `--root` to read results from another location, including
earlier `view3/<model>/add3/paper_stopping_results` directories.

```bash
python supplementary/stopping/summarize_results.py --model SLCP --exp-tag V3
python supplementary/stopping/summarize_results.py --model GLM --root /path/to/paper_stopping_results --output glm_stopping_summary.csv
```

Install the [analysis dependencies](../../requirements-analysis.txt) for
NumPy and pandas. The script reads parameters from saved experiment
configurations and writes `stopping_summary.csv` under the results root unless
`--output` is supplied. Each row reports the simulation count for VSM and
prior-clipped MAE, MSE and NRMSE for VSM versus truth, VSM versus MLE and MLE
versus truth. Standard deviations use `ddof=1` and are zero for a single
repetition. The same output reports overlapping condition counts and
exclusive final reasons. If merged metrics are missing, the script reads
shard metrics and can reconstruct MLE pairings by observation size and run ID.
Incomplete or invalid rows are annotated in `status` and `notes` and produce a
nonzero exit status.

## Verify the reporting workflow

```bash
python -B -m unittest discover -s supplementary/stopping/tests -v
```

The tests use synthetic CSVs and exercise the real summary entry point
without running simulation or training.
