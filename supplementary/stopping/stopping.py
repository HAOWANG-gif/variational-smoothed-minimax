"""Dependency-free accounting for the optional paper-level stopping rule.

The controller observes completed stages. It never performs simulations or
decides whether an individual minimax iteration is small enough to stop.
"""

import math
from numbers import Integral, Real


def validate_stop_config(train_config):
    """Validate active stopping settings, leaving budget-mode extras inactive."""
    mode = train_config.get("overall_stop_mode", "budget")
    if mode not in ("budget", "paper"):
        raise ValueError("overall_stop_mode must be 'budget' or 'paper'.")
    if mode == "budget":
        return mode, None, None, None

    for key in ("delta_stop", "K_stop", "epsilon_min"):
        if key not in train_config:
            raise ValueError(f"Paper stopping requires an explicit {key}.")
    for key in ("delta_stop", "epsilon_min"):
        value = train_config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{key} must be a finite positive number.")
    k_stop = train_config["K_stop"]
    if isinstance(k_stop, bool) or not isinstance(k_stop, Integral) or k_stop <= 0:
        raise ValueError("K_stop must be a positive integer.")
    return mode, float(train_config["delta_stop"]), int(k_stop), float(train_config["epsilon_min"])


class StageStoppingController:
    """Record raw stage changes, then anneal epsilon and evaluate all criteria.

    ``epsilon`` is the bandwidth to use for the next solve. No epsilon or theta
    criterion is evaluated before the first stage, even when epsilon_initial is
    already below epsilon_min. An exhausted initialization budget skips solving.
    Public histories/reports contain only ordinary Python values.
    """

    def __init__(self, train_config, epsilon_initial):
        self.mode, self.delta_stop, self.K_stop, self.epsilon_min = validate_stop_config(train_config)
        self.gamma = train_config.get("gamma", 0.5)
        self.simulation_budget = int(train_config["n_sim"])
        self.n_init_sim = int(train_config.get("n_init_sim", 0))
        if self.simulation_budget < self.n_init_sim:
            raise ValueError("n_sim must be at least the initialization simulation count n_init_sim.")
        self.epsilon_initial = float(epsilon_initial)
        self.epsilon = self.epsilon_initial
        self.consecutive_small_stages = 0
        self.theta_small_stage_count = 0
        self.inner_early_stop_count = 0
        self.stage_history = []
        self.termination_report = {}
        self._update_report(
            n_sim=self.n_init_sim, stop_theta=False, stop_epsilon=False,
            stop_budget=self.n_init_sim >= self.simulation_budget,
        )

    @property
    def should_stop(self):
        report = self.termination_report
        return report["stop_theta"] or report["stop_epsilon"] or report["stop_budget"]

    def _update_report(self, n_sim, stop_theta, stop_epsilon, stop_budget, fallback_reason=None):
        if stop_theta:
            reason = "theta_convergence"
        elif stop_epsilon:
            reason = "epsilon_min"
        elif stop_budget:
            reason = "simulation_budget"
        else:
            reason = fallback_reason
        # Mutate the existing dict so the trainer can expose the same object.
        self.termination_report.update({
            "overall_stop_mode": self.mode,
            "stop_reason": reason,
            "stop_theta": bool(stop_theta),
            "stop_epsilon": bool(stop_epsilon),
            "stop_budget": bool(stop_budget),
            "n_stages": len(self.stage_history),
            "n_sim": int(n_sim),
            "n_init_sim": self.n_init_sim,
            "training_n_sim": int(n_sim) - self.n_init_sim,
            "simulation_budget": self.simulation_budget,
            "epsilon_initial": self.epsilon_initial,
            "epsilon_final": float(self.epsilon),
            "gamma": float(self.gamma),
            "delta_stop": self.delta_stop,
            "K_stop": self.K_stop,
            "epsilon_min": self.epsilon_min,
            "consecutive_small_stages": self.consecutive_small_stages,
            "theta_small_stage_count": self.theta_small_stage_count,
            "inner_early_stop_count": self.inner_early_stop_count,
            "inner_iterations_saved_total": sum(row["inner_iterations_saved"] for row in self.stage_history),
        })

    def record_stage(
        self, theta_start, theta_end, n_sim, inner_iterations,
        inner_early_stop_hit=False, inner_iterations_saved=0,
        inner_stop_reason="minimax_iters",
    ):
        """Observe one completed solve and return its stage diagnostic record."""
        start = tuple(float(value) for value in theta_start)
        end = tuple(float(value) for value in theta_end)
        if not start or len(start) != len(end):
            raise ValueError("theta_start and theta_end must be nonempty vectors of equal length.")
        changes = [abs(after - before) for before, after in zip(start, end)]
        # max([finite, nan]) can hide NaN; an invalid parameter must never
        # become evidence for stage convergence merely by its vector position.
        theta_change_inf = math.nan if any(math.isnan(value) for value in changes) else max(changes)
        if self.mode == "paper":
            if theta_change_inf <= self.delta_stop:
                self.consecutive_small_stages += 1
                self.theta_small_stage_count += 1
            else:
                self.consecutive_small_stages = 0
        self.inner_early_stop_count += int(bool(inner_early_stop_hit))

        epsilon_used = self.epsilon
        self.epsilon = self.gamma * epsilon_used
        stop_theta = self.mode == "paper" and self.consecutive_small_stages >= self.K_stop
        stop_epsilon = self.mode == "paper" and self.epsilon <= self.epsilon_min
        stop_budget = n_sim >= self.simulation_budget
        row = {
            "stage": len(self.stage_history) + 1,
            "epsilon_used": float(epsilon_used),
            "epsilon_next": float(self.epsilon),
            "theta_change_inf": float(theta_change_inf),
            "consecutive_small_stages": self.consecutive_small_stages,
            "n_sim": int(n_sim),
            "inner_iterations": int(inner_iterations),
            "inner_early_stop_hit": bool(inner_early_stop_hit),
            "inner_iterations_saved": int(inner_iterations_saved),
            "inner_stop_reason": inner_stop_reason,
            "stop_theta": bool(stop_theta),
            "stop_epsilon": bool(stop_epsilon),
            "stop_budget": bool(stop_budget),
        }
        self.stage_history.append(row)
        self._update_report(n_sim, stop_theta, stop_epsilon, stop_budget)
        return row

    def finalize(self, n_sim, fallback_reason=None):
        """Finalize accounting, including a solver that could do no iterations."""
        report = self.termination_report
        self._update_report(
            n_sim, report["stop_theta"], report["stop_epsilon"],
            n_sim >= self.simulation_budget, fallback_reason=fallback_reason,
        )
        return self.termination_report
