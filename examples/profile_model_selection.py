# examples/profile_model_selection.py
from __future__ import annotations

import time
from pathlib import Path
import sys

# Make src/ visible so we can import dmm and the example modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Make examples/ visible so we can import experiment scripts as modules
EXAMPLES_DIR = PROJECT_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import numpy as np

# Import the serial and parallel experiment functions + N_GRID
import experiment_model_selection as ser
import experiment_model_selection_parallel as par


def profile_setting(
    setting: str,
    n_rep: int = 50,
    n_jobs: int = -1,
) -> None:
    """
    Profile serial vs parallel for a single setting ("well", "eps", "skew").

    We use a reduced n_grid and n_rep so it runs in reasonable time.
    """
    # Use a smaller grid for profiling (e.g., first 5 sample sizes)
    n_grid_small = ser.N_GRID[:5]

    print(f"\n=== Setting: {setting!r} | n_rep={n_rep} | n_grid={n_grid_small.tolist()} ===")

    # --- Serial ---
    t0 = time.perf_counter()
    res_ser = ser.run_experiment(
        setting=setting,
        n_grid=n_grid_small,
        n_rep=n_rep,
        k_max=ser.K_MAX,
        seed=123,
    )
    t1 = time.perf_counter()
    time_ser = t1 - t0

    # --- Parallel ---
    t0 = time.perf_counter()
    res_par = par.run_experiment_parallel(
        setting=setting,
        n_grid=n_grid_small,
        n_rep=n_rep,
        k_max=par.K_MAX,
        seed=123,
        n_jobs=n_jobs,
    )
    t1 = time.perf_counter()
    time_par = t1 - t0

    # --- Print comparison ---
    print(f"Serial   time: {time_ser:7.2f} s")
    print(f"Parallel time: {time_par:7.2f} s  (n_jobs={n_jobs})")
    if time_par > 0:
        print(f"Speedup (serial / parallel): {time_ser / time_par:5.2f}x")

    # Optional sanity check: compare results roughly
    for key in ["prop_correct_aic", "prop_correct_bic", "prop_correct_dic"]:
        diff = np.max(np.abs(res_ser[key] - res_par[key]))
        print(f"max |{key}_serial - {key}_parallel| = {diff:.3f}")


def main():
    # You can adjust n_rep and n_jobs here
    n_rep = 30      # fewer reps for quick profiling
    n_jobs = -1     # use all cores; set to e.g. 4 if you want

    for setting in ["well", "eps", "skew"]:
        profile_setting(setting, n_rep=n_rep, n_jobs=n_jobs)


if __name__ == "__main__":
    main()