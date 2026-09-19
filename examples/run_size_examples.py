from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fionna import fit_fionna, generate_linear_null, generate_shared_profile_null


SETTINGS = {
    "linear_n400": (generate_linear_null, 5, 20260710),
    "shared_profile_n400": (generate_shared_profile_null, [2, 2, 2, 2], 20267010),
}


def run_replication(task):
    name, replication = task
    generator, K, base_seed = SETTINGS[name]
    seed = base_seed + replication
    data = generator(seed=seed)
    result = fit_fionna(data, K, seed + 1_000_000)
    return {
        "setting": name,
        "replication": replication,
        "theta0": data.theta0,
        "alpha0": data.alpha0,
        "delta0": data.delta0,
        **{key: result[key] for key in (
            "theta_hat", "alpha_hat", "delta_hat",
            "se_theta", "se_alpha", "se_delta", "elapsed_sec"
        )},
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cutoff = norm.ppf(0.975)
    for name, group in raw.groupby("setting", sort=False):
        row = {"setting": name, "replications": len(group)}
        for effect in ("theta", "alpha", "delta"):
            estimate = group[f"{effect}_hat"]
            truth = group[f"{effect}0"]
            standard_error = group[f"se_{effect}"]
            covered = (estimate - cutoff * standard_error <= truth) & (
                truth <= estimate + cutoff * standard_error
            )
            row[f"{effect}_bias"] = (estimate - truth).mean()
            row[f"{effect}_mcsd"] = estimate.std(ddof=1)
            row[f"{effect}_ase"] = standard_error.mean()
            row[f"{effect}_coverage"] = covered.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    tasks = [(name, replication) for name in SETTINGS
             for replication in range(args.replications)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        raw = pd.DataFrame(pool.map(run_replication, tasks))
    summary = summarize(raw)
    print(summary.to_string(index=False))
    if args.output:
        summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
