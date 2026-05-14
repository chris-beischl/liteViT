import glob
import os
import subprocess
from argparse import ArgumentParser

import mlflow
import pandas as pd


def fetch_runs(
    experiment_names: list[str] | None,
    models: list[str] | None,
    datasets: list[str] | None,
    seeds: list[int] | None,
) -> pd.DataFrame:
    runs = mlflow.search_runs(experiment_names=experiment_names)
    if models is not None:
        runs = runs[runs["params.model_cfg"].isin(models)]
    if datasets is not None:
        runs = runs[runs["params.data_cfg"].isin(datasets)]
    # seed is logged as string by MLflow — cast before filtering
    if seeds is not None:
        runs = runs[runs["params.seed"].astype(int).isin(seeds)]
    return runs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--storage", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--experiment-names", "-e", type=str, nargs="+")
    parser.add_argument("--models", "-m", type=str, nargs="+")
    parser.add_argument("--datasets", "-d", type=str, nargs="+")
    parser.add_argument("--seeds", "-s", type=int, nargs="+")

    args = parser.parse_args()

    mlflow.set_tracking_uri(uri=args.storage)

    runs = fetch_runs(args.experiment_names, args.models, args.datasets, args.seeds)
    checkpoint_paths = runs["params.checkpoint_path"]

    for checkpoint_path in checkpoint_paths:
        matches = glob.glob(f"{checkpoint_path}/last-*.ckpt")
        if not matches:
            print(f"WARNING: no checkpoint found at {checkpoint_path}")
            continue
        checkpoint = matches[0]
        result = subprocess.run(
            ["uv", "run", "python", "eval.py", "--checkpoint", checkpoint]
        )
        if result.returncode != 0:
            print(f"WARNING: eval failed for {checkpoint}")

    # re-query after eval so test metrics logged by eval.py are included
    runs = fetch_runs(args.experiment_names, args.models, args.datasets, args.seeds)

    result_cols = [
        "params.model_cfg",
        "params.data_cfg",
        "params.seed",
        "params.model/depth",
        "params.model/embed_dim",
        "params.model/num_heads",
    ]
    # discover test metric columns dynamically rather than hardcoding names
    test_cols = [c for c in runs.columns if c.startswith("metrics.test_")]
    df = runs[result_cols + test_cols]

    experiments = "-".join(args.experiment_names) if args.experiment_names else "all"
    models = "-".join(args.models) if args.models else "all"
    datasets = "-".join(args.datasets) if args.datasets else "all"
    filename = f"results/{experiments}_{models}_{datasets}.csv"

    os.makedirs("results", exist_ok=True)
    df.to_csv(filename, index=False)

    # aggregate per model/dataset combination for the results table
    summary = df.groupby(["params.model_cfg", "params.data_cfg"])[test_cols].agg(
        ["mean", "std"]
    )
    summary_filename = f"results/summary_{experiments}_{models}_{datasets}.csv"
    summary.to_csv(summary_filename)
    print(summary)
