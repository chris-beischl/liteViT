import subprocess
from argparse import ArgumentParser
from itertools import product

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--models", "-m", nargs="+", type=str)
    parser.add_argument("--datasets", "-d", nargs="+", type=str)
    parser.add_argument("--n_seeds", "-s", type=int)
    parser.add_argument("--logger", "-l", type=str, default="mlflow")
    parser.add_argument("--optimizer", "-o", type=str, default="comparison")
    parser.add_argument("--scheduler", type=str, default="cosine_warmup")
    parser.add_argument("--callbacks", "-c", type=str, default="default")

    args = parser.parse_args()

    for model, dataset, seed in product(
        args.models, args.datasets, range(args.n_seeds)
    ):
        run_name = f"{model}_{dataset}_seed{seed}"

        print(f"INFO: starting run {run_name}")

        cmd = [
            "uv",
            "run",
            "python",
            "train.py",
            f"model={model}",
            f"data={dataset}",
            f"seed={seed}",
            f"logger={args.logger}",
            f"optimizer={args.optimizer}",
            f"scheduler={args.scheduler}",
            f"callbacks={args.callbacks}",
            "logger.experiment_name=comparison",
            f"logger.run_name={run_name}",
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"WARNING: run failed — model={model}, dataset={dataset}, seed={seed}"
            )
