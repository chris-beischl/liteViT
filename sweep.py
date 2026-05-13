import importlib
from argparse import ArgumentParser

import hydra
import optuna


def try_cast(value: str) -> int | float | str:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sweep", "-s", type=str)
    parser.add_argument("--config", "-c", type=str, default="train")
    parser.add_argument("--override", "-o", action="append")
    parser.add_argument("--n_trials", "-n", type=int, default=20)
    parser.add_argument("--direction", "-d", type=str, default="maximize")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_studies.db")
    parser.add_argument("--enqueue", "-e", action="append")

    args = parser.parse_args()

    sweep_module = importlib.import_module(f"scripts.sweeps.{args.sweep}")
    print(sweep_module)
    # todo set right arguments for study
    study = optuna.create_study(
        study_name=sweep_module.STUDY_NAME,
        direction=args.direction,
        storage=args.storage,
        load_if_exists=True,
    )

    if args.enqueue is not None:
        pairs = [pair.split("=") for pair in args.enqueue]
        for idx, pair in enumerate(pairs):
            if len(pair) != 2:
                raise ValueError(
                    f"Invalid enqueue argument '{args.enqueue[idx]}': "
                    "expected 'key=value' format"
                )
        trial_params = {pair[0]: try_cast(pair[1]) for pair in pairs}
        study.enqueue_trial(params=trial_params)
        print(f"--enqueue specified: overriding n_trials to 1 (was {args.n_trials})")
        args.n_trials = 1

    with hydra.initialize(config_path="configs/", version_base=None):
        objective = sweep_module.make_objective(args.config, args.override)
        print(objective)
        study.optimize(objective, n_trials=args.n_trials)
