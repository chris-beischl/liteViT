import importlib
from argparse import ArgumentParser
import hydra
import optuna


if __name__ == "__main__": 
    parser = ArgumentParser()
    parser.add_argument("--sweep", "-s", type=str)
    parser.add_argument("--config", "-c", type=str, default="train")
    parser.add_argument("--override", "-o", action="append")
    parser.add_argument("--n_trials", "-n", type=int, default=20)
    parser.add_argument("--direction", "-d", type=str, default="maximize")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_studies.db")

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
    with hydra.initialize(config_path="configs/", version_base=None): 
        objective = sweep_module.make_objective(args.config, args.override)
        print(objective)
        study.optimize(objective, n_trials=args.n_trials)
        
    
