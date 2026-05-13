from datetime import datetime

import hydra
import optuna

from train import run_training

STUDY_NAME = "FashionMNIST-init"


def make_objective(config_name, fixed_overrides: list[str] | None = None):
    if fixed_overrides is None:
        fixed_overrides = []
    print(fixed_overrides)

    def objective(trial: optuna.Trial) -> float:
        # sample params

        depth = trial.suggest_categorical("model.depth", [2, 4, 6, 8])
        embed_dim = trial.suggest_categorical("model.embed_dim", [64, 128, 256])
        head_dim = trial.suggest_categorical("model.head_dim", [16, 32, 64])
        dropout = trial.suggest_categorical("model.dropout", [0.0, 0.1, 0.2])
        drop_path_rate = trial.suggest_categorical(
            "model.drop_path_rate", [0.0, 0.1, 0.2]
        )
        lr = trial.suggest_float("optimizer.lr", low=1e-5, high=1e-3, log=True)
        wd = trial.suggest_float(
            "optimizer.weight_decay", low=1e-5, high=0.01, log=True
        )

        num_heads = embed_dim // head_dim

        overrides = fixed_overrides + [
            f"model.depth={depth}",
            f"model.embed_dim={embed_dim}",
            f"model.num_heads={num_heads}",
            f"model.drop_path_rate={drop_path_rate}",
            f"model.dropout={dropout}",
            f"optimizer.lr={lr}",
            f"optimizer.weight_decay={wd}",
        ]

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        overrides += [f"logger.run_name={STUDY_NAME}_{trial.number}_{now}"]

        cfg = hydra.compose(config_name=config_name, overrides=overrides)
        return run_training(cfg)

    return objective
