import hydra
import optuna
from datetime import datetime

from train import run_training

# Unique name for this study — used by Optuna for storage and resuming
STUDY_NAME = "my-sweep"


def make_objective(config_name: str, fixed_overrides: list[str] | None = None):
    """Objective factory for Optuna.

    Args:
        config_name: Hydra config name (e.g. "train")
        fixed_overrides: Hydra overrides applied to every trial (e.g. ["data=fashion_mnist"])

    Returns:
        Optuna objective function that returns the metric to optimize.
    """
    if fixed_overrides is None:
        fixed_overrides = []

    def objective(trial: optuna.Trial) -> float:
        # --- Define search space ---
        lr = trial.suggest_float("optimizer.lr", 1e-5, 1e-3, log=True)
        depth = trial.suggest_categorical("model.depth", [2, 4, 6, 8])
        # add more parameters here...

        # --- Build overrides ---
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        overrides = fixed_overrides + [
            f"model.depth={depth}",
            f"optimizer.lr={lr}",
            f"logger.run_name={STUDY_NAME}_{trial.number}_{now}",
        ]

        cfg = hydra.compose(config_name=config_name, overrides=overrides)
        return run_training(cfg)

    return objective
