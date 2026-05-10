import hydra

import lightning as L
from torchmetrics import MetricCollection
from omegaconf import OmegaConf

from litevit.conf import register_configs
from litevit.training.module import ClassificationModule

register_configs()


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg):
    L.seed_everything(cfg.seed, workers=True)

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    loss = hydra.utils.instantiate(cfg.loss)
    optimizer_cfg = cfg.optimizer
    scheduler_cfg = cfg.scheduler
    metrics = (
        MetricCollection({
            name: hydra.utils.instantiate(metric_cfg)
            for name, metric_cfg in cfg.metrics.items()
        })
        if cfg.metrics is not None
        else None
    )

    module = ClassificationModule(
        model=model,
        loss=loss,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
        metrics=metrics,
    )

    data = hydra.utils.instantiate(cfg.data)
    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.callbacks.callbacks]
    logger = hydra.utils.instantiate(cfg.logger) if cfg.logger is not None else None
    
    if hasattr(logger, 'log_hyperparams'): 
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )
    trainer.fit(module, datamodule=data)
    
    # return validation accuracy for optuna to use for hparam selection
    return trainer.callback_metrics["val_accuracy"].item()

if __name__ == "__main__":
    main()
