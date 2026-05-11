from typing import Any

import torch
import lightning as L
from torchmetrics import MetricCollection

import hydra
from omegaconf import OmegaConf

from argparse import ArgumentParser

from litevit.training import ClassificationModule

def eval(ckpt):
    cfg = OmegaConf.create(ckpt.get('cfg'))
    run_id = ckpt.get('run_id')
    
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

    module.load_state_dict(ckpt['state_dict'])

    data = hydra.utils.instantiate(cfg.data)
    logger = hydra.utils.instantiate(cfg.logger, run_id=run_id) if cfg.logger is not None else None

    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger
    )
    trainer.test(module, datamodule=data)
    return trainer.callback_metrics


if __name__ == "__main__": 
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', '-c', type=str)
    
    args = parser.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False) 
    eval(ckpt)

# ckpt = torch.load("checkpoints/2026-05-11/15-08-42/epoch=17-step=3600.ckpt", map_location="cpu", weights_only=False)
# print(ckpt.keys())
# print(ckpt.get("run_id"))
# print(OmegaConf.to_yaml(ckpt.get("cfg")))