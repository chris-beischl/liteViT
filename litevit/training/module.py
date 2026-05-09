import torch
import torch.nn as nn
import lightning as L

from torchmetrics import MetricCollection

from omegaconf import DictConfig
import hydra


class ClassificationModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig | None = None,
        metrics: MetricCollection | None = None,
    ):
        super().__init__()
        self.model = model
        self.loss = loss

        
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        
        self.train_metrics = metrics.clone(prefix="train_") if metrics is not None else None
        self.val_metrics = metrics.clone(prefix="val_") if metrics is not None else None
        self.test_metrics = metrics.clone(prefix="test_") if metrics is not None else None
        
        self.save_hyperparameters(
            ignore=["model", "loss", "metrics"]
        )

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_cfg)(self.model.parameters())
        if self.scheduler_cfg is not None:
            scheduler = hydra.utils.instantiate(self.scheduler_cfg)(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }
            }
        return optimizer
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int): 
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        
        # update all training metrics
        if self.train_metrics is not None:
            self.train_metrics.update(y_hat, y)
        
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # update all validation metrics
        if self.val_metrics is not None:
            self.val_metrics.update(y_hat, y)
        
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        
        # update all test metrics
        if self.test_metrics is not None:
            self.test_metrics.update(y_hat, y)
            
    def on_train_epoch_end(self):
        if self.train_metrics is not None:
            train_metric_results = self.train_metrics.compute()
            self.log_dict(train_metric_results, on_step=False, on_epoch=True)
            self.train_metrics.reset()
            
    def on_validation_epoch_end(self):
        if self.val_metrics is not None:
            val_metric_results = self.val_metrics.compute()
            self.log_dict(val_metric_results, on_step=False, on_epoch=True, prog_bar=True)
            self.val_metrics.reset()
            
    def on_test_epoch_end(self):
        if self.test_metrics is not None:
            test_metric_results = self.test_metrics.compute()
            self.log_dict(test_metric_results, on_step=False, on_epoch=True)
            self.test_metrics.reset()