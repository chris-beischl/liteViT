from dataclasses import dataclass, field
from typing import Any
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# @dataclass
# class ModelConfig:
#     _target_: str = "microvit.models.vit.build_vit"
#     embed_dim: int = MISSING
#     patch_embed_cfg: dict = field(default_factory=dict)
#     attention_cfg: dict = field(default_factory=dict)
#     block_cfg: dict = field(default_factory=dict)
#     num_classes: int = MISSING 
#     num_heads: int = MISSING
#     depth: int = MISSING
#     dropout: float = 0.0
#     drop_path_rate: float = 0.0

# @dataclass
# class TransformConfig:
#     _target_: str = MISSING

# @dataclass
# class DataConfig:
#     _target_: str = MISSING
#     batch_size: int = MISSING
#     num_workers: int = 0
#     pin_memory: bool = False
#     split_perc: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
#     data_dir: str = "./data"
#     train_transform: TransformConfig | None = None
#     eval_transform: TransformConfig | None = None
    
# @dataclass
# class LossConfig:
#     _target_: str = MISSING
    
# @dataclass
# class OptimizerConfig:
#     _target_: str = MISSING
#     _partial_: bool = True
    
# @dataclass
# class SchedulerConfig:
#     _target_: str = MISSING
#     _partial_: bool = True
    
# @dataclass
# class MetricsConfig:
#     _target_: str = "torchmetrics.MetricCollection"
    
# @dataclass
# class CallbacksConfig:
#     callbacks: list = field(default_factory=list)

# @dataclass  
# class TrainerConfig:
#     _target_: str = "lightning.Trainer"
#     accelerator: str = "auto"
#     devices: int = 1
#     strategy: str = "auto"
#     max_epochs: int = 10
#     log_every_n_steps: int = 10
    
# @dataclass
# class LoggerConfig:
#     _target_: str = MISSING
#     experiment_name: str = MISSING
#     tracking_uri: str = "./mlruns"

# @dataclass
# class TrainConfig: 
#     model: ModelConfig = MISSING
#     data: DataConfig = MISSING
#     loss: LossConfig = MISSING
#     optimizer: OptimizerConfig = MISSING
#     scheduler: SchedulerConfig | None = None
#     metrics: MetricsConfig | None = None
#     callbacks: CallbacksConfig | None = None
#     trainer: TrainerConfig = MISSING
#     logger: LoggerConfig | None = None
#     seed: int = 0

@dataclass
class TrainConfig:
    model: Any = MISSING
    data: Any = MISSING
    optimizer: Any = MISSING
    scheduler: Any = None
    metrics: Any = None
    callbacks: Any = None
    trainer: Any = MISSING
    logger: Any = None
    loss: Any = MISSING
    seed: int = 0
   
def register_configs():  
    cs = ConfigStore.instance()
    cs.store(name="train_config", node=TrainConfig)