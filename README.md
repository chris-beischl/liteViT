# liteViT

A from-scratch Vision Transformer implementation — no timm, no pretrained weights. Built to run and train on a MacBook Pro (Apple Silicon, MPS backend) and designed for rapid experimentation with depth, learning rate, drop path, and custom attention or transformer block architectures.

## Setup

```bash
uv sync
```

## Training

```bash
uv run python train.py
```

Config is managed with Hydra. Override any group or value from the CLI:

```bash
# swap dataset
uv run python train.py data=fashion_mnist

# change model size or hyperparameters
uv run python train.py model=tiny trainer.max_epochs=50 model.drop_path_rate=0.1

# enable checkpointing
uv run python train.py callbacks=default
```

Runs are tracked with MLflow. To inspect results:

```bash
uv run mlflow ui --backend-store-uri mlruns/
```
