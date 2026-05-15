# liteViT

A from-scratch Vision Transformer implementation — no timm, no pretrained weights. Built to run on Apple Silicon (MPS backend) and designed for rapid experimentation with depth, learning rate, drop path, and pluggable attention/transformer block architectures.

**Stack:** PyTorch · PyTorch Lightning · Hydra · MLflow · Optuna · torchmetrics

[![pytest](https://github.com/chris-beischl/liteViT/actions/workflows/pytest.yml/badge.svg)](https://github.com/chris-beischl/liteViT/actions/workflows/pytest.yml) [![static-analysis](https://github.com/chris-beischl/liteViT/actions/workflows/static_analysis.yml/badge.svg)](https://github.com/chris-beischl/liteViT/actions/workflows/static_analysis.yml)

---

## Table of Contents

- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Hyperparameter Sweeps](#hyperparameter-sweeps)
- [Model Comparison](#model-comparison)
  - [Results](#results)
- [Inspecting Results](#inspecting-results)
- [Project Structure](#project-structure)
- [Attention Mechanisms](#attention-mechanisms)
- [Datasets](#datasets)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Tests](#tests)

---

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv manages the Python version and all dependencies automatically.

```bash
uv sync
```

To run model comparisons involving timm, install the comparison dependencies:

```bash
uv sync --group comparison
```

---

## Training

```bash
uv run python train.py                                              # default config
uv run python train.py data=fashion_mnist                           # swap dataset
uv run python train.py data=fashion_mnist model=lite_experimental   # swap model
uv run python train.py model.depth=6 model.drop_path_rate=0.1       # override params
uv run python train.py callbacks=default                            # enable checkpointing
```

**Swapping attention mechanisms:**

```bash
# MHA (default)
uv run python train.py data=fashion_mnist model=lite_experimental 'model/attention@model.attention_cfg=vanilla'

# Multi-Query Attention
uv run python train.py data=fashion_mnist model=lite_experimental 'model/attention@model.attention_cfg=mqa'

# Grouped Query Attention
uv run python train.py data=fashion_mnist model=lite_experimental 'model/attention@model.attention_cfg=gqa'
```

Config is managed with Hydra — any group or value can be overridden from the CLI.

---

## Evaluation

Evaluate a checkpoint against the test set. Results are logged into the same MLflow run as training:

```bash
uv run python eval.py --checkpoint checkpoints/<date>/<time>/best-epoch=X-val_accuracy=Y.ckpt
```

---

## Hyperparameter Sweeps

Sweeps are managed with Optuna and defined as self-contained modules in `scripts/sweeps/`. Use `scripts/sweeps/template.py` as a starting point.

**Run a sweep:**

```bash
uv run python sweep.py \
  --sweep fashion_mnist_init \           # sweep module name (scripts/sweeps/<name>.py)
  --override data=fashion_mnist \        # fixed Hydra overrides applied to every trial
  --override model=lite_experimental \
  --n_trials 30                          # total number of trials
```

Sweep results persist in `optuna_studies.db` and resume automatically if interrupted.

---

## Model Comparison

Requires comparison dependencies (`uv sync --group comparison`). Run a systematic comparison across models, datasets, and seeds:

```bash
uv run python scripts/comparison.py \
  --models lite_comparison timm_comparison \   # model config names (configs/model/<name>.yaml)
  --datasets fashion_mnist cifar10 \           # data config names
  --n_seeds 5                                  # seeds 0..n-1, runs sequentially
```

Runs the full cartesian product of models × datasets × seeds. Each run is logged to MLflow under the `comparison` experiment with run names like `lite_comparison_fashion_mnist_seed0`.

Optional overrides (all have sensible defaults):

```bash
  --optimizer comparison \   # configs/optimizer/<name>.yaml (default: comparison)
  --scheduler cosine_warmup \ # configs/scheduler/<name>.yaml (default: cosine_warmup)
  --callbacks default \       # configs/callbacks/<name>.yaml (default: default)
  --logger mlflow             # configs/logger/<name>.yaml (default: mlflow)
```

Failed runs print a warning and are skipped — rerun them individually with `train.py`.

### Results

All results are mean ± std over 5 seeds. Hyperparameters were selected via Optuna sweep on liteViT and held fixed for timm to ensure a fair comparison. The only architectural difference is positional embeddings: liteViT uses fixed sincos, timm uses learnable.

**FashionMNIST** — depth=6, embed\_dim=128, num\_heads=8, drop\_path\_rate=0.1, lr=7e-4, wd=3e-3

| Model | Dataset | Test Accuracy | Test F1 |
|---|---|---|---|
| liteViT | FashionMNIST | **93.56 ± 0.14%** | **93.55 ± 0.14%** |
| timm ViT | FashionMNIST | 93.15 ± 0.07% | 93.13 ± 0.07% |

**To reproduce:**

```bash
uv sync --group comparison
uv run python scripts/comparison.py \
  --models lite_comparison timm_comparison \
  --datasets fashion_mnist \
  --n_seeds 5
uv run python scripts/collect_results.py \
  -e comparison \
  -m lite_comparison timm_comparison \
  -d fashion_mnist
```

---

## Inspecting Results

**MLflow UI** — training metrics, hyperparameters, run comparison:

```bash
uv run python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Optuna Dashboard** — sweep optimization history and parameter importance:

```bash
uv run optuna-dashboard sqlite:///optuna_studies.db
```

**Optuna CLI** — list trials from the terminal:

```bash
uv run optuna trials --storage sqlite:///optuna_studies.db --study-name FashionMNIST-init
```

---

## Project Structure

```
train.py                        # training entry point
eval.py                         # checkpoint evaluation
sweep.py                        # Optuna sweep runner
scripts/
  comparison.py                 # multi-model/dataset/seed comparison runner
  collect_results.py            # eval all checkpoints, export CSV + summary
  sweeps/                       # Optuna sweep objective definitions
    template.py                 # starting point for new sweeps
    fashion_mnist_init.py
    cifar10_init.py
configs/
  train.yaml                    # root Hydra config
  model/                        # model configs (lite_comparison, timm_comparison, ...)
    attention/                  # attention sub-configs (vanilla, mqa, gqa)
    block/                      # block sub-configs (vanilla, rms_norm)
    patch_embed/                # patch embed sub-configs (conv)
  data/                         # dataset configs (mnist, fashion_mnist, kmnist, cifar10)
  optimizer/                    # optimizer configs (adam_w, comparison)
  scheduler/                    # lr scheduler configs (cosine, cosine_warmup)
  callbacks/                    # callback configs (default, no_checkpoint)
  trainer/                      # trainer config
  logger/                       # MLflow logger config
litevit/
  models/
    vit.py                      # ViT nn.Module + build_vit factory
    patch_embed.py              # BasePatchEmbed ABC + ConvPatchEmbed
    timm_vit.py                 # build_timm_vit factory (requires comparison group)
    attention/                  # BaseAttention + VanillaAttention, MQA, GQA
    block/                      # BaseTransformerBlock + VanillaTransformerBlock
  data/                         # BaseDataModule + MNIST, FashionMNIST, KMNIST, CIFAR10
  training/
    module.py                   # ClassificationModule (LightningModule)
    callbacks.py                # CheckpointMetadataCallback
    utils.py                    # build_lr_scheduler_with_warmup
  utils/                        # drop_path, pos_embed, resolve
tests/
```

---

## Attention Mechanisms

| Config | Description | Reference |
|---|---|---|
| `vanilla` | Multi-Head Self-Attention | Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017 |
| `mqa` | Multi-Query Attention — shared K/V across all heads | Shazeer, [*Fast Transformer Decoding*](https://arxiv.org/abs/1911.02150), 2019 |
| `gqa` | Grouped Query Attention — shared K/V across groups | Ainslie et al., [*GQA*](https://arxiv.org/abs/2305.13245), 2023 |

---

## Datasets

| Config | Dataset | Classes | Resolution | Reference |
|---|---|---|---|---|
| `mnist` | MNIST | 10 | 28×28 grayscale | LeCun et al., 1998 |
| `fashion_mnist` | FashionMNIST | 10 | 28×28 grayscale | Xiao et al., [*Fashion-MNIST*](https://arxiv.org/abs/1708.07747), 2017 |
| `kmnist` | KMNIST | 10 | 28×28 grayscale | Clanuwat et al., [*Deep Learning for Classical Japanese Literature*](https://arxiv.org/abs/1812.01718), 2018 |
| `cifar10` | CIFAR-10 | 10 | 32×32 RGB | Krizhevsky, [*Learning Multiple Layers of Features from Tiny Images*](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), 2009 |

---

## Pre-commit Hooks

Install the hooks once after cloning:

```bash
uv run pre-commit install
```

Hooks run automatically on every commit:

- **pre-commit-hooks**: trailing whitespace, end-of-file fixing, YAML/TOML validation, merge conflict detection
- **nbclearout**: strips Jupyter notebook outputs before committing
- **ruff**: linting (with auto-fix) and formatting
- **mypy**: strict type checking across `litevit/`, `train.py`, `eval.py`, `sweep.py`

To run manually on all files:

```bash
uv run pre-commit run --all-files
```

---

## Tests

```bash
uv run pytest             # all tests with coverage
uv run pytest --no-cov    # skip coverage (faster)
```
