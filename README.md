# PGM-single-cell

To create the environment with all the packages and especially scvi-tools:
```conda env create -f environment.yml```

# VAE Training and Evaluation Script

## Commands

- `--data`:  Path to the dataset (AnnData format), default: `None` so cortex dataset.
- `--latent_dims`: Latent dimensions for each model, default: `10`.
- `--model_names`: Model names `simple_vae` or `gm_vae`.
- `--likelihood`: Likelihood type: `nb`, `zinb`, `p`, `zip`, default: `nb`.
- `--training`: Train (`True`) or load an existing model (`False`), default: `True`.
- `--training`: Evaluate the model, default: `True`.
- `--model_saves`: Paths to save/load models, default: `None`.
- `--max_epochs`: Max number of training epochs per model, default: `10`.
- `--cluster_type`: Cluster types: `cell_type` or `precise_labels`, default:`cell_type`.
- `--accelerator`: Use of specific accelerator, default: `auto`.
- `--use_wandb`: Activate wandb logs, default: `True` or `False`.

## Examples

```bash
python main.py --latent_dims 10 --model_names simple_vae --training True --eval True --max_epochs 20 --likelihood nb --use_wandb False
```
