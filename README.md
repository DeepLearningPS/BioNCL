# BioNCL

BioNCL is a biomedical knowledge-graph link-prediction implementation built on the HEAT encoder and a contrastive learning objective. This release contains the code required to train and evaluate HEAT with either binary cross-entropy (BCE) loss or the BioNCL contrastive loss.

## Contents

```
BioNCL/
├── checkpoints/             # One BCE and one BioNCL checkpoint per dataset
├── config/                  # Logging configuration
├── data/                    # Minimal train/validation/test splits and ID maps
├── new_pyHGT_rel/           # HEAT model implementation
├── run_lkgnn.py             # Training and evaluation entry point
├── data_loader.py
├── helper.py
└── requirements.txt
```

The included datasets are `DRKG9k-25_dense`, `New_PharmKG8k-28`, and `BioKG10k_14`. Each dataset directory contains only the split files and entity/relation ID maps needed for reproduction. Triples use tab-separated `head relation tail` format.

## Environment

Use Python 3.10 or later and a CUDA-enabled PyTorch installation. Install PyTorch and the PyTorch Geometric packages compatible with your CUDA version following the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), then install the remaining packages:

```bash
pip install -r requirements.txt
```

All commands below use one GPU. Select the desired GPU by changing `CUDA_VISIBLE_DEVICES`.

## Train

The following command trains the BCE baseline on DRKG9k-25_dense. Model files and logs are deliberately written to the ignored `outputs/` directory.

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 run_lkgnn.py \
  --dataset DRKG9k-25_dense --conv_name heat --optimizer adam --change_lr 0 \
  --loss_function bcel_loss --score_func conve --batch_size 1024 \
  --model_dir outputs/checkpoints --logdir outputs/logs
```

To train BioNCL, change `--loss_function bcel_loss` to `--loss_function contrastive_loss`:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 run_lkgnn.py \
  --dataset DRKG9k-25_dense --conv_name heat --optimizer adam --change_lr 0 \
  --loss_function contrastive_loss --score_func conve --batch_size 1024 \
  --model_dir outputs/checkpoints --logdir outputs/logs
```

Replace `DRKG9k-25_dense` with `New_PharmKG8k-28` or `BioKG10k_14` to run the other datasets. The command-line settings follow the original experiment configuration in `lkgnn_analy.md`.

## Evaluate released checkpoints

Evaluate the released DRKG BCE checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 run_lkgnn.py \
  --dataset DRKG9k-25_dense --conv_name heat --optimizer adam --change_lr 0 \
  --loss_function bcel_loss --score_func conve --batch_size 1024 \
  --model_dir checkpoints --logdir outputs/logs --test 1 \
  --save_path DRKG9k-25_dense_heat_bce.pt
```

Evaluate the released BioNCL checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 run_lkgnn.py \
  --dataset DRKG9k-25_dense --conv_name heat --optimizer adam --change_lr 0 \
  --loss_function contrastive_loss --score_func conve --batch_size 1024 \
  --model_dir checkpoints --logdir outputs/logs --test 1 \
  --save_path DRKG9k-25_dense_heat_bioncl.pt
```

Released checkpoint names follow this convention:

| Dataset | BCE checkpoint | BioNCL checkpoint |
| --- | --- | --- |
| DRKG9k-25_dense | `DRKG9k-25_dense_heat_bce.pt` | `DRKG9k-25_dense_heat_bioncl.pt` |
| New_PharmKG8k-28 | `New_PharmKG8k-28_heat_bce.pt` | `New_PharmKG8k-28_heat_bioncl.pt` |
| BioKG10k_14 | `BioKG10k_14_heat_bce.pt` | `BioKG10k_14_heat_bioncl.pt` |

## Data location

By default, the program reads datasets from `data/` inside this repository. To keep data outside the repository, set `BIONCL_DATA_ROOT` to a directory containing the three dataset folders:

```bash
export BIONCL_DATA_ROOT=/path/to/BioNCL-data
```

## GitHub release

The checkpoint files are tracked through Git LFS (`checkpoints/*.pt`). Before committing or cloning the repository, install and initialize Git LFS:

```bash
git lfs install
git add .
git commit -m "Initial BioNCL release"
```

Training outputs, runtime logs, Python caches, and generated embeddings are excluded through `.gitignore`.

## License

See [LICENSE](LICENSE).
