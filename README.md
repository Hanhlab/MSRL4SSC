# MSRL4SSC: A multi-source satellite image representation learning framework for soil salt content retrieval:

1. cross-modal self-supervised learning from S2 and S1 patches;
2. image-only modality-optional task adaptation;
3. PCA(4) plus ten-regressor SSC evaluation.

## Required external inputs

- a user-provided sample index and its referenced patch files;
- an official CROMA checkpoint;
- conventional predictors required only by Stage 3.

The example paths in `configs/five_index_full.yaml` are placeholders. The S2 channel order is `SI3, S_2, S_3, kNDMI, TCARI`; S1 uses `VV, VH`.

## Environment

```bash
conda activate torch_suc
pip install -r requirements.txt
```

## Stage 1 and Stage 2

```bash
python train.py \
  --data-root /path/to/example_dataset \
  --pretrained-checkpoint /path/to/CROMA_base.pt \
  --base-config configs/five_index_full.yaml \
  --output-root outputs \
  --resume
```

Stage 1 initializes the MSRL backbone from official CROMA weights and reconstructs masked S1 and S2 patches with normalized MSE. Stage 2 fine-tunes the final transformer block. 

## Stage 3

```bash
python evaluate_stage3.py \
  --feature-root outputs/selected_features \
  --sample-index outputs/master_index_seed74.csv \
  --output-dir outputs/stage3
```

For every outer fold, PCA is fitted only to the 128-dimensional image state of the four training folds. Four PCs are concatenated with the Stage-3 conventional predictors and evaluated using Huber, Ridge, CatBoost, PLSR, KNN, ElasticNet, ExtraTrees, GradientBoosting, RandomForest, and ANN.

## Files

- `train.py`: spatial five-fold Stage-1/2 orchestration.
- `evaluate_stage3.py`: Stage-3 PCA and ten-regressor evaluation.
- `scripts/`: Stage-1 and Stage-2 training entry points.
- `msrl4ssc/`: model, loss, data interface, checkpoint adaptation, and regressors.
- `configs/five_index_full.yaml`: schematic experiment configuration.

The historical accuracy from an earlier implementation cannot be assigned to this image-only Stage-2 version without retraining.

