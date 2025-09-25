#!/bin/bash
DATA_DIR="data/split_seed_1860"
CHECKPOINT_DIR="models/split_seed_1860"
PLOT_DIR="experimental_results/models/plots/split_seed_1860/mlc"
WANDB_DIR="wandb"

python scripts/train_mlc.py \
    --data_dir $DATA_DIR \
    --data_file_name "systematicity_seed_1860" \
    --checkpoint_dir $CHECKPOINT_DIR \
    --wandb_dir $WANDB_DIR \
    --wandb_name "train_mlc_seed_1860" \
    --plot_dir $PLOT_DIR \
    --plot_freq 0 \
    --batch_size 200 \
    --nepochs 300 \
    --seed 1860