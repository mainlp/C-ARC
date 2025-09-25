#!/bin/bash
DATA_PATH="data/all_episodes.jsonl"

scripts/split_data_systematicity.py \
    --data_path $DATA_PATH \
    --frac_test_compositions 0.2 \
    --min_func_examples 5000 \
    --no_shuffle_study_examples \
    --num_primitives 2 \
    --num_compositions 2 \
    --seed 1860