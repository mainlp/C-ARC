#!/bin/bash
DATA_DIR="data"

python scripts/gen_data.py --data_dir $DATA_DIR --plot_freq 1000 --num_samples 100000 --max_tries 500