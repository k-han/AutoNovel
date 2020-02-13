#!/usr/bin/env bash

python auto_novel.py \
        --dataset_root $1 \
        --exp_root $2 \
        --warmup_model_dir $3 \
        --lr 0.1 \
        --step_size 170 \
        --batch_size 128 \
        --epochs 200 \
        --rampup_length 80 \
        --rampup_coefficient 50 \
        --seed 0 \
        --dataset_name svhn \
        --model_name resnet_svhn \
        --mode train