#!/usr/bin/env bash

python auto_novel.py \
        --dataset_root $1 \
        --exp_root $2 \
        --warmup_model_dir $3 \
        --lr 0.1 \
        --gamma 0.1 \
        --weight_decay 1e-4 \
        --step_size 170 \
        --batch_size 128 \
        --epochs 200 \
        --rampup_length 50 \
        --rampup_coefficient 5.0 \
        --dataset_name cifar10 \
        --IL \
        --increment_coefficient 0.05 \
        --seed 0 \
        --model_name resnet_IL_cifar10 \
        --mode train