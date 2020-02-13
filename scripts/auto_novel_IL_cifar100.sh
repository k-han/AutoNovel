#!/usr/bin/env bash

python auto_novel.py \
        --dataset_root $1 \
        --exp_root $2 \
        --warmup_model_dir $3 \
        --lr 0.1 \
        --gamma 0.1 \
        --weight_decay 1e-4 \
        --step_size 340 \
        --batch_size 256 \
        --epochs 400 \
        --rampup_length 300 \
        --rampup_coefficient 25 \
        --num_labeled_classes 80 \
        --num_unlabeled_classes 20 \
        --dataset_name cifar100 \
        --IL \
        --increment_coefficient 0.05 \
        --seed 0 \
        --model_name resnet_IL_cifar100 \
        --mode train