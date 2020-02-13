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
        --rampup_length 150 \
        --rampup_coefficient 50 \
        --num_labeled_classes 80 \
        --num_unlabeled_classes 20 \
        --dataset_name cifar100 \
        --seed 0 \
        --model_name resnet_cifar100 \
        --mode train