#!/usr/bin/env bash

path="data/experiments/"
mkdir -p $path
cd $path

wget http://www.robots.ox.ac.uk/~vgg/research/auto_novel/asset/pretrained.zip

unzip pretrained.zip && rm pretrained.zip

cd ../