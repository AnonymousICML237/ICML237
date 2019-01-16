#!/bin/bash

python attack.py \
  --attack 'fgsm' \
  --generate 'False' \
  --droplast 'True' \
  --model 'vgg' \
  --enable_lat 'False'\
  --modelpath "../model_path/naive_param.pkl" \
  --dataroot "../data/train/cifar10/" \
  --model_batchsize 128 \
  --dropout 'True' \
  --dataset 'cifar10' \
  --attack_batchsize 128 \
  --attack_epsilon 8 \
  --attack_alpha 1 \
  --attack_iter 6 \
  --attack_momentum 1.0 \
  --savepath "../save_path/" \
  --imgpath "../img_path/" \
  --dplat_epsilon 0.6 \
  --dplat_pronum 7
