#!/bin/bash

python CIFAR10-P_test.py \
  --batchsize 128 \
  --model_name 'OAT' \
  --model_path '/model_path/VGG_OAT.pkl' \
  --distotion_root '/distotion_root/'
