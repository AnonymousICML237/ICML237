#!/bin/bash

python EAT.py \
  --batchsize 128 \
  --epoch 100 \
  --lr 0.0005 \
  --test_flag 'True' \
  --test_data_path "/test_data_path/test_adv(eps_0.03).p" \
  --test_label_path "/test_label_path/test_label.p" \
  --train_data_path "/train_data_path/" \
  --model_path "/model_path/" \ 
