#!/bin/bash

python OAT.py \
  --batchsize 128 \
  --epoch 80 \
  --lr 0.0005 \
  --test_flag 'True' \
  --test_data_path "/test_data_path/test_data_cln.p" \
  --test_label_path "/test_label_path/test_label.p" \
  --train_data_path "/train_data_path/" \
  --model_path "/model_path/" \
  --dropout 'False'
