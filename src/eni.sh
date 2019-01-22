#!/bin/bash

python eni.py \
  --batchsize 128 \
  --enable_lat 'False' \
  --data_path "../data_path/" \
  --label_path "../label_path/" \
  --model_path "../model_path/" \
  --save_path "../save_path/" \
  --pro_num 5 \
  --alpha 0.7 \
  --epsilon 0.3 \
  --dropout 'True' \
  --start_idx 1000 \
  --length 100 \
  --batchsize_noise 1000 \
  --model_name '' \
  --model_path_noise '' \
  --clean_path "" \
  --distortion_root ''
  
