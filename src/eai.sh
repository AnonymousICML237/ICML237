#!/bin/bash

python eai.py \
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
  --length 100
  
