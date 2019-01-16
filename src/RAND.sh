#!/bin/bash

python RAND.py \
  --batch_size 128 \
  --img_size 32 \
  --img_resize 35 \
  --iter_times 1 \
  --num_classes 10 \
  --test_data_path "../data_path/test_adv(eps_0.031).p" \
  --test_label_path "../label_path/test_label.p" \
  --model 'vgg' \
  --model_path "../model_path/lat_param.pkl"
