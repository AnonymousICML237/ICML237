#!/bin/bash

echo 'training model...'
python main.py --batchsize 128 --test_flag 'False' --model_path '' --test_data_path '' --train_data_path '' --dataset 'cifar10' --model 'vgg'
