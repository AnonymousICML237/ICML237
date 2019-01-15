#!/bin/bash

echo 'testing model...'
python main.py --batchsize 128 --test_flag 'True' --model_path '' --test_data_path '' --dataset 'cifar10' --model 'vgg'
