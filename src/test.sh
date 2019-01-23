#!/bin/bash

echo 'testing model...'
python anp.py --batchsize 128 --lr 0.0001 --epoch 40 --test_flag 'True' --enable_lat 'True' --pro_num 3 --alpha 0.5 --epsilon 0.3 --model_path "." --test_data_path "." --test_label_path "./test_label.p" --train_data_path "./data/cifar10/" --dataset 'cifar10' --model 'vgg' --dropout 'True' 
