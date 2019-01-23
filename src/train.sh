#!/bin/bash

echo 'training model...'
python anp.py --batchsize 128 --lr 0.0001 --epoch 40 --test_flag 'False' --enable_lat 'False' --pro_num 5 --alpha 0.7 --epsilon 0.3 --model_path "/media/dsg3/dsgprivate/lat/model/vgg/" --test_data_path "./test_data_cln.p" --test_label_path "./test_label.p" --train_data_path "./data/cifar10/" --dataset 'cifar10' --model 'vgg' --dropout 'True' 
