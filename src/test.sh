#!/bin/bash

echo 'testing model...'
python dplat.py --batchsize 128 --lr 0.0001 --epoch 40 --test_flag 'True' --enable_lat 'True' --pro_num 3 --alpha 0.5 --epsilon 0.3 --model_path "/media/dsg3/dsgprivate/lat/model/vgg/" --test_data_path "/media/dsg3/dsgprivate/lat/test/new/test_data_cln.p" --test_label_path "/media/dsg3/dsgprivate/lat/test/new/test_label.p" --train_data_path "/media/dsg3/dsgprivate/lat/data/cifar10/" --dataset 'cifar10' --model 'vgg' --dropout 'True' 
