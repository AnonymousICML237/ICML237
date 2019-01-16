#--------------------------------------------------------
# robustness evaluation of eai
#--------------------------------------------------------
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import os
from utils import *
from VGG import VGG16
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable

GPUID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID

def get_bool(string):
    if(string == 'False'):
        return False
    else:
        return True

# Training settings
parser = argparse.ArgumentParser(description='eai implementation')
parser.add_argument('--model_path', help='testing model path')
parser.add_argument('--data_path', help='testing model path')
parser.add_argument('--label_path', help='testing label path')
parser.add_argument('--save_path', help='eai list saving path')
parser.add_argument('--enable_lat', type=get_bool, default=False, help='enable lat')
parser.add_argument('--alpha', type=float, default=0.6, help='alpha')
parser.add_argument('--epsilon', type=float, default=0.6, help='epsilon')
parser.add_argument('--pro_num', type=int, default=5, help='progressive number')
parser.add_argument('--batchsize', type=int, default=128, help='testing batch size')
parser.add_argument('--dropout', type=get_bool, default=True, help='dropout')
parser.add_argument('--start_idx', type=int, default=2000, help='start index')
parser.add_argument('--length', type=int, default=100, help='amount of testing images')
args = parser.parse_args()
#print(args)

model_list = {
'oat': args.model_path + "oat/naive_param.pkl",
'nat': args.model_path + "nat/naive_param.pkl",
'eat': args.model_path + "eat/naive_param.pkl",
'lat': args.model_path + "lat/naive_param.pkl",
'dplat': args.model_path + "dplat/lat_param.pkl",
}

data_list = {
'fgsm-e8':args.data_path + "fgsm/test_adv(eps_0.031).p",
'fgsm-e16':args.data_path + "fgsm/test_adv(eps_0.063).p",
'stepll-e8':args.data_path + "stepll/test_adv(eps_0.031).p",
'stepll-e16':args.data_path + "stepll/test_adv(eps_0.063).p",
'pgd-a16':args.data_path + "pgd/test_adv(alpha_0.063).p",
'pgd-a2':args.data_path + "pgd/test_adv(alpha_0.008).p",
'mifgsm-e8':args.data_path + "mifgsm/test_adv(eps_0.031).p",
}

'''
data_list = {
'fgsm':args.data_path + "fgsm/",
'stepll':args.data_path + "stepll/",
}
'''

lip_list = { 
'oat':list(),   
'nat':list(),   
'eat':list(),
'lat':list(),
'dplat':list()
}

def cal_eai(model,data):
    cnn = VGG16(enable_lat=args.enable_lat,
                epsilon=args.epsilon,
                pro_num=args.pro_num,
                batch_size=args.batchsize,
                if_dropout=args.dropout
                )
    cnn.cuda()
    model_path = model_list[model]
    if os.path.exists(model_path):
        cnn.load_state_dict(torch.load(model_path))
        print('load model successfully.')
    else:
        print("load failed.")
    model = cnn
    # get test_data , test_label from .p file
    clean_data, test_label, size = read_data_label(data_list['clean'],args.label_path)
    test_data, test_label, size = read_data_label(data_list[data],args.label_path)
    if size == 0:
        print("reading data failed.")
        return
    if data == 'clean':
        sel_clean = clean_data[args.start_idx:args.start_idx+args.length]
        sel_test = test_data[args.start_idx+args.length:args.start_idx+2*args.length]
        sel_clean_label = test_label[args.start_idx:args.start_idx+args.length]
        sel_test_label = test_label[args.start_idx+args.length:args.start_idx+2*args.length]
    else:
        sel_clean = clean_data[args.start_idx:args.start_idx+args.length]
        sel_test = test_data[args.start_idx:args.start_idx+args.length]
        sel_clean_label = test_label[args.start_idx:args.start_idx+args.length]
        sel_test_label = test_label[args.start_idx:args.start_idx+args.length]

    
    # create dataset
    clean_set = Data.TensorDataset(sel_clean, sel_clean_label)
    test_set = Data.TensorDataset(sel_test, sel_test_label)
    
    clean_loader = Data.DataLoader(
        dataset=clean_set,
        batch_size=args.length,   
        shuffle=False
    )
    
    test_loader = Data.DataLoader(
        dataset=test_set,
        batch_size=args.length,  
        shuffle=False
    )
    c_eai = 0
    criterion = nn.CrossEntropyLoss()
    # Test the model
    model.eval()
    x_cln = 0
    loss_cln = 0
    for x, y in clean_loader:
        x = x.cuda()
        x_cln = x.view(sel_clean.size(0),-1)
        y = y.cuda()
        #print(y)
        with torch.no_grad():
            h = model(x)
        loss = criterion(h, y)
        loss_cln = loss.item()
    model.train()
    model.eval()
    x_tst = 0
    loss_tst = 0
    for x, y in test_loader:
        x = x.cuda()
        y = y.cuda()
        x_tst = x.view(sel_test.size(0),-1)
        with torch.no_grad():
            h = model(x)
        loss = criterion(h, y)
        loss_tst = loss.item()
    model.train()

    dist = 0
    for j in range(sel_test.size(0)):
        dist += torch.max(abs(x_cln[j] - x_tst[j]))     # norm p = inf
    dist = dist / args.length
    c_eai =  abs(loss_cln - loss_tst) / dist 
    
    return c_eai

def generate():
    for model in model_list:
        print('--------- now model is {}:-------------'.format(model))
        for data in data_list:
            c_eai = cal_eai(model,data)
            print('now data is clean <-> {}, eai is {:.2f}'.format(data,c_eai))
            eai_list[model].append(c_eai)

def save():
    if os.path.exists(args.save_path) == False:
        os.mkdir(args.save_path)
    with open(args.save_path+'eai_list.p','wb') as f:
        pickle.dump(eai_list, f, pickle.HIGHEST_PROTOCOL)

def load():
    if os.path.exists(args.save_path) == False:
        print("load data error")
    with open(args.save_path+'eai_list.p', 'rb') as fr:
        eai_list = pickle.load(fr)
    return eai_list

if __name__ == "__main__":
    generate()
    #save()
    #eai_list = load()
