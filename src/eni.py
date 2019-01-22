#--------------------------------------------------------
# robustness evaluation of eni
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

parser.add_argument('--batchsize_noise', type=int, default=1000, help='load batch size')
parser.add_argument('--model_name', help='model name')
parser.add_argument('--model_path_noise',  help='model path')
parser.add_argument('--clean_path', help='clean data path')
parser.add_argument('--distortion_root',help='the path of the folder which contains all kinds of distortions.npy and label.npy')
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

eni_list = { 
'oat':list(),   
'nat':list(),   
'eat':list(),
'lat':list(),
'dplat':list()
}

def cal_eni(model,data):
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
    clean_data, test_label, size = read_data_label(args.clean_path,args.label_path)
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
    c_eni = 0
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
    c_eni =  abs(loss_cln - loss_tst) / dist 
    
    return c_eni

def generate():
    for model in model_list:
        print('--------- now model is {}:-------------'.format(model))
        for data in data_list:
            c_eni = cal_eni(model,data)
            print('now data is clean <-> {}, eai is {:.2f}'.format(data,c_eni))
            eai_list[model].append(c_eni)

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

def gaussian_noise(x, severity=1):
    c = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,0.11,0.12][severity - 1]
    #print(x)
    #x = np.array(x) / 255.
    
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1)

def shot_noise(x, severity=1):
    c = [500, 450, 400, 350, 300, 250, 200, 150 ,100][severity - 1]

    #x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) 


def get_test_loader(x,y):
    test_data = torch.from_numpy(x).float()
    test_label = torch.from_numpy(y).long()
    
    test_dataset = torch.utils.data.TensorDataset(test_data[:args.batchsize_noise],test_label[:args.batchsize_noise])
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                              batch_size = args.batchsize_noise,
                                              shuffle = False,
                                              drop_last = True)
    return test_loader

def get_clean_loader(y):
    test_label = torch.from_numpy(y).long()
    #print(test_label.size())
    clean_path = args.clean_path
    with open(clean_path, 'rb') as fr:
        clean_data = pickle.load(fr)
    clean_dataset = torch.utils.data.TensorDataset(clean_data[:args.batchsize_noise],test_label[:args.batchsize_noise])
    clean_loader = torch.utils.data.DataLoader(dataset = clean_dataset,
                                              batch_size = args.batchsize_noise,
                                              shuffle = False,
                                              drop_last = True)
    return clean_loader

def test_model(model,clean_loader,test_loader,distortion_name,model_name):
    eni = 0
    criterion = nn.CrossEntropyLoss()
    # Test the model
    model.eval()
    x_cln = 0
    loss_cln = 0
    for x, y in clean_loader:
        x = x.cuda()
        x_cln = x.view(x.size(0),-1)
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
        x_tst = x.view(x.size(0),-1)
        with torch.no_grad():
            h = model(x)
        loss = criterion(h, y)
        loss_tst = loss.item()
    model.train()

    dist = 0
    for j in range(x.size(0)):
        dist += torch.max(abs(x_cln[j] - x_tst[j]))     # norm p = inf
    dist = dist / x.size(0)
    eni =  abs(loss_cln - loss_tst) / dist 
    print('eps-ENI of the model VGG_' + model_name + ' on the ' + distortion_name + ': {:.4f} '.format(eni))

def test_model2(model,clean_loader,severity,model_name):
    criterion = nn.CrossEntropyLoss()
    # Test the model
    model.eval()
    x_cln = 0
    loss_cln = 0
    x_tst1 = 0
    loss_tst1 = 0
    x_tst2 = 0
    loss_tst2 = 0
    for x, y in clean_loader:
        x = x.cuda()
        x_cln = x.view(x.size(0),-1)
        #print(x_cln)
        y = y.cuda()
        #print(y)
        with torch.no_grad():
            h = model(x)
        loss = criterion(h, y)
        loss_cln = loss.item()

        x1 = torch.Tensor(x.size()).cuda()
        for i in range(x.size(0)):
            tmp = gaussian_noise(x[i].cpu().numpy().transpose(1,2,0),severity)
            x1[i] = torch.from_numpy(tmp.transpose(2,0,1)).float().cuda()
        x_tst1 = x1.view(x.size(0),-1)
        #print(x_tst1)
        #print(y)
        with torch.no_grad():
            h1 = model(x1)
        loss = criterion(h1, y)
        loss_tst1 = loss.item()
        
        x2 = torch.Tensor(x.size()).cuda()
        for j in range(x.size(0)):
            tmp = shot_noise(x[j].cpu().numpy().transpose(1,2,0),severity)
            x2[j] = torch.from_numpy(tmp.transpose(2,0,1)).float().cuda()
        x_tst2 = x2.view(x.size(0),-1)
        #print(x_tst2)
        y = y.cuda()
        #print(y)
        with torch.no_grad():
            h2 = model(x2)
        loss = criterion(h2, y)
        loss_tst2 = loss.item()
    model.train()
    
    dist1 = 0
    for k in range(x.size(0)):
        dist1 += torch.max(abs(x_cln[k] - x_tst1[k]))     # norm p = inf
    dist1 = dist1 / x.size(0)
    eni1 =  abs(loss_cln - loss_tst1) / dist1 
    print('eps-ENI of VGG_' + model_name + ' on gaussian_noise : {:.4f}(scale:{}) '.format(eni1,severity))

    dist2 = 0
    for m in range(x.size(0)):
        dist2 += torch.max(abs(x_cln[m] - x_tst2[m]))     # norm p = inf
    dist2 = dist2 / x.size(0)
    eni2 =  abs(loss_cln - loss_tst2) / dist2 
    print('eps-ENI of VGG_' + model_name + ' on shot_noise : {:.4f}(scale:{}) '.format(eni2,severity))

def noise():
    distortion_name = ['gaussian_noise','shot_noise','impulse_noise','speckle_noise','gaussian_blur','defocus_blur','glass_blur',
                       'snow','frost','fog','brightness','contrast','elastic_transform','motion_blur','zoom_blur','pixelate',
                       'jpeg_compression','spatter','saturate']
    label_name = 'labels.npy'
    
    #load model
    net = VGG16(enable_lat = False,
                epsilon = 0,
                pro_num = 1,
                batch_size = args.batchsize_noise,
                if_dropout = True)
    net.cuda()
    net.load_state_dict(torch.load(args.model_path_noise))
    
    label_root = args.distortion_root + label_name
    y = np.load(label_root)
    clean_loader = get_clean_loader(y)
    
    
    #error_rates = []
    for i in range(len(distortion_name)):
        data_root = args.distortion_root + distortion_name[i] + '.npy'
        label_root = args.distortion_root + label_name
        #load data
        x = np.load(data_root)
        #print(x.shape)
        #for j in range(0,10):
        #    plt.imsave('./cifar-c/{}_{}.png'.format(distortion_name[i],j), x[j])
        x = x.transpose((0,3,1,2))
        x = x/255.0
        y = np.load(label_root)
        #data_loader
        test_loader = get_test_loader(x,y)
        test_model(net,clean_loader,test_loader,distortion_name[i],args.model_name)
    
    '''
    for severity in range(1,10):    
        test_model2(net,clean_loader,severity,args.model_name)
    '''
    #print('mCE (unnormalized by VGG_' + args.model_name + ' errors) (%):{:.2f}'.format(np.mean(error_rates)))
if __name__ == "__main__":
    generate()
    #save()
    #eai_list = load()

    noise()
