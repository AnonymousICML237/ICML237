#--------------------------------------------------------
# original adversarial training
#--------------------------------------------------------
import torch
import torch.nn as nn
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import os
from utils import read_data_label
from VGG import *
from math import *
from numpy.random import normal

from torch.autograd import Variable

def get_bool(string):
    if(string == 'False'):
        return False
    else:
        return True

# Training settings
parser = argparse.ArgumentParser(description='lat implementation')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--epoch', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--input_ch', type=int, default=3, help='input image channels')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate')
parser.add_argument('--test_flag', type=get_bool, default=True, help='test or train')
parser.add_argument('--test_data_path', default="/test_data_path/test_data_cln.p", help='test data path')
parser.add_argument('--test_label_path', default="/test_label_path/tset_label.p", help='test label path')
parser.add_argument('--train_data_path', default="/train_data_path/", help='training dataset path')
parser.add_argument('--model_path', default="/model_path/", help='model_path')
parser.add_argument('--batchnorm', type=get_bool, default=True, help='batch normalization')
parser.add_argument('--dropout', type=get_bool, default=False, help='dropout') 
args = parser.parse_args()

def fgsm(model,criterion,batch_size,alpha,X_input,Y_input):
    model.eval()
    adv_num = floor(alpha * batch_size)


    X = Variable(X_input[0:adv_num].clone(), requires_grad = True).cuda()
    Y = Variable(Y_input[0:adv_num].clone(), requires_grad = False).cuda()

    h= model(X)
    loss = criterion(h, Y)
    model.zero_grad()
    if X.grad is not None:
        X.grad.data.fill_(0)
    loss.backward()
        
    X_adv = X.detach() + 0.031 * torch.sign(X.grad)
    X_adv = torch.clamp(X_adv,0,1)         
    

    one_batch_X = torch.cat((X_adv,X_input[adv_num:].clone().cuda()),0)
    one_batch_Y = torch.cat((Y,Y_input[adv_num:].clone().cuda()),0)
    model.train()
    return one_batch_X,one_batch_Y

def train_op(model):

    # load training data and test set
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    train_data = torchvision.datasets.CIFAR10(
        root=args.train_data_path,
        train=True,
        transform=transform,
        download=False
    )
    test_data = torchvision.datasets.CIFAR10(
        root=args.train_data_path,
        train=False)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = nn.CrossEntropyLoss()
    curr_lr = args.lr
    for epoch in range(args.epoch):
        for step, (x, y) in enumerate(train_loader):
            if not len(y) == args.batchsize:
                continue
            x,y = fgsm(model,loss_func,args.batchsize,0.5,x,y)
            b_x = Variable(x).cuda()
            b_y = Variable(y).cuda()
            model.zero_reg()
            # progressive process
            
            iter_input_x = b_x
            iter_input_x.requires_grad = True
            iter_input_x.retain_grad()

            logits = model(iter_input_x)
            loss = loss_func(logits, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save model and test
            if (step+1) % 100 == 0:
                print('epoch : ' + str(epoch))
                model.zero_reg()
                test_op(model)
                print('saving model...')
                torch.save(model.state_dict(), args.model_path + 'VGG_OAT.pkl')

            
            # print batch-size predictions from training data
            if (step+1) % 10 == 0:
                model.zero_reg()
                model.eval()
                with torch.no_grad():
                    test_output = model(b_x)
                train_loss = loss_func(test_output, b_y)
                pred_y = torch.max(test_output, 1)[1].cuda().data.cpu().squeeze().numpy()
                Accuracy = float((pred_y == b_y.data.cpu().numpy()).astype(int).sum()) / float(b_y.size(0))
                print('train loss: %.4f' % train_loss.data.cpu().numpy(), '| train accuracy: %.2f' % Accuracy)
                model.train(True)
            


def test_op(model):
    # get test_data , test_label from .p file
    test_data, test_label, size = read_data_label(args.test_data_path,args.test_label_path)

    if size == 0:
        print("reading data failed.")
        return

    test_data = test_data.cuda()
    test_label = test_label.cuda()
    
    # create dataset
    testing_set = Data.TensorDataset(test_data, test_label)

    testing_loader = Data.DataLoader(
        dataset=testing_set,
        batch_size=args.batchsize, # without minibatch cuda will out of memory
        shuffle=False,
        #num_workers=2
        drop_last=True,
    )
    
    # Test the model
    model.eval()
    correct = 0
    total = 0
    for x, y in testing_loader:
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            h = model(x)
        _, predicted = torch.max(h.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    print('Accuracy of the model on the test images: {:.2f} %'.format(100 * correct / total))        
    #print('now is {}'.format(type(model)))
    model.train(True)


if __name__ == "__main__":
    real_model_path = args.model_path + 'VGG_OAT.pkl'

    
    cnn = VGG16(enable_lat = False,
                epsilon = 0,
                pro_num = 1,
                batch_size = args.batchsize,
                if_dropout = args.dropout)
    cnn.cuda()

    if os.path.exists(real_model_path):
        cnn.load_state_dict(torch.load(real_model_path))
        print('load model.')
    else:
        print("load failed.")

    if args.test_flag:
        test_op(cnn)
    else:
        train_op(cnn)
        
