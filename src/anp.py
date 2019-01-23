#---------------------------------------------------------------------
# This file includes training and testing of vanilla or ANP model
#---------------------------------------------------------------------
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

from torch.autograd import Variable

def get_bool(string):
    if(string == 'False'):
        return False
    else:
        return True

GPUID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID

# Training settings
parser = argparse.ArgumentParser(description='lat implementation')
parser.add_argument('--batchsize', type=int, default=128, help='training batch size')
parser.add_argument('--epoch', type=int, default=40, help='number of epochs to train for')
parser.add_argument('--input_ch', type=int, default=3, help='input image channels')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate')
parser.add_argument('--alpha', type=float, default=0.6, help='alpha')
parser.add_argument('--epsilon', type=float, default=0.6, help='epsilon')
parser.add_argument('--enable_lat', type=get_bool, default=True, help='enable lat')
parser.add_argument('--test_flag', type=get_bool, default=True, help='test or train')
parser.add_argument('--adv_flag', type=get_bool, default=False, help='adv or clean')
parser.add_argument('--test_data_path', default="../data_path/fgsm_eps_0.5.p", help='test data path')
parser.add_argument('--test_label_path', default="../data_path/label.p", help='test label path')
parser.add_argument('--train_data_path', default="../data/mnist/", help='training dataset path')
parser.add_argument('--model_path', default="../model/", help='number of classes')
parser.add_argument('--pro_num', type=int, default=8, help='progressive number')
parser.add_argument('--batchnorm', type=get_bool, default=True, help='batch normalization')
parser.add_argument('--dropout', type=get_bool, default=True, help='dropout')
parser.add_argument('--dataset', default='mnist', help='data set')
parser.add_argument('--model', default='lenet', help='target model, [lenet, resnet, vgg, ...]')
parser.add_argument('--logfile',default='log.txt',help='log file to accord validation process')

args = parser.parse_args()
#print(args)


def train_op(model):
    f=open(args.logfile,'w')
    # load training data and test set
    if args.dataset == 'mnist':
        train_data = torchvision.datasets.MNIST(
            root=args.train_data_path,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=False
        )
        test_data = torchvision.datasets.MNIST(
            root=args.train_data_path,
            train=False)

    if args.dataset == 'cifar10':
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

    if args.dataset == 'mnist':
        test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:args.batchsize].cuda() / 255.
        test_y = test_data.test_labels[:args.batchsize].cuda()
    if args.dataset == 'cifar10':
        test_x = torch.Tensor(test_data.test_data).view(-1,3,32,32)[:args.batchsize].cuda() / 255.
        test_y = torch.Tensor(test_data.test_labels)[:args.batchsize].cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = nn.CrossEntropyLoss()
    curr_lr = args.lr

    for epoch in range(args.epoch):
        adjust_learning_rate(args.lr, optimizer, epoch)
        for step, (x, y) in enumerate(train_loader):

            if not args.enable_lat:
                args.pro_num = 1
            if not len(y) == args.batchsize:
                continue
            b_x = Variable(x).cuda()
            b_y = Variable(y).cuda()
            if args.enable_lat:
                model.zero_reg()
            # progressive process
            for iter in range(args.pro_num):
                iter_input_x = b_x
                iter_input_x.requires_grad = True
                iter_input_x.retain_grad()

                logits = model(iter_input_x)
                loss = loss_func(logits, b_y)
                optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_(model.parameters(),args.batchsize)
                optimizer.step()

                # ANP: save grad in backward propagation
                # L2 norm
                #-------------------------------------------------
                if args.model == 'vgg':
                    if args.enable_lat:
                        model.save_grad(args.alpha)
                #-------------------------------------------------

            # test acc for validation set
            if (step+1) % 100 == 0:
                if args.enable_lat:
                    model.zero_reg()
                f.write('step {},'.format(step))
                print('epoch={}/{}, step={}/{}'.format(epoch,args.epoch,step,len(train_loader)))
                test_op(model,f)
                #if args.enable_lat:
                    #test_adv(model,0.031,f)
                    #test_adv(model,0.063,f)

            # save model
            if (step+1) % 200 == 0:
                print('saving model...')
                print('lat={}, pro/eps/a={}/{}/{}'.format(args.enable_lat, args.pro_num, args.epsilon, args.alpha))
                if args.enable_lat:
                    torch.save(model.state_dict(), args.model_path + 'lat_param.pkl')
                else:
                    torch.save(model.state_dict(), args.model_path + 'naive_param.pkl')


            # print batch-size predictions from training data
            if step % 20 == 0:
                if args.enable_lat:
                    model.zero_reg()
                #model.eval()
                with torch.no_grad():
                    test_output = model(b_x)
                train_loss = loss_func(test_output, b_y)
                pred_y = torch.max(test_output, 1)[1].cuda().data.cpu().squeeze().numpy()
                Accuracy = float((pred_y == b_y.data.cpu().numpy()).astype(int).sum()) / float(b_y.size(0))
                print('train loss: %.4f' % train_loss.data.cpu().numpy(), '| train accuracy: %.2f' % Accuracy)
                #model.train()
        #end for batch

    # end for epoch
    f.close()


def test_op(model,f=None):
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
        batch_size=args.batchsize,
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
    if f != None:
        f.write('Accuracy of the model on the test images: {:.2f} %'.format(100 * correct / total))
        f.write('\n')
    #print('now is {}'.format(type(model)))
    model.train(True)


if __name__ == "__main__":
    if args.enable_lat:
        real_model_path = args.model_path + "lat_param.pkl"
        print('loading the LAT model')
    else:
        real_model_path = args.model_path + "naive_param.pkl"
        print('loading the naive model')
    
    if args.test_flag:
        args.enable_lat = False
    
    if args.model == 'vgg':
        cnn = VGG16(enable_lat=args.enable_lat,
                    epsilon=args.epsilon,
                    pro_num=args.pro_num,
                    batch_size=args.batchsize,
                    if_dropout=args.dropout
                    )
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

