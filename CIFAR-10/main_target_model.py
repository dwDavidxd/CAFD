from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime

from networks import *
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--save_freq', '-p', default=2, type=int, help='print frequency (default: 10)')
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

transform_test = transforms.Compose([
    transforms.ToTensor()])

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

# Test only option
if (args.testOnly):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+file_name+'.t7')
    net = checkpoint['net']

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        acc = 100.*correct/total
        print("| Test Result\tAcc@1: %.2f%%" %(acc))

    sys.exit(0)

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+file_name+'.t7')
    net = checkpoint['net']
    start_epoch = checkpoint['epoch']
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    net.apply(conv_init)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch):
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.item(), 100.*correct/total))
        sys.stdout.flush()

def test(epoch):
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint
        acc = 100.*correct/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))

        if epoch % args.save_freq == 0:
            print('| Saving model...\t\t\tTop1 = %.2f%%' %(acc))
            state = {
                    'net':net.module if use_cuda else net,
                    'acc':acc,
                    'epoch':epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'
            if not os.path.isdir(save_point):
                os.mkdir(save_point)
            torch.save(state, save_point+file_name+'.t7')
            

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))
