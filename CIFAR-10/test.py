from __future__ import print_function

import torch.backends.cudnn as cudnn
import config as cf

import torchvision.transforms as transforms

import os

import argparse


from networks import *
from torch.autograd import Variable
from dataload import DatasetIMG, DatasetNPY
from torch.utils.data import DataLoader
# import pickle

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='vggnet', type=str, help='model')
parser.add_argument('--depth', default=19, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=20, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10adver'
                                         '', type=str, help='dataset = [cifar10/cifar100]')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
# start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')

# src_domain = './results/HGD/VGG/PGD/advu'
# src_domain = './results/APE/PGD/advu'
src_domain = './data/adv_example/test/VGG/FWA/advu_npy'
label_dirs = './results/label_true_test.pkl'


batch_size = 500

trans = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

test_data = DatasetNPY(img_dirs=src_domain, label_dirs=label_dirs, transform=trans)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

num_classes = 10

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


print('\n[Test Phase] : Model setup')
assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
_, file_name = getNetwork(args)
checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
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
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)

        '''
        print(predicted[7])
        print(predicted[8])
        print(predicted[9])
        print(predicted[10])
        print(predicted[11])
        '''
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # break

    acc = 100.*correct/total
    print("| Test Result\tAcc@1: %.2f%%" %(acc))
    print(100-acc)
