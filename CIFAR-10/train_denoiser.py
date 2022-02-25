import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

import argparse
import time

import os

from networks.denoiser import Denoiser

from processor import AverageMeter, accuracy

from dataload import DatasetIMG_Dual, DatasetNPY_Dual
from torchvision import transforms

import math
from os.path import join
from example_cam import cam_criteria, cam_feature_criteria, CAM_tensor, get_last_conv_name, getNetwork, CAM_feature_tensor
from utils.BalancedDataParallel import BalancedDataParallel
import torch.backends.cudnn as cudnn

from networks.networks_NRP import Discriminator


irange = range


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def adjust_learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 60):
        optim_factor = 3
    elif(epoch > 50):
        optim_factor = 2
    elif(epoch > 40):
        optim_factor = 1

    return init*math.pow(0.3, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """


    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(tensor[k])
            k = k + 1
    return grid

def save_checkpoint(state, save_dir, base_name="best_model"):
    """Saves checkpoint to disk"""
    directory = save_dir
    filename = base_name + ".pth.tar"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    #im = im.convert('L')
    im.save(filename, quality=100)
    
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training. See code for default values.')

# STORAGE LOCATION VARIABLES
parser.add_argument('--traindirs_cln', default='./results/clean/train/clean_npy', type=str,
                    help='path of clean trainset')
parser.add_argument('--traindirs_adv', default='./results/CAM_pgd_adver/cam_mse/train/adver_npy', type=str,
                    help='path of adversarial trainset')
parser.add_argument('--traindirs_label', default='./results/label_true_train.pkl', type=str,
                    help='path of training label')
#parser.add_argument('--testdirs_cln', default='./results/clean/test/clean_npy', type=str,
#                    help='path of clean testset')
#parser.add_argument('--testdirs_adv', default='./results/CAM_pgd_adver/cam_mse/test/adver_npy', type=str,
#                   help='path of adversarial testset')
# parser.add_argument('--testdirs_cln', default='./data/adv_example/test/VGG/PGD_it40_8/cln_npy', type=str,
#                    help='path of clean testset')
# parser.add_argument('--testdirs_adv', default='./data/adv_example/test/VGG/PGD_it40_8/cln_npy', type=str,
#                     help='path of adversarial testset')
parser.add_argument('--testdirs_label', default='./results/label_true_test.pkl', type=str,
                   help='path of test label')
parser.add_argument('--save_dir', '--sd', default='./checkpoint_denoise/CIFAR/', type=str, help='Path to Model')
parser.add_argument('--net_type', default='vggnet', type=str, help='model')
parser.add_argument('--depth', default=19, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--Tcheckpoint', default='./checkpoint')
parser.add_argument('--layer-name', type=str, default=None, help='last convolutional layer name')
parser.add_argument('--weight_mse', default=0, type=float, help='weight_mse 0.1')
parser.add_argument('--weight_adv', default=0.005, type=float, help='weight_adv 0.001')
parser.add_argument('--weight_act', default=1000, type=float, help='weight_act')
parser.add_argument('--weight_label', default=0.001, type=float, help='weight_lable')


# MODEL HYPERPARAMETERS
parser.add_argument('--lr', default=0.001, metavar='lr', type=float, help='Learning rate')
parser.add_argument('--itr', default=70, metavar='iter', type=int, help='Number of iterations')
parser.add_argument('--batch_size', default=300, metavar='batch_size', type=int, help='Batch size')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, help='weight decay (default: 2e-4)')
parser.add_argument('--print_freq', '-p', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--save_freq', '-p', default=2, type=int, help='print frequency (default: 10)')


# OTHER PROPERTIES
parser.add_argument('--gpu', default="0,1", type=str, help='GPU devices to use (0-7) (default: 0,1)')
parser.add_argument('--mode', default=0, type=int, help='Wether to perform test without trainig (default: 0)')
parser.add_argument('--path_denoiser', default='./checkpoint_denoise/CIFAR/plus_act_10000_lable_0.01_adv_0.005/best_model.pth.tar', type=str, help='Denoiser path')
parser.add_argument('--saveroot', default='./results/CAM_pgd_adver/defense/PGD/adv', type=str, help='output images')

args = parser.parse_args()

setup_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# Other Variables

TRAIN_AND_TEST = 0
TEST = 1

save_dir = args.save_dir
start_epoch = 1

# Set Model Hyperparameters
learning_rate = args.lr
batch_size = args.batch_size
num_epochs = args.itr
print_freq = args.print_freq
use_cuda = torch.cuda.is_available()

trans = transforms.ToTensor()

train_data = DatasetNPY_Dual(imgcln_dirs=args.traindirs_cln, imgadv_dirs=args.traindirs_adv,
                                  label_dirs=args.traindirs_label, transform=trans)
test_data = DatasetNPY_Dual(imgcln_dirs=args.testdirs_cln, imgadv_dirs=args.testdirs_adv,
                                 label_dirs=args.testdirs_label, transform=trans)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)

# Load Denoiser
denoiser = Denoiser(x_h=32, x_w=32)
# denoiser = NRP(3,3,64,5)

# Load Discriminator
netD = Discriminator(3, 32)

# Load Target Model
print('\n[Test Phase] : Model setup')
assert os.path.isdir(args.Tcheckpoint), 'Error: No Tcheckpoint directory found!'
_, file_name = getNetwork(args)
checkpoint = torch.load(args.Tcheckpoint + os.sep + args.dataset + os.sep + file_name + '.t7')
target_model = checkpoint['net']
del checkpoint

if use_cuda:
    print(">>> SENDING MODEL TO GPU...")
    denoiser = BalancedDataParallel(30, denoiser, dim=0).cuda()
    target_model = BalancedDataParallel(30, target_model, dim=0).cuda()
    netD = BalancedDataParallel(30, netD, dim=0).cuda()
    cudnn.benchmark = True

target_model.eval()
target_model.training = False

# load loss
layer_name = get_last_conv_name(target_model) if args.layer_name is None else args.layer_name
ACT_stable = cam_feature_criteria(CAM_feature_tensor(target_model, layer_name)).cuda()
MSE_stable = torch.nn.MSELoss().cuda()
BCE_stable = torch.nn.BCEWithLogitsLoss().cuda()

best_pred = 0.0
worst_pred = float("inf")


def grad_step(x_batch, y_batch):
    """ Performs a step during training. """
    # Compute output for example
    logits = target_model(x_batch)
    # loss = nn.CrossEntropyLoss()(logits, y_batch)

    return logits

    # Update Mean loss for current iteration


def no_grad_step(x_batch, y_batch):
    """ Performs a step during testing."""

    with torch.no_grad():
        logits = target_model(x_batch)
        # loss = nn.CrossEntropyLoss()(logits, y_batch)

    # Update Mean loss for current iteration
    #         losses.update(loss.item(), x_batch.size(0))
    #         prec1 = accuracy(logits.data, y_batch, k=k)
    #         top1.update(prec1.item(), x_batch.size(0))

    return logits


def train(epoch):
    denoiser.train()
    netD.train()

    optimizer = optim.Adam(denoiser.parameters(), lr=adjust_learning_rate(learning_rate, epoch),
                           weight_decay=args.weight_decay)

    optimizer_D = optim.Adam(netD.parameters(), lr=adjust_learning_rate(learning_rate, epoch),
                           weight_decay=args.weight_decay)

    losses_label = AverageMeter()
    losses_mse = AverageMeter()
    losses_adv = AverageMeter()
    losses_act = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    for i, (x, x_adv, y) in enumerate(train_loader):

        t_real = torch.ones((x.size(0), 1))
        t_fake = torch.zeros((x.size(0), 1))
        if use_cuda:
            x, x_adv, y = x.cuda(), x_adv.cuda(), y.cuda()
            t_real, t_fake = t_real.cuda(), t_fake.cuda()

        # train netD
        y_pred = netD(x)
        noise = denoiser.forward(x_adv).detach()
        x_smooth = x_adv + noise
        # x_smooth = denoiser.forward(x_adv).detach()
        y_pred_fake = netD(x_smooth)

        loss_D = (BCE_stable(y_pred - torch.mean(y_pred_fake), t_real) +
                  BCE_stable(y_pred_fake - torch.mean(y_pred), t_fake)) / 2

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # 3. Compute denoised image. Need to check this...
        noise = denoiser.forward(x_adv)
        x_smooth = x_adv + noise
        # x_smooth = denoiser.forward(x_adv)

        # adv_loss
        y_pred = netD(x)
        y_pred_fake = netD(x_smooth)

        loss_adv = ((BCE_stable(y_pred - torch.mean(y_pred_fake), t_fake) +
                    BCE_stable(y_pred_fake - torch.mean(y_pred), t_real)) / 2) * args.weight_adv

        # 4. Get logits from smooth and denoised image
        logits_smooth= grad_step(x_smooth, y)
        # logits_org = grad_step(x, y)

        # 5. Compute loss

        #loss_ori = torch.sum( torch.abs(logits_smooth - logits_org) ) / x.size(0) * args.weight_ori

        loss_mse = MSE_stable(x_smooth, x) * args.weight_mse

        loss_act, loss_label = ACT_stable(x_smooth, x)

        # loss_label = loss_act

        loss_act = loss_act * args.weight_act

        loss_label = loss_label * args.weight_label

        loss = loss_mse + loss_adv + loss_act + loss_label

        # 6. Update Mean loss for current iteration
        losses_label.update(loss_label.item(), x.size(0))
        losses_mse.update(loss_mse.item(), x.size(0))
        losses_adv.update(loss_adv.item(), x.size(0))
        losses_act.update(loss_act.item(), x.size(0))
        losses.update(loss.item(), x.size(0))
        prec1 = accuracy(logits_smooth.data, y)
        top1.update(prec1.item(), x.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # Set grads to zero for new iter
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Train-Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_mse {loss_mse.val:.4f} ({loss_mse.avg:.4f})\t'
                  'Loss_adv {loss_adv.val:.4f} ({loss_adv.avg:.4f})\t'
                  'Loss_act {loss_act.val:.4f} ({loss_act.avg:.4f})\t'
                  'Loss_label {loss_label.val:.4f} ({loss_label.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, loss_mse=losses_mse, loss_adv=losses_adv, loss_act=losses_act, loss_label=losses_label, top1=top1))


def test(epoch, args):
    # global best_pred
    denoiser.eval()
    netD.eval()

    losses_label = AverageMeter()
    losses_mse = AverageMeter()
    losses_adv = AverageMeter()
    losses_act = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    with torch.no_grad():
        for i, (x, x_adv, y) in enumerate(test_loader):

            t_real = torch.ones((x.size(0), 1))
            t_fake = torch.zeros((x.size(0), 1))
            if use_cuda:
                x, x_adv, y = x.cuda(), x_adv.cuda(), y.cuda()
                t_real, t_fake = t_real.cuda(), t_fake.cuda()

            # 3. Compute denoised image. Need to check this...
            noise = denoiser.forward(x_adv)
            x_smooth = x_adv + noise

            # x_smooth = denoiser.forward(x_adv)
            # adv loss
            y_pred = netD(x)
            y_pred_fake = netD(x_smooth)

            loss_adv = ((BCE_stable(y_pred - torch.mean(y_pred_fake), t_fake) +
                        BCE_stable(y_pred_fake - torch.mean(y_pred), t_real)) / 2) * args.weight_adv

            # 4. Get logits from smooth and denoised image
            logits_smooth= grad_step(x_smooth, y)
            # logits_org = grad_step(x, y)

            # 5. Compute loss
            loss_mse = MSE_stable(x_smooth, x) * args.weight_mse

            # loss_ori = torch.sum(torch.abs(logits_smooth - logits_org)) / x.size(0) * args.weight_ori

            loss_act, loss_label = ACT_stable(x_smooth, x)

            # loss_label = loss_act

            loss_act = loss_act * args.weight_act

            loss_label = loss_label * args.weight_label

            loss = loss_mse + loss_adv + loss_act + loss_label
            # 6. Update Mean loss for current iteration
            losses_label.update(loss_label.item(), x.size(0))
            losses_mse.update(loss_mse.item(), x.size(0))
            losses_adv.update(loss_adv.item(), x.size(0))
            losses_act.update(loss_act.item(), x.size(0))
            losses.update(loss.item(), x.size(0))
            prec1 = accuracy(logits_smooth.data, y)
            top1.update(prec1.item(), x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test-Epoch: [{0}][{1}/{2}]'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss_mse {loss_mse.val:.4f} ({loss_mse.avg:.4f})\t'
                      'Loss_adv {loss_adv.val:.4f} ({loss_adv.avg:.4f})\t'
                      'Loss_act {loss_act.val:.4f} ({loss_act.avg:.4f})\t'
                      'Loss_label {loss_label.val:.4f} ({loss_label.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(test_loader), batch_time=batch_time,
                    loss=losses, loss_mse=losses_mse, loss_adv=losses_adv, loss_act=losses_act, loss_label=losses_label, top1=top1))

                out = torch.stack((x, x_smooth))  # 2, bs, 3, 32, 32
                out = out.transpose(1, 0).contiguous()  # bs, 2, 3, 32, 32
                out = out.view(-1, x.size(-3), x.size(-2), x.size(-1))

                save_image(out, join('./checkpoint_denoise', 'test_recon_{}.png'.format(i)), nrow=20)

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

        if epoch % args.save_freq == 0:
        # is_best = best_pred <= top1.avg
        # if is_best:
            # best_pred = top1.avg
            save_checkpoint(denoiser.state_dict(), save_dir)
            print('save the model')


def evaluate(path_denoiser, saveroot):
    cnt = 0
    denoiser.load_state_dict(torch.load(path_denoiser))
    denoiser.eval()

    top1 = AverageMeter()

    for i, (_, x_adv, y) in enumerate(test_loader):

        if use_cuda:
            x_adv, y = x_adv.cuda(), y.cuda()


        noise = denoiser.forward(x_adv)
        x_smooth = x_adv + noise
        # x_smooth = denoiser.forward(x_adv)

        logits_smooth = grad_step(x_smooth, y)
        prec1 = accuracy(logits_smooth.data, y)
        top1.update(prec1.item(), x_adv.size(0))

        for n in range(x_smooth.size(0)):
            cnt += 1
            out = torch.unsqueeze(x_smooth[n], 0)
            save_image(out, join(saveroot, '{}.png'.format(cnt)), nrow=1, padding=0)

    print(' * Prec@1 {top1.avg:.4f}'.format(top1=top1))


if args.mode == TRAIN_AND_TEST:
    print("==================== TRAINING ====================")
    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = ' + str(learning_rate))

    elapsed_time = 0
    for epoch in range(start_epoch, start_epoch+num_epochs):
        start_time = time.time()

        train(epoch)
        test(epoch)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' %(get_hms(elapsed_time)))

    print('\n[Phase 4] : Testing model')
    print('* Test results : Acc@1 = %.4f' %(best_pred))

if args.mode == TEST:
    print("==================== TESTING ====================")
    evaluate(args.path_denoiser, args.saveroot)
