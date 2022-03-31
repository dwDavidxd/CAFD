import torchvision.transforms as transforms
import torchvision

import os, imageio, argparse

from networks import *
import torch.backends.cudnn as cudnn
from utils.cam_pgd_attack import LinfCAMAttack
from utils.BalancedDataParallel import BalancedDataParallel


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def getNetwork(args):
    num_classes = 10
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


def get_last_conv_name(net):

    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name

'''
class CAM_tensor_simple(object):

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.weight = None
        self.net.eval()
        self.handlers = []
        self._register_hook()
        self._get_weight()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        # print("feature shape:{}".format(output.size()))

    def _get_weight(self):
        params = list(self.net.parameters())
        self.weight = params[-2].squeeze()

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]

        index = np.argmax(output.cpu().data.numpy())

        weight = self.weight[index]  # [C]

        feature = self.feature[0]  # [C,H,W]

        cam = feature * weight.unsqueeze(-1).unsqueeze(-1)  # [C,H,W]
        # cam = cam.sum(axis=0)  # [H,W]
        cam = torch.max(cam, torch.zeros(cam.size()).cuda())  # ReLU


        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)

        cam = torch.nn.functional.interpolate(cam.unsqueeze(0), (32, 32), mode='bilinear')

        return cam
'''

'''
class CAM_tensor_nohook(object):
    
    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = int(layer_name)
        self.weight = None
        self.net.eval()
        self.handlers = []
        self._get_weight()
        self.feature_net = nn.Sequential(*list(net.module.features))[:self.layer_name].eval()
        self.feature_net = BalancedDataParallel(5, self.feature_net, dim=0).cuda()
        print(self.feature_net)
        
    def _get_weight(self):
        params = list(self.net.parameters())
        self.weight = params[-2].squeeze()

    def __call__(self, inputs):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]

        index = np.argmax(output.cpu().data.numpy(), axis=1)

        weight = torch.zeros((inputs.size(0), self.weight.size(1))).cuda()

        weight[:] = self.weight[index[:]]

        feature = self.feature_net(inputs)  # [B,C,H,W]

        cam = feature * weight.unsqueeze(-1).unsqueeze(-1)  # [B,C,H,W]
        # cam = cam.sum(axis=0)  # [H,W]
        cam = torch.max(cam, torch.zeros(cam.size()).cuda())  # ReLU


        cam = cam.clone() - torch.min(cam.clone())
        cam = cam.clone() / torch.max(cam.clone())

        # print(cam.size())
        cam = torch.nn.functional.interpolate(cam, (32, 32), mode='bilinear')

        return cam
'''

'''
class CAM_tensor_oneGPU(object):

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.weight = None
        self.net.eval()
        self.handlers = []
        self._register_hook()
        self._get_weight()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        # print("feature shape:{}".format(output.size()))

    def _get_weight(self):
        params = list(self.net.parameters())
        self.weight = params[-2].squeeze()

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy(), axis=1)

        weight = torch.zeros((inputs.size(0), self.weight.size(1))).cuda()

        weight[:] = self.weight[index[:]]

        feature = self.feature  # [B,C,H,W]

        cam = feature * weight.unsqueeze(-1).unsqueeze(-1)  # [B,C,H,W]
        # cam = cam.sum(axis=0)  # [H,W]
        cam = torch.max(cam, torch.zeros(cam.size()).cuda())  # ReLU

        cam = cam.clone() - torch.min(cam.clone())
        cam = cam.clone() / torch.max(cam.clone())

        # print(cam.size())
        cam = torch.nn.functional.interpolate(cam, (32, 32), mode='bilinear')
        # cam = cam.squeeze()
        # resize to 224*224
        # cam = cv2.resize(cam, (32, 32))

        return cam
'''

class CAM_feature_tensor(object):

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = {}
        self.weight = None
        # self.net.eval()
        self.handlers = []
        self._register_hook()
        self._get_weight()

    def _get_features_hook(self, module, input, output):
        self.feature[input[0].device] = output
        # print("feature shape:{}".format(output.size()))

    def _get_weight(self):
        params = list(self.net.parameters())
        self.weight = params[-2].squeeze()

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs):

        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]

        index = np.argmax(output.cpu().data.numpy(), axis=1)

        weight = torch.zeros((inputs.size(0), self.weight.size(1))).cuda()

        weight[:] = self.weight[index[:]]

        feature = []

        for i in self.feature:
            feature.append(self.feature[i].to(torch.device("cuda:0")))

        feature = torch.cat(feature[:], 0)

        cam = feature #* weight.unsqueeze(-1).unsqueeze(-1)  # [B,C,H,W]
        # cam = cam.sum(axis=0)  # [H,W]
        #cam = torch.max(cam, torch.zeros(cam.size()).cuda())  # ReLU

        #cam = cam.clone() - torch.min(cam.clone())
        #cam = cam.clone() / torch.max(cam.clone())

        '''
        for i in range(cam.size(0)):
            cam[i] = cam[i].clone() - torch.min(cam[i].clone())
            cam[i] = cam[i].clone() / torch.max(cam[i].clone())
        '''
        cam = torch.nn.functional.interpolate(cam, (32, 32), mode='bilinear')

        return cam, output

class CAM_tensor(object):

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = {}
        self.weight = None
        # self.net.eval()
        self.handlers = []
        self._register_hook()
        self._get_weight()

    def _get_features_hook(self, module, input, output):
        self.feature[input[0].device] = output
        # print("feature shape:{}".format(output.size()))

    def _get_weight(self):
        params = list(self.net.parameters())
        self.weight = params[-2].squeeze()

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]

        index = np.argmax(output.cpu().data.numpy(), axis=1)

        weight = torch.zeros((inputs.size(0), self.weight.size(1))).cuda()

        weight[:] = self.weight[index[:]]

        feature = []

        for i in self.feature:
            feature.append(self.feature[i].to(torch.device("cuda:0")))

        feature = torch.cat(feature[:], 0)

        cam = feature * weight.unsqueeze(-1).unsqueeze(-1)  # [B,C,H,W]
        # cam = torch.sum(cam, dim=1)  # [H,W]

        cam = torch.max(cam, torch.zeros(cam.size()).cuda())  # ReLU

        cam = cam.clone() - torch.min(cam.clone())
        cam = cam.clone() / torch.max(cam.clone())

        '''
        for i in range(cam.size(0)):
            cam[i] = cam[i].clone() - torch.min(cam[i].clone())
            cam[i] = cam[i].clone() / torch.max(cam[i].clone())
        '''
        cam = torch.nn.functional.interpolate(cam, (32, 32), mode='bilinear')

        return cam


'''
class cam_criteria_simple(nn.Module):
    def __init__(self, cam):
        super(cam_criteria_simple, self).__init__()
        # you can try other losses
        self.cam = cam
        self.mse = nn.MSELoss()

    def forward(self, adv, org):

        for num in range(adv.size()[0]):

            mask_adv_temp = self.cam(adv[num].unsqueeze(0))
            mask_ori_temp = self.cam(org[num].unsqueeze(0))

            if num == 0:
                mask_adv = mask_adv_temp
                mask_ori = mask_ori_temp
            else:
                mask_adv = torch.cat((mask_adv, mask_adv_temp), 0)
                mask_ori = torch.cat((mask_ori, mask_ori_temp), 0)
        out = self.mse(mask_adv, mask_ori)
        return out
'''

class cam_std_criteria(nn.Module):
    def __init__(self, cam):
        super(cam_std_criteria, self).__init__()
        # you can try other losses
        self.cam = cam
        self.mse = nn.MSELoss()

    def forward(self, adv, org):

        mask_adv = self.cam(adv)
        mask_ori = self.cam(org)

        mask_adv = mask_adv.view(mask_adv.size(0), mask_adv.size(1), -1)
        mask_ori = mask_ori.view(mask_ori.size(0), mask_ori.size(1), -1)

        mask_adv = torch.std(mask_adv, dim=2)
        mask_ori = torch.std(mask_ori, dim=2)

        out = self.mse(mask_adv, mask_ori) * 100 # - torch.mean(torch.mean(mask_adv, dim=1))

        # print('loss:' + str(out.item()), flush=True)
        # print('std:' + str(torch.mean(torch.mean(mask_adv, dim=1)).item()), flush=True)

        return out


class cam_criteria(nn.Module):
    def __init__(self, cam):
        super(cam_criteria, self).__init__()
        # you can try other losses
        self.cam = cam
        self.mse = nn.MSELoss()

    def forward(self, adv, org):

        mask_adv = self.cam(adv)
        mask_ori = self.cam(org)

        out = self.mse(mask_adv, mask_ori)
        return out


class cam_feature_criteria(nn.Module):
    def __init__(self, cam):
        super(cam_feature_criteria, self).__init__()
        # you can try other losses
        self.cam = cam
        self.mse = nn.MSELoss()

    def forward(self, adv, org):

        mask_adv, weight_adv = self.cam(adv)
        mask_ori, weight_ori = self.cam(org)

        out1 = self.mse(mask_adv, mask_ori)
        out2 = torch.sum(torch.abs(weight_adv - weight_ori)) / weight_adv.size(0)
        return out1, out2


)

def main(args):
    setup_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    if not os.path.exists(args.savedircln):
        os.makedirs(args.savedircln)

    if not os.path.exists(args.savediradv):
        os.makedirs(args.savediradv)

    if not os.path.exists(args.savedirclnnpy):
        os.makedirs(args.savedirclnnpy)

    if not os.path.exists(args.savediradvnpy):
        os.makedirs(args.savediradvnpy)

    # GPU
    use_cuda = torch.cuda.is_available()

    batch_size = args.batch_size
    trans = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])

    # data_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=trans)
    data_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=trans)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # load target network
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/' + args.dataset + os.sep + file_name + '.t7')
    net = checkpoint['net']

    if use_cuda:
        # net.cuda()
        # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        net = BalancedDataParallel(5, net, dim=0).cuda()
        cudnn.benchmark = True

    net.eval()
    net.training = False

    layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name
    cam = CAM_tensor(net, layer_name)

    loss_fn = cam_criteria(cam).cuda() 
    # loss_fn = ssp_criteria(net, args.ssp_layer).cuda()


    adversary = LinfCAMAttack(
        [net], loss_fn=loss_fn, eps=args.eps,
        nb_iter=args.iters, eps_iter=args.step_size, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)

    # start train
    cnt = 0
    counter = 0
    for i, (img, label) in enumerate(data_loader):

        if use_cuda:
            img, label = img.cuda(), label.cuda()

        adv = adversary.perturb(img, None)

        # Distance between image and adversary
        print((adv-img).max()*255)

        # predict

        outputs = net(adv)
        _, predicted = torch.max(outputs.data, 1)
        total = label.size(0)
        correct = predicted.eq(label.data).cpu().sum()

        acc = 100.*correct/total
        print("| Test Attack Result\tAcc@1: %.2f%%" %(acc))

        outputs = net(img)
        _, predicted = torch.max(outputs.data, 1)
        total = label.size(0)
        correct = predicted.eq(label.data).cpu().sum()

        acc = 100. * correct / total
        print("| Test Clean Result\tAcc@1: %.2f%%" % (acc))

        for img_index in range(adv.size()[0]):
            cnt += 1
            cln_path = os.path.join(args.savedircln, (str(cnt)+'.png'))
            adv_path = os.path.join(args.savediradv, (str(cnt) + '.png'))
            cln_path_npy = os.path.join(args.savedirclnnpy, (str(cnt) + '.npy'))
            adv_path_npy = os.path.join(args.savediradvnpy, (str(cnt) + '.npy'))

            cln_to_save = np.transpose(img[img_index, :, :, :].detach().cpu().numpy(), (1, 2, 0))
            adv_to_save = np.transpose(adv[img_index, :, :, :].detach().cpu().numpy(), (1, 2, 0))

            np.save(cln_path_npy, cln_to_save)
            np.save(adv_path_npy, adv_to_save)

            cln_to_save = (cln_to_save * 255).round().astype(np.uint8)
            imageio.imwrite(cln_path, cln_to_save, format='png')
            adv_to_save = (adv_to_save * 255).round().astype(np.uint8)
            imageio.imwrite(adv_path, adv_to_save, format='png')

        counter += 1

        print('Number of Images Processed:', (i + 1) * batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CAM Attack')
    parser.add_argument('--savedircln', default='./results/clean/test_clean_img')
    parser.add_argument('--savediradv', default='./results/CAM_pgd_adver/cam_mse/test/adver_img')
    parser.add_argument('--savedirclnnpy', default='./results/clean/test_clean_npy')
    parser.add_argument('--savediradvnpy', default='./results/CAM_pgd_adver/cam_mse/test/adver_npy')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size')
    parser.add_argument('--eps', type=int, default=8/255, help='pertrbation budget')
    parser.add_argument('--step_size', type=float, default=0.01, help='Step size')
    parser.add_argument('--iters', type=int, default=100, help='Number of SSP Iterations')
    parser.add_argument('--ssp_layer', type=int, default=50, help='VGG layer that is going to be used in SSP')
    parser.add_argument('--net_type', default='vggnet', type=str, help='model')
    parser.add_argument('--depth', default=19, type=int, help='depth of model')
    parser.add_argument('--widen_factor', default=20, type=int, help='width of model')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
    parser.add_argument('--layer-name', type=str, default=None, help='last convolutional layer name')
    # parser.add_argument('--classid', type=int, default=None, help='class id')

    args = parser.parse_args()
    print(args)

    main(args)
