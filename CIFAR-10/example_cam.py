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
