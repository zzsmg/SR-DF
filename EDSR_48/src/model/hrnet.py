"""
This HRNet implementation is modified from the following repository:
https://github.com/HRNet/HRNet-Semantic-Segmentation
"""
import glob
import logging
import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .utils import load_url
from mit_semseg.lib.nn import SynchronizedBatchNorm2d
# from models.modules.utils.nn.modules.batchnorm import SynchronizedBatchNorm2d


BatchNorm2d = SynchronizedBatchNorm2d
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


__all__ = ['hrnetv2']


model_urls = {
    'hrnetv2': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/hrnetv2_w48-imagenet.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):

        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        fea = x[0].detach()
        global fea_list
        fea_list = []

        for i in range(4):
            fea = self.branches[0][i](fea)
            fea_list.append(fea)


        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=(height_output, width_output),
                        mode='bilinear',
                        align_corners=False)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HRNetV2(nn.Module):
    def __init__(self, n_class, use_input_norm=False, **kwargs):
        super(HRNetV2, self).__init__()
        extra = {
            'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 'NUM_BLOCKS': (4, 4), 'NUM_CHANNELS': (48, 96), 'FUSE_METHOD': 'SUM'},
            'STAGE3': {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 'NUM_BLOCKS': (4, 4, 4), 'NUM_CHANNELS': (48, 96, 192), 'FUSE_METHOD': 'SUM'},
            'STAGE4': {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC', 'NUM_BLOCKS': (4, 4, 4, 4), 'NUM_CHANNELS': (48, 96, 192, 384), 'FUSE_METHOD': 'SUM'},
            'FINAL_CONV_KERNEL': 1
            }
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        # print(self.bn1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer) #2
        num_branches_pre = len(num_channels_pre_layer) #1

        transition_layers = []
        for i in range(num_branches_cur):  #当前通道数
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,#'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 'NUM_BLOCKS': (4, 4), 'NUM_CHANNELS': (48, 96), 'FUSE_METHOD': 'SUM'
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    # def forward(self, x, return_feature_maps=False):
    #     # print(x)
    #     x = self.conv1(x)
    #     # print(x)
    #     x = self.bn1(x)
    #     # print(x)
    #     x = self.relu(x)
    #     x = self.conv2(x)
    #     # print(x)
    #     z = []
    #     x = self.bn2(x)
    #     x = self.relu(x)
    #     y = x.detach()
    #     for module in self.layer1:
    #         y = module(y)
    #         z.append(y)
    #     # for module in z:
    #     #     print(module.size())
    #
    #     x = self.layer1(x)
    #     print(len(z))
    #
    #
    #     x_list = []
    #     for i in range(self.stage2_cfg['NUM_BRANCHES']):
    #         if self.transition1[i] is not None:
    #
    #             x_list.append(self.transition1[i](x))
    #         else:
    #             x_list.append(x)
    #
    #     for block in self.stage2:
    #         x_list = block(x_list)
    #         z.append(x_list[0])
    #     z += fea_list
    #     print(len(z)) # 9
    #     y_list = x_list.copy()
    #
    #     x_list = []
    #     for i in range(self.stage3_cfg['NUM_BRANCHES']):
    #         if self.transition2[i] is not None:
    #             x_list.append(self.transition2[i](y_list[-1]))
    #         else:
    #             x_list.append(y_list[i])
    #
    #     for block in self.stage3:
    #         x_list = block(x_list)
    #         z.append(x_list[0])
    #     y_list = x_list.copy()
    #     print(len(z)) # 13
    #     x_list = []
    #     for i in range(self.stage4_cfg['NUM_BRANCHES']):
    #         if self.transition3[i] is not None:
    #             x_list.append(self.transition3[i](y_list[-1]))
    #         else:
    #             x_list.append(y_list[i])
    #
    #     for block in self.stage4:
    #         x_list = block(x_list)
    #         z.append(x_list[0])
    #     print(len(z)) # 16
    #     x = x_list
    #     x0_h, x0_w = x[0].size(2), x[0].size(3)
    #     # print(x[1].size(),x[2].size(),x[3].size())
    #     x1 = F.interpolate(
    #         x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
    #     x2 = F.interpolate(
    #         x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
    #     x3 = F.interpolate(
    #         x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
    #
    #     hrouput = torch.cat([x[0], x1, x2, x3], 1)
    #
    #
    #     return z, hrouput
    def forward(self, x, return_feature_maps=False):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        z = []
        z.append(x.detach())
        x = self.layer1(x)

        z.append(x.detach())
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        z.append(y_list[0].detach())

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        z.append(y_list[0].detach())

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        z.append(x[0].detach())

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(
            x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x2 = F.interpolate(
            x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x3 = F.interpolate(
            x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=False)

        x = torch.cat([x[0], x1, x2, x3], 1)

        # x = self.last_layer(x)
        return [x], z

def hrnetv2(pretrained=False, **kwargs):
    model = HRNetV2(n_class=1000, **kwargs)
    if pretrained:
        # model.load_state_dict(load_url(model_urls['hrnetv2']), strict=False)
        model_path = '/media/zz/Work1/Pycharm Projects/mysrnet/trained_model/encoder_epoch_30.pth'
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage),  strict=False)
    return model
if __name__ == '__main__':
    import torch.nn as nn
    from data import util
    import numpy as np
    import matplotlib
    from matplotlib import colors
    # matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_test = hrnetv2(pretrained=True, use_input_norm=True).cuda()
    test_img_folder = "/media/zz/Work1/Pycharm Projects/mysrnet/OutdoorSceneTrain_v2/animal"
    for idx, path in enumerate(glob.glob(test_img_folder + '/*')):
        imgname = os.path.basename(path)
        basename = os.path.splitext(imgname)[0]
        print(idx + 1, basename)
        # read image
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_show = img
        img = util.modcrop(img, 8)
        img = img * 1.0 / 255
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        # MATLAB imresize
        # You can use the MATLAB to generate LR images first for faster imresize operation
        img_LR = util.imresize(img, 1 / 4, antialiasing=True)
        img_LR = util.imresize(img_LR, 4, antialiasing=True)
        img_LR = img_LR.unsqueeze(0)
        img_LR = img_LR.cuda()

        # #
        with torch.no_grad():
            _, _, output= model_test(img_LR)
            # output = nn.Softmax(1)(output[-1])
            output = output.squeeze().cpu().numpy()
        Nr = 4
        Nc = 6
        fig, axs = plt.subplots(Nr, Nc)
        fig.suptitle('Multiple images')

        images = []
        for i in range(Nr):
            for j in range(Nc):
                # Generate data with a range that varies from one plot to the next.
                data = output[6 * j + i]
                images.append(axs[i, j].imshow(data))
                axs[i, j].label_outer()

        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # for im in images:
        #     im.set_norm(norm)
        plt.show()
        # plt.figure()
        # for i in range(3):
        #     for j in range(3):
        #         ax = plt.subplot(2, 5, i*3+j+2)
        #         plt.xticks(())
        #         plt.yticks(())
        #         num = fea_nums[i*3+j]-1
        #         plt.imshow(output[num], cmap="gray")
        # ax = plt.subplot(2, 5, 1)
        # plt.xticks(())
        # plt.yticks(())
        # plt.imshow(img_show[:, :, [2, 1, 0]])
        # plt.savefig(f"/media/zz/Work1/Pycharm Projects/mysrnet/fea_map(384)/{basename}-fea.png",dpi=300)

        # with torch.no_grad():
        #     _, _, fea = model_test(img_LR)
        #     print(fea.size())
        # break
