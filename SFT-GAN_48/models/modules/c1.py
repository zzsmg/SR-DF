"""
作者：86185
日期：2021年03月03日
"""
import os

import torch
import torch.nn as nn
from PIL import Image

from mit_semseg.lib.nn import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d
# from modules.utils.util import colorEncode
# import csv
# from scipy.io import loadmat
# colors = loadmat('/media/zz/Work1/original code/semantic-segmentation-pytorch-master/data/color150.mat')['colors']
# names = {}
# with open('/media/zz/Work1/original code/semantic-segmentation-pytorch-master/data/object150_info.csv') as f:
#     reader = csv.reader(f)
#     next(reader)
#     for row in reader:
#         names[int(row[0])] = row[5].split(";")[0]

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )
# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=720, use_softmax=True):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out):
        conv5 = conv_out[-1]
        # print(conv5)
        x = self.cbr(conv5)
        # print(x)
        x = self.conv_last(x)
        segSize = (conv5.size()[2],conv5.size()[3])

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x
def visualize_result(data, pred):
    (img, info) = data

    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)
    print(pred_color.shape)
    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save("./test.png")
if __name__ == '__main__':

    import hrnet
    import cv2
    from torchvision import transforms
    import numpy as np
    path = "/media/zz/Work1/datasets/DIV2K/DIV2K_train_LR_bicubic/X4/0406x4.png"
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # print(img)
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225])
    img = np.float32(np.array(img)) / 255.

    img = img.transpose((2, 0, 1))
    # img = normalize(torch.from_numpy(img.copy()))
    img = torch.unsqueeze(torch.from_numpy(img.copy()), 0).cuda()
    print(img)

    encode_model = hrnet.__dict__['hrnetv2'](pretrained=True, use_input_norm=True).cuda().eval()
    with torch.no_grad():
        encode, M = encode_model(img, return_feature_maps=True)
        # print(encode)
    # encode = torch.unsqueeze(encode, 0).cuda()
    print(M[0])
    decode = C1().cuda().eval()
    decode.load_state_dict(
        torch.load("/media/zz/Work1/Pycharm Projects/mysrnet/trained_model/decoder_epoch_30.pth", map_location=lambda storage, loc: storage), strict=False)
    pred = decode(encode)
    _,seg = torch.max(pred, dim=1)
    # seg = seg.detach().cpu().squeeze().numpy()
    print(seg)
    # img = img.detach().cpu().squeeze().numpy()
    # img = np.resize(img, (3, 118, 180))
    # img = img.transpose((1, 2, 0))
    # print(img.shape)
    # visualize_result((img, "test"), seg)
