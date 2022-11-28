'''
Codes for testing YSRNET
'''

import os
import glob
import numpy as np
import cv2
import torch
from data import util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# import architectures as arch
from models.modules import c1, hrnet, sft_arch
import models.modules.seg_arch as seg_arch
# from plt_numpy import plt_from_numpy

# options
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_path = '/media/zz/Others/wk/Projects/SRMF/SRMF_5/experiments/ade20k-seg150_mynet_4_OST_6th/models/200000_G.pth'  # torch version


test_img_folder_name = 'samples'  # image folder name
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> 'cpu'
# device = torch.device('cpu')

save_prob_path = '../data/' + test_img_folder_name + '_segprob'  # probability maps
save_byteimg_path = '../data/' + test_img_folder_name + '_byteimg'  # segmentation annotations
save_colorimg_path = '../data/' + test_img_folder_name + '_colorimg'  # segmentaion color results



test_img_folder = '/media/zz/Work1/wk/D/datasets_x2_EDSR_new_DIV800/DIV2K/DIV2K_test_HR'
save_result_path = "./result_200000_new"
os.makedirs(save_result_path)


# if 'torch' in model_path:  # torch version
#     model = arch.SFT_Net_torch()
# else:  # pytorch version
model = sft_arch.SFT_Net()
model.load_state_dict(torch.load(model_path), strict=False)
model.eval()
model = model.to(device)
encode_model = hrnet.__dict__['hrnetv2'](pretrained=True, use_input_norm=True).cuda().eval()
decode = c1.C1().cuda().eval()

decode.load_state_dict(torch.load("/media/zz/Others/wk/Projects/SRMF/SRMF_5/trained_model/decoder_epoch_30.pth", map_location=lambda storage, loc: storage), strict=False)
segnet = seg_arch.OutdoorSceneSeg().cuda()

segnet.load_state_dict(torch.load('/media/zz/Others/wk/Projects/SRMF/SRMF_5/trained_model/segmentation_OST_bic.pth'), strict=False)
segnet.eval()
print('Testing MYSRNET ...')


for idx, path in enumerate(glob.glob(test_img_folder + '/*')):
    # if "HR" not in path:
    #     continue
    imgname = os.path.basename(path)
    basename = os.path.splitext(imgname)[0]
    print(idx + 1, basename)
    # read image
    img = cv2.imread(path)
    img_show = img


    img = img * 1.0 / 255
    if img.ndim == 2:
        continue
        img = np.expand_dims(img, axis=2)

    h, w, _ = img.shape

    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

    # MATLAB imresize
    # You can use the MATLAB to generate LR images first for faster imresize operation
    img_LR = util.imresize(img, 1 / 4, antialiasing=True)
    img_FLR = util.imresize(img_LR, 4, antialiasing=True)

    img_FLR = img_FLR[[2, 1, 0], :, :]
    img_LR = img_LR[[2, 1, 0], :, :]
    img = img[[2, 1, 0], :, :]
    img = img.unsqueeze(0)
    img = img.to(device)
    img_LR = img_LR.unsqueeze(0)
    img_LR = img_LR.to(device)
    img_FLR = img_FLR.unsqueeze(0)
    img_FLR = img_FLR.to(device)

    with torch.no_grad():
        fea, mfea = encode_model(img_FLR, return_feature_maps=True)
        seg = decode(fea)
        output = model((img_LR, mfea))
        output = output.data.float().cpu().squeeze()
    output = util.tensor2img(output)
    util.save_img(output, os.path.join(save_result_path, basename + '_48.png'))

