import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
from torchvision import transforms
from models.modules.hrnet import hrnetv2
from models.modules.c1 import C1
import data.util as util


class LRHRDataset(data.Dataset):
    '''
    Read LR and HR image pairs.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.paths_HR = None
        self.LR_env = None  # environment for lmdb
        self.HR_env = None
        self.netHR = hrnetv2(pretrained=True, use_input_norm=True).cuda().eval()
        self.netC1 = C1().cuda().eval()
        self.netC1.load_state_dict(
            torch.load("/media/zz/Others/wk/Projects/SRMF/SFT-GAN_48/trained_model/decoder_epoch_30.pth",
                       map_location=lambda storage, loc: storage), strict=False)
        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_HR = sorted([os.path.join(opt['dataroot_HR'], line.rstrip('\n')) \
                        for line in f])
            if opt['dataroot_LR'] is not None:
                raise NotImplementedError('Now subset only supports generating LR on-the-fly.')
        else:  # read image list from lmdb or image files
            self.HR_env, self.paths_HR = util.get_image_paths(opt['data_type'], opt['dataroot_HR'])
            self.LR_env, self.paths_LR = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])

        assert self.paths_HR, 'Error: HR path is empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                'HR and LR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_HR))

        self.random_scale_list = [1]
        # self.HR_size_list = [96, 256, 512]

    def __getitem__(self, index):
        # print("+++")
        HR_path, LR_path = None, None
        scale = self.opt['scale']
        HR_size = self.opt['HR_size']
        # HR_size = random.choice(self.HR_size_list)
        # get HR image
        HR_path = self.paths_HR[index]
        img_HR = util.read_img(self.HR_env, HR_path)
        # print(img_HR)

        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_HR = util.modcrop(img_HR, scale * 4)

        # change color space if necessary
        if self.opt['color']:
            img_HR = util.channel_convert(img_HR.shape[2], self.opt['color'], [img_HR])[0]

        # get LR image
        if self.paths_LR:
            LR_path = self.paths_LR[index]
            img_LR = util.read_img(self.LR_env, LR_path)
            img_LR = util.modcrop(img_LR, scale)
            img_fake_HR = img_LR

        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                # img_HR = util.mincrop(img_HR, 1000, 1000)
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_HR.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, HR_size)
                W_s = _mod(W_s, random_scale, scale, HR_size)
                img_HR = cv2.resize(np.copy(img_HR), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                # force to 3 channels
                if img_HR.ndim == 2:
                    img_HR = cv2.cvtColor(img_HR, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_HR.shape
            # using matlab imresize
            img_LR = util.imresize_np(img_HR, 1 / scale, True)
            img_fake_HR = util.imresize_np(img_LR, scale, True)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)
        # get
        # print(img_fake_HR)

        img_tensor = torch.from_numpy(np.transpose(img_fake_HR[:, :, [2, 1, 0]], (2, 0, 1))).float().unsqueeze(0).cuda()
        img_fea, img_mfea = self.netHR(img_tensor)
        # print(img_fea[0])
        # exit()
        img_seg = self.netC1(img_fea)
        img_seg = img_seg.detach().squeeze(0).cpu().numpy()
        for i in range(len(img_mfea)):
            img_mfea[i] = img_mfea[i].detach().squeeze(0).cpu().numpy()

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_HR.shape
            if H < HR_size or W < HR_size:
                img_HR = cv2.resize(
                    np.copy(img_HR), (HR_size, HR_size), interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LR = util.imresize_np(img_HR, 1 / scale, True)
                # img_fake_HR = util.imresize_np(img_LR, scale, True)
                if img_LR.ndim == 2:
                    img_LR = np.expand_dims(img_LR, axis=2)

            H, W, C = img_LR.shape
            LR_size = HR_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size -1))
            rnd_w = random.randint(0, max(0, W - LR_size -1))
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            img_seg = img_seg[:, rnd_h // 4:rnd_h // 4 + LR_size // 4, rnd_w // 4:rnd_w // 4 + LR_size // 4]

            for i in range(len(img_mfea)):
                img_mfea[i] = img_mfea[i][:, rnd_h // 4:rnd_h // 4 + LR_size // 4, rnd_w // 4:rnd_w // 4 + LR_size // 4]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            img_HR = img_HR[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, :]
            # img_fake_HR = img_fake_HR[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, :]
            # augmentation - flip, rotate
            img_LR, img_HR = util.augment([img_LR, img_HR], self.opt['use_flip'], \
                self.opt['use_rot'])

            if 'building' in HR_path:
                category = 1
            elif 'plant' in HR_path:
                category = 2
            elif 'mountain' in HR_path:
                category = 3
            elif 'water' in HR_path:
                category = 4
            elif 'sky' in HR_path:
                category = 5
            elif 'grass' in HR_path:
                category = 6
            elif 'animal' in HR_path:
                category = 7
            else:
                category = 0  #
        else:
            category = -1  # during val, useless
        # change color space if necessary
        if self.opt['color']:
            img_LR = util.channel_convert(C, self.opt['color'], [img_LR])[0] # TODO during val no definetion

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
            # ftgan seg not change rgb -> bgr
            # img_fake_HR = img_fake_HR[:, :, [2, 1, 0]]
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()
        # img_fake_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_fake_HR, (2, 0, 1)))).float()
        # img_fake_HR = img_fake_HR / 255
        # normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225])
        # img_fake_HR = normalize(img_fake_HR)
        # img_HR = normalize(img_HR)
        if LR_path is None:
            LR_path = HR_path
        # if self.opt['phase'] == 'train':
        #     if random.random() < 0.1:
        #         img_LR *= 0


        return {'LR': img_LR, 'HR': img_HR, 'LR_path': LR_path, 'HR_path': HR_path, 'category': category,
                "multiple_fea": img_mfea}


    def __len__(self):
        return len(self.paths_HR)
