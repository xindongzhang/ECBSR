import os
import glob
import random
import pickle

import numpy as np
import imageio
import torch
import torch.utils.data as data
import skimage.color as sc
import time
from utils import ndarray2tensor

class DIV2K(data.Dataset):
    def __init__(self, HR_folder, LR_folder, train=True, augment=True, scale=2, colors=1, patch_size=96, repeat=168, store_in_ram=True):
        super(DIV2K, self).__init__()
        self.HR_folder = HR_folder
        self.LR_folder = LR_folder
        self.augment   = augment
        self.train     = train
        self.img_postfix = '.png'
        self.scale = scale
        self.colors = colors
        self.store_in_ram = store_in_ram
        self.patch_size = patch_size
        self.repeat = repeat
        self.nums_trainset = 0

        self.hr_filenames = []
        self.lr_filenames = []
        ## generate dataset
        start_idx = 0
        end_idx = 0
        if self.train: start_idx, end_idx = 1, 801
        else:          start_idx, end_idx = 801, 901

        for i in range(start_idx, end_idx):
            idx = str(i).zfill(4)
            hr_filename = os.path.join(self.HR_folder, idx + self.img_postfix)
            lr_filename = os.path.join(self.LR_folder, 'X{}'.format(self.scale), idx + 'x{}'.format(self.scale) + self.img_postfix)
            self.hr_filenames.append(hr_filename)
            self.lr_filenames.append(lr_filename)
        self.nums_trainset = len(self.hr_filenames)
        ## if store in ram
        self.hr_images = []
        self.lr_images = []
        if self.store_in_ram:
            LEN = end_idx - start_idx
            for i in range(LEN):
                if (i+1) % 50 == 0:
                    print("load {}-images in memory!".format(i+1))
                lr_image, hr_image = imageio.imread(self.lr_filenames[i], pilmode="RGB"), imageio.imread(self.hr_filenames[i], pilmode="RGB")
                if self.colors == 1:
                    lr_image, hr_image = sc.rgb2ycbcr(lr_image)[:, :, 0:1], sc.rgb2ycbcr(hr_image)[:, :, 0:1]
                self.hr_images.append(hr_image)
                self.lr_images.append(lr_image) 
    def __len__(self):
        if self.train:
            return self.nums_trainset * self.repeat
        else:
            return len(self.hr_filenames)
    def __getitem__(self, idx):
        # get periodic index
        idx = idx % self.nums_trainset
        # get whole image
        if self.store_in_ram:
            lr, hr = self.lr_images[idx], self.hr_images[idx]
        else:
            lr, hr = imageio.imread(self.lr_filenames[idx], pilmode="RGB"), imageio.imread(self.hr_filenames[idx], pilmode="RGB")
            if self.colors == 1:
                lr, hr = sc.rgb2ycbcr(lr)[:, :, 0:1], sc.rgb2ycbcr(hr)[:, :, 0:1]
        if self.train:
            # crop patch randomly
            lr_h, lr_w, _ = lr.shape
            hp = self.patch_size
            lp = self.patch_size // self.scale
            lx, ly = random.randrange(0, lr_w - lp + 1), random.randrange(0, lr_h - lp + 1)
            hx, hy = lx * self.scale, ly * self.scale
            lr_patch, hr_patch = lr[ly:ly+lp, lx:lx+lp, :], hr[hy:hy+hp, hx:hx+hp, :]
            # augment data
            if self.augment:
                #print("data augmentation!")
                hflip = random.random() > 0.5
                vflip = random.random() > 0.5
                rot90 = random.random() > 0.5
                if hflip: lr_patch, hr_patch = lr_patch[:, ::-1, :], hr_patch[:, ::-1, :]
                if vflip: lr_patch, hr_patch = lr_patch[::-1, :, :], hr_patch[::-1, :, :]
                if rot90: lr_patch, hr_patch = lr_patch.transpose(1,0,2), hr_patch.transpose(1,0,2)
            # numpy to tensor
            lr_patch, hr_patch = ndarray2tensor(lr_patch), ndarray2tensor(hr_patch)
            return lr_patch, hr_patch
        else:
            lr, hr = ndarray2tensor(lr), ndarray2tensor(hr)
            return lr, hr

if __name__ == '__main__':
    HR_folder = '/Users/xindongzhang/Documents/SRData/DIV2K/DIV2K_train_HR'
    LR_folder = '/Users/xindongzhang/Documents/SRData/DIV2K/DIV2K_train_LR_bicubic'
    argment   = True
    div2k = DIV2K(HR_folder, LR_folder, train=True, argment=True, scale=2, colors=1, patch_size=96, repeat=168, store_in_ram=False)

    print("numner of sample: {}".format(len(div2k)))
    start = time.time()
    for idx in range(10):
        lr, hr = div2k[idx]
        print(lr.shape, hr.shape)
    end = time.time()
    print(end - start)
