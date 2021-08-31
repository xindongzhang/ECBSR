import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import datetime
import os
import sys

def calc_psnr(sr, hr):
    diff = (sr - hr) / 255.00
    mse  = diff.pow(2).mean()
    psnr = -10 * math.log10(mse)                    
    return float(psnr)

def calc_ssim(sr, hr):
    # def ssim(
    #     X,
    #     Y,
    #     data_range=255,
    #     size_average=True,
    #     win_size=11,
    #     win_sigma=1.5,
    #     win=None,
    #     K=(0.01, 0.03),
    #     nonnegative_ssim=False,
    # )
    ssim_val = ssim(sr, hr, data_range=255, size_average=True)
    return float(ssim_val)
    
def ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    tensor = torch.from_numpy(ndarray_chw).float()
    return tensor


def cur_timestamp_str():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)

    content = "{}-{}{}-{}{}".format(year, month, day, hour, minute)
    return content


class ExperimentLogger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')
    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_stat_dict():
    stat_dict = {
        'epochs': 0,
        'losses': [],
        'ema_loss': 0.0,
        'set5': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        },
        'set14': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        },
        'b100': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        },
        'u100': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        }
    }
    return stat_dict

if __name__ == '__main__':
    timestamp = cur_timestamp_str()
    print(timestamp)