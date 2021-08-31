import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainSR(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors):
        super(PlainSR, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        backbone = []
        backbone += [nn.Conv2d(self.colors, self.channel_nums, kernel_size=3, padding=1), nn.PReLU(num_parameters=self.channel_nums)]
        for i in range(self.module_nums):
            backbone += [nn.Conv2d(self.channel_nums, self.channel_nums, kernel_size=3, padding=1), nn.PReLU(num_parameters=self.channel_nums)]
        backbone += [nn.Conv2d(self.channel_nums, self.colors*self.scale*self.scale, kernel_size=3, padding=1)]

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        y = self.backbone(x) + x
        y = self.upsampler(y)
        return y
