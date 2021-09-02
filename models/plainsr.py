import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3X3(nn.Module):
    def __init__(self, inp_planes, out_planes, act_type='prelu'):
        super(Conv3X3, self).__init__()

        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type

        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.act  = None

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        y = self.conv3x3(x)
        if self.act_type != 'linear':
            y = self.act(y)
        return y

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
        backbone += [Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type)]
        for i in range(self.module_nums):
            backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type)]
        backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.colors*self.scale*self.scale, act_type='linear')]
        self.backbone = nn.Sequential(*backbone)

        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        y = self.backbone(x) + x
        y = self.upsampler(y)
        return y
