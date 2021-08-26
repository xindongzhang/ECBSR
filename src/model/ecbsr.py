from model import common
# import common
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchsummary

from model.ecb import ECB

def make_model(args, parent=False):
    return ECBSR(args)

class ECBSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ECBSR, self).__init__()

        scale = args.scale[0]
        self.scale = scale
        self.n_stage  = args.m_ecbsr
        self.n_feats  = args.c_ecbsr
        self.with_idt = args.idt_ecbsr 
        self.n_colors = args.n_colors

        self.dm = args.dm_ecbsr
        self.act_type = args.act

        self.head = nn.Sequential(
            ECB(self.n_colors, out_planes=self.n_feats, depth_multiplier=self.dm, act_type=self.act_type, with_idt = self.with_idt)
        )

        modules = []
        for i in range(self.n_stage):
            modules.append(ECB(self.n_feats, out_planes=self.n_feats, depth_multiplier=self.dm, act_type=self.act_type, with_idt = self.with_idt))
        self.body = nn.Sequential(*modules)

        self.tail = nn.Sequential(
            ECB(self.n_feats, out_planes=self.scale * self.scale * self.n_colors, depth_multiplier=self.dm, act_type='linear', with_idt = self.with_idt)
        )

        self.upsampler = nn.Sequential(
            nn.PixelShuffle(self.scale)
        )

    def forward(self, x):
        head = self.head(x)
        body = self.body(head)
        out = self.tail(body) + x
        out  = self.upsampler(out)
        return out

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))

if __name__ == "__main__":
    pass