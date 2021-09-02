import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ecbsr import ECBSR
from models.plainsr import PlainSR
from torch.utils.data import DataLoader
import math
import argparse, yaml
import utils
import os
from tqdm import tqdm


parser = argparse.ArgumentParser(description='ECBSR convertor')

## yaml configuration files
parser.add_argument('--config', type=str, default=None, help = 'pre-config file for training')

## paramters for ecbsr
parser.add_argument('--scale', type=int, default=4, help = 'scale for sr network')
parser.add_argument('--colors', type=int, default=1, help = '1(Y channls of YCbCr), 3(RGB)')
parser.add_argument('--m_ecbsr', type=int, default=4, help = 'number of ecb')
parser.add_argument('--c_ecbsr', type=int, default=8, help = 'channels of ecb')
parser.add_argument('--idt_ecbsr', type=int, default=0, help = 'incorporate identity mapping in ecb or not')
parser.add_argument('--act_type', type=str, default='relu', help = 'prelu, relu, splus, rrelu')
parser.add_argument('--pretrain', type=str, default=None, help = 'path of pretrained model')

parser.add_argument('--target_frontend', type=str, default='onnx', help = 'target front-end for inference engine, e.g. onnx/pb/tflite')
parser.add_argument('--output_folder', type=str, default='./', help = 'output folder')

parser.add_argument('--is_dynamic_batches', type=int, default=0, help = 'dynamic batches or not')
parser.add_argument('--inp_n', type=int, default=1, help = 'batch size of input data')
parser.add_argument('--inp_c', type=int, default=1, help = 'channel size of input data')
parser.add_argument('--inp_h', type=int, default=270, help = 'height of input data')
parser.add_argument('--inp_w', type=int, default=480, help = 'width of input data')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)

    device = torch.device('cpu')
    ## definitions of model, loss, and optimizer
    model_ecbsr = ECBSR(module_nums=args.m_ecbsr, channel_nums=args.c_ecbsr, with_idt=args.idt_ecbsr, act_type=args.act_type, scale=args.scale, colors=args.colors).to(device)
    model_plain = PlainSR(module_nums=args.m_ecbsr, channel_nums=args.c_ecbsr, act_type=args.act_type, scale=args.scale, colors=args.colors).to(device)
    
    if args.pretrain is not None:
        print("load pretrained model: {}!".format(args.pretrain))
        model_ecbsr.load_state_dict(torch.load(args.pretrain))
    else:
        raise ValueError('the pretrain path is invalud!')
    
    ## copy weights from ecbsr to plainsr
    depth = len(model_ecbsr.backbone)
    for d in range(depth):
        module = model_ecbsr.backbone[d]
        act_type = module.act_type
        RK, RB = module.rep_params()
        model_plain.backbone[d].conv3x3.weight.data = RK
        model_plain.backbone[d].conv3x3.bias.data = RB

        if act_type == 'relu':     pass
        elif act_type == 'linear': pass
        elif act_type == 'prelu':  model_plain.backbone[d].act.weight.data = module.act.weight.data
        else: raise ValueError('invalid type of activation!')
    
    ## convert model to onnx
    output_name = utils.cur_timestamp_str()
    if args.target_frontend == 'onnx':
        output_name = os.path.join(args.output_folder, output_name + '.onnx')
        batch_size = args.inp_n
        fake_x = torch.rand(batch_size, args.inp_c, args.inp_h, args.inp_w, requires_grad=False)

        dynamic_params = None
        if args.is_dynamic_batches:
            dynamic_params = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

        torch.onnx.export(
            model_plain, 
            fake_x, 
            output_name, 
            export_params=True, 
            opset_version=10, 
            do_constant_folding=True, 
            input_names=['input'], 
            output_names=['output'],
            dynamic_axes=dynamic_params
        )

    elif args.target_frontend == 'pb':
        output_name = os.path.join(args.output_folder, output_name + '.pb')
        raise ValueError('TBD')
    elif args.target_frontend == 'tflite':
        output_name = os.path.join(args.output_folder, output_name + '.tflite')
        raise ValueError('TBD')
    else:
        raise ValueError('invalid type of frontend!')
    