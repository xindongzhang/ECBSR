import torch
import torch.nn as nn
import torch.nn.functional as F
from datas.benchmark import Benchmark
from datas.div2k import DIV2K
from models.ecbsr import ECBSR
from torch.utils.data import DataLoader
import math
import argparse, yaml
import utils
import os
from tqdm import tqdm
import logging
import sys
import time


parser = argparse.ArgumentParser(description='ECBSR')

## yaml configuration files
parser.add_argument('--config', type=str, default=None, help = 'pre-config file for training')

## paramters for ecbsr
parser.add_argument('--scale', type=int, default=2, help = 'scale for sr network')
parser.add_argument('--colors', type=int, default=1, help = '1(Y channls of YCbCr)')
parser.add_argument('--m_ecbsr', type=int, default=4, help = 'number of ecb')
parser.add_argument('--c_ecbsr', type=int, default=8, help = 'channels of ecb')
parser.add_argument('--idt_ecbsr', type=int, default=0, help = 'incorporate identity mapping in ecb or not')
parser.add_argument('--act_type', type=str, default='prelu', help = 'prelu, relu, splus, rrelu')
parser.add_argument('--pretrain', type=str, default=None, help = 'path of pretrained model')

## parameters for model training
parser.add_argument('--patch_size', type=int, default=64, help = 'patch size of HR image')
parser.add_argument('--batch_size', type=int, default=32, help = 'batch size of training data')
parser.add_argument('--data_repeat', type=int, default=1, help = 'times of repetition for training data')
parser.add_argument('--data_augment', type=int, default=1, help = 'data augmentation for training')
parser.add_argument('--epochs', type=int, default=600, help = 'number of epochs')
parser.add_argument('--test_every', type=int, default=1, help = 'test the model every N epochs')
parser.add_argument('--log_every', type=int, default=1, help = 'print log of loss, every N steps')
parser.add_argument('--log_path', type=str, default="./experiments/")
parser.add_argument('--lr', type=float, default=5e-4, help = 'learning rate of optimizer')
parser.add_argument('--store_in_ram', type=int, default=0, help = 'store the whole training data in RAM or not')

## hardware specification
parser.add_argument('--gpu_id', type=int, default=0, help = 'gpu id for training')
parser.add_argument('--threads', type=int, default=1, help = 'number of threads for training')

## dataset specification
parser.add_argument('--div2k_hr_path', type=str, default='/Users/xindongzhang/Documents/SRData/DIV2K/DIV2K_train_HR', help = '')
parser.add_argument('--div2k_lr_path', type=str, default='/Users/xindongzhang/Documents/SRData/DIV2K/DIV2K_train_LR_bicubic', help = '')
parser.add_argument('--set5_hr_path', type=str, default='/Users/xindongzhang/Documents/SRData/benchmark/Set5/HR', help = '')
parser.add_argument('--set5_lr_path', type=str, default='/Users/xindongzhang/Documents/SRData/benchmark/Set5/LR_bicubic', help = '')
parser.add_argument('--set14_hr_path', type=str, default='/Users/xindongzhang/Documents/SRData/benchmark/Set14/HR', help = '')
parser.add_argument('--set14_lr_path', type=str, default='/Users/xindongzhang/Documents/SRData/benchmark/Set14/LR_bicubic', help = '')
parser.add_argument('--b100_hr_path', type=str, default='/Users/xindongzhang/Documents/SRData/benchmark/B100/HR', help = '')
parser.add_argument('--b100_lr_path', type=str, default='/Users/xindongzhang/Documents/SRData/benchmark/B100/LR_bicubic', help = '')
parser.add_argument('--u100_hr_path', type=str, default='/Users/xindongzhang/Documents/SRData/benchmark/Urban100/HR', help = '')
parser.add_argument('--u100_lr_path', type=str, default='/Users/xindongzhang/Documents/SRData/benchmark/Urban100/LR_bicubic', help = '')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)
    
    if args.colors == 3:
        raise ValueError("ECBSR is trained and tested with colors=1.")

    device = None
    if args.gpu_id >= 0 and torch.cuda.is_available():
        print("use cuda & cudnn for acceleration!")
        print("the gpu id is: {}".format(args.gpu_id))
        device = torch.device('cuda:{}'.format(args.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        print("use cpu for training!")
        device = torch.device('cpu')
    torch.set_num_threads(args.threads)

    div2k = DIV2K(
        args.div2k_hr_path, 
        args.div2k_lr_path, 
        train=True, 
        augment=args.data_augment, 
        scale=args.scale, 
        colors=args.colors, 
        patch_size=args.patch_size, 
        repeat=args.data_repeat, 
        store_in_ram=args.store_in_ram
    )

    set5  = Benchmark(args.set5_hr_path, args.set5_lr_path, scale=args.scale, colors=args.colors, store_in_ram=args.store_in_ram)
    set14 = Benchmark(args.set14_hr_path, args.set14_lr_path, scale=args.scale, colors=args.colors, store_in_ram=args.store_in_ram)
    b100  = Benchmark(args.b100_hr_path, args.b100_lr_path, scale=args.scale, colors=args.colors, store_in_ram=args.store_in_ram)
    u100  = Benchmark(args.u100_hr_path, args.u100_lr_path, scale=args.scale, colors=args.colors, store_in_ram=args.store_in_ram)

    train_dataloader = DataLoader(dataset=div2k, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    valid_dataloaders = []
    valid_dataloaders += [{'name': 'set5', 'dataloader': DataLoader(dataset=set5, batch_size=1, shuffle=False)}]
    valid_dataloaders += [{'name': 'set14', 'dataloader': DataLoader(dataset=set14, batch_size=1, shuffle=False)}]
    valid_dataloaders += [{'name': 'b100', 'dataloader': DataLoader(dataset=b100, batch_size=1, shuffle=False)}]
    valid_dataloaders += [{'name': 'u100', 'dataloader': DataLoader(dataset=u100, batch_size=1, shuffle=False)}]

    ## definitions of model, loss, and optimizer
    model = ECBSR(module_nums=args.m_ecbsr, channel_nums=args.c_ecbsr, with_idt=args.idt_ecbsr, act_type=args.act_type, scale=args.scale, colors=args.colors).to(device)
    loss_func = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.pretrain is not None:
        print("load pretrained model: {}!".format(args.pretrain))
        model.load_state_dict(torch.load(args.pretrain))
    else:
        print("train the model from scratch!")

    ## auto-generate the output logname
    timestamp = utils.cur_timestamp_str()
    experiment_name = "ecbsr-x{}-m{}c{}-{}-{}".format(args.scale, args.m_ecbsr, args.c_ecbsr, args.act_type, timestamp)
    experiment_path = os.path.join(args.log_path, experiment_name)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    experiment_model_path = os.path.join(experiment_path, 'models')
    if not os.path.exists(experiment_model_path):
        os.makedirs(experiment_model_path)

    log_name = os.path.join(experiment_path, "log.txt")
    sys.stdout = utils.ExperimentLogger(log_name, sys.stdout)
    stat_dict = utils.get_stat_dict()


    ## save training paramters
    exp_params = vars(args)
    exp_params_name = os.path.join(experiment_path, 'config.yml')
    with open(exp_params_name, 'w') as exp_params_file:
        yaml.dump(exp_params, exp_params_file, default_flow_style=False)


    timer_start = time.time()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        stat_dict['epochs'] = epoch
        print("##===========Epoch: {}=============##".format(epoch))
        for iter, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            lr, hr = batch
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = loss_func(sr, hr)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss)

            if (iter + 1) % args.log_every == 0:

                cur_steps = (iter+1)*args.batch_size
                total_steps = len(train_dataloader.dataset)
                fill_width = math.ceil(math.log10(total_steps))
                cur_steps = str(cur_steps).zfill(fill_width)

                epoch_width = math.ceil(math.log10(args.epochs))
                cur_epoch = str(epoch).zfill(epoch_width)

                avg_loss = epoch_loss / (iter + 1)
                stat_dict['losses'].append(avg_loss)

                timer_end = time.time()
                duration = timer_end - timer_start
                timer_start = timer_end
                print("Epoch:{}, {}/{}, loss: {:.4f}, time: {:.3f}".format(cur_epoch, cur_steps, total_steps, avg_loss, duration))

        if (epoch + 1) % args.test_every == 0:
            torch.set_grad_enabled(False)
            test_log = ""
            for valid_dataloader in valid_dataloaders:
                avg_psnr = 0.0
                avg_ssim = 0.0
                name = valid_dataloader['name']
                loader = valid_dataloader['dataloader']
                for lr, hr in tqdm(loader, ncols=80):
                    lr, hr = lr.to(device), hr.to(device)
                    sr = model(lr)
                    # crop
                    hr = hr[:, :, args.scale:-args.scale, args.scale:-args.scale]
                    sr = sr[:, :, args.scale:-args.scale, args.scale:-args.scale]
                    # quantize
                    hr = hr.clamp(0, 255)
                    sr = sr.clamp(0, 255)
                    # calculate psnr
                    psnr = utils.calc_psnr(sr, hr)       
                    ssim = utils.calc_ssim(sr, hr)         
                    avg_psnr += psnr
                    avg_ssim += ssim
                avg_psnr = round(avg_psnr/len(loader), 2)
                avg_ssim = round(avg_ssim/len(loader), 4)
                stat_dict[name]['psnrs'].append(avg_psnr)
                stat_dict[name]['ssims'].append(avg_ssim)
                if stat_dict[name]['best_psnr']['value'] < avg_psnr:
                    stat_dict[name]['best_psnr']['value'] = avg_psnr
                    stat_dict[name]['best_psnr']['epoch'] = epoch
                if stat_dict[name]['best_ssim']['value'] < avg_ssim:
                    stat_dict[name]['best_ssim']['value'] = avg_ssim
                    stat_dict[name]['best_ssim']['epoch'] = epoch
                test_log += "[{}-X{}], PSNR/SSIM: {:.2f}/{:.4f} (Best: {:.2f}/{:.4f}, Epoch: {}/{})\n".format(
                    name, args.scale, float(avg_psnr), float(avg_ssim), 
                    stat_dict[name]['best_psnr']['value'], stat_dict[name]['best_ssim']['value'], 
                    stat_dict[name]['best_psnr']['epoch'], stat_dict[name]['best_ssim']['epoch'])
            # print log & flush out
            print(test_log)
            sys.stdout.flush()
            # save model
            saved_model_path = os.path.join(experiment_model_path, 'model_x{}_{}.pt'.format(args.scale, epoch))
            torch.save(model.state_dict(), saved_model_path)
            torch.set_grad_enabled(True)
            # save stat dict
            ## save training paramters
            stat_dict_name = os.path.join(experiment_path, 'stat_dict.yml')
            with open(stat_dict_name, 'w') as stat_dict_file:
                yaml.dump(stat_dict, stat_dict_file, default_flow_style=False)