import argparse
import random
import time
import glob
from tqdm import tqdm   
import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from arch.modules import UNet_conditional, EMA
from arch.ddpm_conditional import Diffusion
from arch.mnist import ClassifierMNIST, QuerierMNIST
import ops
from pathlib import Path
import utils
import wandb


os.environ["CUDA_VISIBLE_DEVICES"]="4"
def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--batch_size', type=int, default=64) # 128
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--tau_start', type=float, default=1.0)
    parser.add_argument('--tau_end', type=float, default=0.2)
    parser.add_argument('--lambd', type=float, default=0.5)
    parser.add_argument('--sampling', type=str, default='random')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='mnist')
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--tail', type=str, default='', help='tail message')
    parser.add_argument('--ckpt_path', type=str, default=None, help='load checkpoint')
    parser.add_argument('--save_dir', type=str, default='./saved/', help='save directory')
    parser.add_argument('--run_name', type=str, default='debug_uncond', help='save directory')
    parser.add_argument('--data_dir', type=str, default='./data/', help='save directory')
    parser.add_argument('--num_viz', type=int, default=3)
    parser.add_argument('--cfg_scale', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_sampling_steps', type=int, default=500)

    args = parser.parse_args()
    if args.num_viz > args.batch_size:
        args.num_viz = args.batch_size
    return args


def main(args):
    ## Setup
    # wandb
    run = wandb.init(project="Variational-IP", name=args.name, mode=args.mode)
    # model_dir = os.path.join(args.save_dir, f'{run.id}')
    model_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.run_name), exist_ok=True)
    utils.save_params(model_dir, vars(args))
    wandb.config.update(args)

    # cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('DEVICE:', device)
    print( 'N Dev, Curr Dev', torch.cuda.device_count(), torch.cuda.current_device())
    # random
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    assert args.lambd <= 1 and args.lambd >= 0

    ## Constants
    QUERY_ALL = 676 # 26*26
    PATCH_SIZE = 3
    THRESHOLD = 0.85
    H = W = 28

    ## Data
    transform = transforms.Compose([transforms.ToTensor(),  
                                    transforms.Lambda(lambda x: torch.where(x < 0.5, -1., 1.))])
    trainset = datasets.MNIST(args.data_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(args.data_dir, train=False, transform=transform, download=True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)

    ## Model
    unet = UNet_conditional(c_in=1, c_out=1, label_dim=11, size=28)
    unet = nn.DataParallel(unet).to(device)
    ema = EMA(0.995)
    ema_unet = copy.deepcopy(unet).eval().requires_grad_(False)
    diffusion = Diffusion(noise_steps=args.num_sampling_steps, img_size=H, device=device)

    ## Optimization
    criterion = nn.MSELoss() # Mean reduction
    optimizer = optim.AdamW(list(unet.parameters()),
                           amsgrad=True, lr=args.lr) # From ddpm, optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    tau_vals = np.linspace(args.tau_start, args.tau_end, args.epochs)

    ## Load checkpoint
    if args.ckpt_path is not None:
        ckpt_dict = torch.load(args.ckpt_path, map_location='cpu')
        unet.load_state_dict(ckpt_dict['unet'])
        ema_unet.load_state_dict(ckpt_dict['ema_unet'])
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        scheduler.load_state_dict(ckpt_dict['scheduler'])
        print('Checkpoint Loaded!')

    ## Train
    for epoch in range(args.epochs):
        # evaluation
        evaluation = True # False
        if (epoch % 1 == 0 or epoch == args.epochs - 1) and evaluation:

            unet.eval()
            ema_unet.eval()
            sample_model = unet
            for test_images, _ in tqdm(testloader):
                test_images = test_images.to(device)[:args.num_viz]
                # test_labels = test_labels.to(device)
                N, H, C, W = test_images.shape


                # save gt
                if epoch == 0:
                    gt_plot = (test_images.clamp(-1, 1) + 1) / 2
                    gt_plot = torch.repeat_interleave((gt_plot * 255).type(torch.uint8), args.num_viz, 0)
                    utils.save_images(gt_plot, os.path.join(args.save_dir, args.run_name, f"0-gt.png"), nrow=args.num_viz)

                # save unconditional sampling
                sampled_images = diffusion.sample(sample_model, n=args.num_viz, labels=[None], cfg_scale=args.cfg_scale)
                utils.save_images(sampled_images, os.path.join(args.save_dir, args.run_name, f"{epoch}-nq0.png"), nrow=args.num_viz)
                break

            # logging
            # wandb.log({
            #     'test_epoch': epoch,
            #     'qry_need_avg': qry_need_avg,
            #     'qry_need_std': qry_need_std
            # })

        # Training
        unet.train()
        ema_unet.train()
        for train_i, (train_images, train_labels) in enumerate(tqdm(trainloader)):
            train_images = train_images.to(device)
            # train_labels = train_labels.to(device)
            optimizer.zero_grad()

            t = diffusion.sample_timesteps(train_images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(train_images, t)
            predicted_noise = unet(x_t, t, None)
            loss = criterion(noise, predicted_noise)

            # backprop
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_unet, unet)

            # logging
            wandb.log({
                'epoch': epoch,
                'loss': loss.item(),
                'lr': utils.get_lr(optimizer),
                'gradnorm_dif': utils.get_grad_norm(unet),
            })
        scheduler.step()

        # saving
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                'unet': unet.state_dict(),
                'ema_unet': ema_unet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
                os.path.join(model_dir, 'ckpt', f'epoch{epoch}.ckpt'))



# args.epochs = 300
# args.batch_size = 14
# args.image_size = 64
# args.num_classes = 10
# args.dataset_path = r"C:\Users\dome\datasets\cifar10\cifar10-64\train"
# args.device = "cuda"
# args.lr = 3e-4
# train(args)

if __name__ == '__main__':
    args = parseargs()    
    main(args)


