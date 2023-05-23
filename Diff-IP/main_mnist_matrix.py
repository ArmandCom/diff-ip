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


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--batch_size', type=int, default=200) # 128
    parser.add_argument('--max_queries', type=int, default=50)
    parser.add_argument('--max_queries_test', type=int, default=15)
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
    parser.add_argument('--run_name', type=str, default='diff_mnist_mix_2', help='save directory') #'diff_mnist_mix'
    parser.add_argument('--data_dir', type=str, default='./data/', help='save directory')
    parser.add_argument('--num_viz', type=int, default=3)
    parser.add_argument('--cfg_scale', type=float, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_sampling_steps', type=int, default=300)
    parser.add_argument('--sample_all', type=bool, default=False)
    parser.add_argument('--gpu_id', type=str, default="1")


    args = parser.parse_args()
    if args.num_viz > args.batch_size:
        args.num_viz = args.batch_size
    return args


def main(args):
    ## Setup
    # wandb
    run = wandb.init(project="Variational-IP", name=args.name, mode=args.mode)
    model_dir = os.path.join(args.save_dir, f'{run.id}')
    os.makedirs(model_dir, exist_ok=True)

    model_dir_ckpt = os.path.join(args.save_dir, args.run_name)
    os.makedirs(os.path.join(model_dir_ckpt, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.run_name), exist_ok=True)

    utils.save_params(model_dir, vars(args)) #TODO: or model_dir_ckpt?
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
    unet = UNet_conditional(c_in=2, c_out=1, size=28)
    unet = nn.DataParallel(unet).to(device)
    ema = EMA(0.995)
    ema_unet = copy.deepcopy(unet).eval().requires_grad_(False)
    diffusion = Diffusion(noise_steps=args.num_sampling_steps, img_size=H, device=device)

    # TODO: Querier should also be conditioned to X_0, aside from the set of queries.
    querier = QuerierMNIST(num_classes=QUERY_ALL, tau=args.tau_start)
    querier = nn.DataParallel(querier).to(device)

    ## Optimization
    criterion = nn.MSELoss() # Mean reduction
    optimizer = optim.AdamW(list(querier.parameters()) + list(unet.parameters()),
                           amsgrad=True, lr=args.lr) # From ddpm, optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    tau_vals = np.linspace(args.tau_start, args.tau_end, args.epochs)

    ## Load checkpoint
    if args.ckpt_path is not None:
        ckpt_dict = torch.load(args.ckpt_path, map_location='cpu')
        unet.load_state_dict(ckpt_dict['unet'])
        ema_unet.load_state_dict(ckpt_dict['ema_unet'])
        querier.load_state_dict(ckpt_dict['querier'])
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        scheduler.load_state_dict(ckpt_dict['scheduler'])
        print('Checkpoint Loaded!')

    ## Train
    for epoch in range(args.epochs):
        # evaluation
        evaluation = True # False
        # Training
        unet.train()
        ema_unet.train()
        querier.train()
        tau = tau_vals[epoch]
        for train_i, (train_images, train_labels) in enumerate(tqdm(trainloader)):
            train_images = train_images.to(device)
            # train_labels = train_labels.to(device)
            querier.module.update_tau(tau)
            optimizer.zero_grad()

            # initial random sampling
            if args.sampling == 'biased':
                num_queries = torch.randint(low=0, high=QUERY_ALL, size=(train_images.size(0),))
                mask, masked_image = ops.adaptive_sampling(train_images, num_queries, querier, PATCH_SIZE, QUERY_ALL)
                print('ERROR: We need to implement the individual query sampling.')
                raise NotImplementedError
            elif args.sampling == 'random':
                mask = ops.random_sampling(args.max_queries, QUERY_ALL, train_images.size(0)).to(device)
                masked_image, S_v, S_ij, split = ops.get_patch_mask(mask, train_images, patch_size=PATCH_SIZE)

            query_vec = querier(masked_image, mask)
            masked_image = ops.update_masked_image(masked_image, train_images, query_vec, patch_size=PATCH_SIZE)

            t = diffusion.sample_timesteps(train_images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(train_images, t)
            if np.random.random() < 0.1:
                x_t = torch.cat([torch.zeros_like(masked_image), x_t], dim=1)
                predicted_noise = unet(x_t, t, None)
                loss = criterion(noise, predicted_noise)
            else:
                x_t = torch.cat([masked_image, x_t], dim=1)
                predicted_noise = unet(x_t, t, None)
                loss_ip = criterion(noise, predicted_noise)
                loss = loss_ip

                wandb.log({
                    'loss_IP': loss_ip.item(),
                })
            # backprop
            if not args.sample_all:
                loss.backward()
            optimizer.step()
            ema.step_ema(ema_unet, unet)

            # logging
            wandb.log({
                'epoch': epoch,
                'loss': loss.item(),
                'lr': utils.get_lr(optimizer),
                'gradnorm_dif': utils.get_grad_norm(unet),
                'gradnorm_qry': utils.get_grad_norm(querier)
            })

            if ( (train_i % 3000 == 0 or epoch == args.epochs - 1)or args.sample_all) and evaluation:

                unet.eval()
                ema_unet.eval()
                querier.eval()
                sample_model = unet #ema_unet
                sample_labels = [0,7,8,2]
                args.num_viz = len(sample_labels)
                test_images_list = []
                for test_images, test_labels in tqdm(testloader):

                    for bid, lab in enumerate(test_labels):
                        if len(sample_labels) == 0:
                            continue
                        elif lab in sample_labels:
                            test_images_list.append(test_images[bid])
                            sample_labels.remove(lab)
                    if len(sample_labels) > 0:
                        continue
                    test_images = torch.stack(test_images_list).to(device)

                    # test_images = test_images.to(device)[8:args.num_viz+8]
                    N, H, C, W = test_images.shape

                    # Compute logits for all queries
                    test_inputs = torch.zeros_like(test_images).to(device)
                    mask = torch.zeros(N, QUERY_ALL).to(device)

                    # save gt
                    if epoch == 0:
                        gt_plot = (test_images.clamp(-1, 1) + 1) / 2
                        gt_plot = torch.repeat_interleave((gt_plot * 255).type(torch.uint8), args.num_viz, 0)
                        utils.save_images(gt_plot, os.path.join(args.save_dir, args.run_name, f"0-gt.png"), nrow=args.num_viz)

                    for i in range(args.max_queries_test):
                        with torch.no_grad():
                            query_vec = querier(test_inputs, mask)
                            # print(query_vec.min(), query_vec.max(), query_vec.mean())
                            # print(test_inputs.min(), test_inputs.max(), test_inputs.mean())
                            # print('\n')

                            if (i == 0 or i == args.max_queries_test - 1 or args.sample_all):
                                sampled_images = diffusion.sample(sample_model, n=args.num_viz, labels=[test_inputs], cfg_scale=args.cfg_scale)
                                utils.save_images(sampled_images, os.path.join(args.save_dir, args.run_name, f"{epoch}-nq{i}.png"), nrow=args.num_viz)
                                if i == args.max_queries_test - 1:
                                    sampled_images = diffusion.sample(ema_unet, n=args.num_viz, labels=[test_inputs], cfg_scale=args.cfg_scale)
                                    utils.save_images(sampled_images, os.path.join(args.save_dir, args.run_name, f"{epoch}-nq{i}_ema.png"), nrow=args.num_viz)
                                gt_plot = (test_inputs.clamp(-1, 1) + 1) / 2
                                gt_plot = torch.repeat_interleave((gt_plot * 255).type(torch.uint8), args.num_viz, 0)
                                utils.save_images(gt_plot, os.path.join(args.save_dir, args.run_name, f"{epoch}-nq{i}-qs.png"), nrow=args.num_viz)

                        mask[np.arange(N), query_vec.argmax(dim=1)] = 1.0
                        test_inputs = ops.update_masked_image(test_inputs, test_images, query_vec, patch_size=PATCH_SIZE)

                    break
                if args.sample_all:
                    exit()
                # logging
                wandb.log({
                    'test_epoch': epoch
                })

                # Training
                unet.train()
                ema_unet.train()
                querier.train()
                tau = tau_vals[epoch]

        scheduler.step()

        # saving
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                'unet': unet.state_dict(),
                'ema_unet': ema_unet.state_dict(),
                'querier': querier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
                os.path.join(model_dir_ckpt, 'ckpt', f'epoch{epoch}.ckpt'))


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
    # os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    main(args)


