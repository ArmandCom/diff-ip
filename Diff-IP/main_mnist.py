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
    parser.add_argument('--batch_size', type=int, default=6) # 128
    parser.add_argument('--max_queries', type=int, default=18)
    parser.add_argument('--max_queries_test', type=int, default=10)
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
    parser.add_argument('--run_name', type=str, default='debug', help='save directory')
    parser.add_argument('--data_dir', type=str, default='./data/', help='save directory')
    parser.add_argument('--num_viz', type=int, default=3)
    parser.add_argument('--cfg_scale', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_sampling_steps', type=int, default=200)

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

    # TODO: Querier should also be conditioned to X_0, aside from the set of queries.
    querier = QuerierMNIST(num_classes=QUERY_ALL, tau=args.tau_start)
    querier = nn.DataParallel(querier).to(device)

    ## Optimization
    criterion = nn.MSELoss() # Mean reduction
    optimizer = optim.Adam(list(querier.parameters()) + list(unet.parameters()),
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
        if (epoch % 1 == 0 or epoch == args.epochs - 1) and evaluation:

            unet.eval()
            ema_unet.eval()
            querier.eval()
            sample_model = unet
            epoch_test_qry_need = []
            for test_images, _ in tqdm(testloader):
                test_images = test_images.to(device)[:args.num_viz]
                # test_labels = test_labels.to(device)
                N, H, C, W = test_images.shape

                # Compute logits for all queries
                test_inputs = torch.zeros_like(test_images).to(device)
                mask = torch.zeros(N, QUERY_ALL).to(device)
                samples, queries = [], []

                # save gt
                if epoch == 0:
                    gt_plot = (test_images.clamp(-1, 1) + 1) / 2
                    gt_plot = torch.repeat_interleave((gt_plot * 255).type(torch.uint8), args.num_viz, 0)
                    utils.save_images(gt_plot, os.path.join(args.save_dir, args.run_name, f"0-gt.png"), nrow=args.num_viz)

                # save unconditional sampling
                sampled_images = diffusion.sample(sample_model, n=args.num_viz, labels=[None], cfg_scale=args.cfg_scale)
                utils.save_images(sampled_images, os.path.join(args.save_dir, args.run_name, f"{epoch}-nq0.png"), nrow=args.num_viz)

                for i in range(args.max_queries_test):
                    with torch.no_grad():
                        query_vec = querier(test_inputs, mask)
                        q_v, q_ij = ops.get_single_query(test_images, query_vec, patch_size=PATCH_SIZE)
                        queries.append(ops.get_labels([q_v, q_ij], size=H))

                        if i == 0 or i == args.max_queries_test - 1:
                            sampled_images = diffusion.sample(sample_model, n=args.num_viz, labels=queries, cfg_scale=args.cfg_scale)
                            utils.save_images(sampled_images, os.path.join(args.save_dir, args.run_name, f"{epoch}-nq{i+1}.png"), nrow=args.num_viz)
                            if i == args.max_queries_test - 1:
                                sampled_images = diffusion.sample(ema_unet, n=args.num_viz, labels=queries, cfg_scale=args.cfg_scale)
                                utils.save_images(sampled_images, os.path.join(args.save_dir, args.run_name, f"{epoch}-nq{i+1}_ema.png"), nrow=args.num_viz)

                        mask[np.arange(N), query_vec.argmax(dim=1)] = 1.0
                        test_inputs = ops.update_masked_image(test_inputs, test_images, query_vec, patch_size=PATCH_SIZE)

                # logits = torch.stack(samples).permute(1, 0, 2)

                # # compute query needed
                # qry_need = ops.compute_queries_needed(logits, threshold=THRESHOLD)
                # epoch_test_qry_need.append(qry_need)
                break
            # mean and std of queries needed
            # epoch_test_qry_need = torch.hstack(epoch_test_qry_need).float()
            # qry_need_avg = epoch_test_qry_need.mean()
            # qry_need_std = epoch_test_qry_need.std()

            # logging
            # wandb.log({
            #     'test_epoch': epoch,
            #     'qry_need_avg': qry_need_avg,
            #     'qry_need_std': qry_need_std
            # })

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
            # Query and update\

            query_vec = querier(masked_image, mask)
            q_v, q_ij = ops.get_single_query(train_images, query_vec, patch_size=PATCH_SIZE)
            # TODO: Check how do we choose ij. If it's different from S.
            #  Why not simply add a Pos Embed to the image?
            t = diffusion.sample_timesteps(train_images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(train_images, t)
            if np.random.random() < 0.1:
                predicted_noise = unet(x_t, t, None)
                loss = criterion(noise, predicted_noise)
            else:
                S, q = ops.get_labels([S_v, S_ij], size=H), ops.get_labels([q_v, q_ij], size=H)
                pred_noise_q = unet(x_t, t, q)

                x_t_S, t_S, noise_S = ops.expand_to_S([x_t, t, noise], split)
                pred_noise_S = unet(x_t_S, t_S, S) #TODO: Check that batch_size>0
                predicted_noise = ops.average_gradients(
                    pred_noise_S.detach(),
                    pred_noise_q,
                    split
                )
                loss_guidance_dif = criterion(noise_S, pred_noise_S)

                loss_ip = criterion(noise, predicted_noise)
                loss = args.lambd * loss_ip + (1 - args.lambd) * loss_guidance_dif

                wandb.log({
                    'loss_IP': loss_ip.item(),
                    'loss_diffusion': loss_guidance_dif.item()
                })
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
                'gradnorm_qry': utils.get_grad_norm(querier)
            })
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


