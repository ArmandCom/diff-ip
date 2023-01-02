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

from arch.modules import UNet_conditional, UNet_conditional_celeba, UNet_conditional_cifar10, EMA
from arch.ddpm_conditional import Diffusion
from arch.mnist import ClassifierMNIST, QuerierMNIST
import ops
from pathlib import Path
import utils
import wandb
from PIL import Image


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, split, transform=None):
        max_num = 50000
        self.img_dir = img_dir
        self.transform = transform
        self.image_paths = os.listdir(self.img_dir)[:max_num]
        random.shuffle(self.image_paths)
        perc_train = int(0.7 * len(self.image_paths))
        if split == 'train':
            self.train = True
            self.image_paths = self.image_paths[:perc_train]
        elif split == 'test':
            self.train = False
            self.image_paths = self.image_paths[perc_train:]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        good = False
        while not good:
            img_path = os.path.join(self.img_dir, self.image_paths[idx])
            try:
                image = (np.array(Image.open(img_path)) / 255)
            except:
                idx = (idx + 1) % len(self.image_paths)
                continue
            good = True

        if self.transform:
            image = self.transform(image).float()
        return image

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--data', type=str, default='celeba')
    parser.add_argument('--batch_size', type=int, default=320) # 128
    parser.add_argument('--max_queries', type=int, default=50)
    parser.add_argument('--max_queries_test', type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--tau_start', type=float, default=1.0)
    parser.add_argument('--tau_end', type=float, default=0.2)
    parser.add_argument('--lambd', type=float, default=0.5)
    parser.add_argument('--sampling', type=str, default='random')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='celeba')
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--tail', type=str, default='', help='tail message')
    parser.add_argument('--ckpt_path', type=str, default=None, help='load checkpoint')
    parser.add_argument('--save_dir', type=str, default='./saved/', help='save directory')
    parser.add_argument('--run_name', type=str, default='debug_matrix_celeba_null', help='save directory')
    parser.add_argument('--data_dir', type=str, default='./data/', help='save directory')
    parser.add_argument('--num_viz', type=int, default=4)
    parser.add_argument('--cfg_scale', type=float, default=3)
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
    PATCH_SIZE = 7
    H = W = 64
    QH = H - PATCH_SIZE + 1
    QW = W - PATCH_SIZE + 1
    QUERY_ALL = QH*QW
    C = 3
    #TODO
    null_val = -10


    ## Data
    transform = transforms.Compose([transforms.ToTensor(),  
                                    transforms.Lambda(lambda x: x * 2 - 1),
                                    transforms.Resize(size=(H, W))
                                    ])
    trainset = CelebADataset(os.path.join(args.data_dir, args.data), split='train', transform=transform)
    testset = CelebADataset(os.path.join(args.data_dir, args.data), split='test', transform=transform)
    # trainset = datasets.CelebA(args.data_dir, split='train', transform=transform, download=True)
    # testset = datasets.CelebA(args.data_dir, split='test', transform=transform, download=True)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    ## Model
    unet = UNet_conditional_cifar10(c_in=2 * C, c_out=C, size=H)
    unet = nn.DataParallel(unet).to(device)
    ema = EMA(0.995)
    ema_unet = copy.deepcopy(unet).eval().requires_grad_(False)
    diffusion = Diffusion(noise_steps=args.num_sampling_steps, img_size=H, in_channels=C, device=device)

    # TODO: Querier should also be conditioned to X_0, aside from the set of queries.
    # TODO: Querier should be deeper for images like this. Use unet too?
    querier = QuerierMNIST(num_classes=QUERY_ALL, tau=args.tau_start,  q_size=(QH, QW), in_channels=2*C)
    querier = nn.DataParallel(querier).to(device)

    ## Optimization
    criterion = nn.MSELoss() # Mean reduction
    optimizer = optim.Adam(list(unet.parameters()), #list(querier.parameters())
                amsgrad=True, lr=args.lr) # From ddpm,
    # optimizer = optim.AdamW(list(querier.parameters()) + list(unet.parameters()), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) # TODO: Add warmup
    # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=- 1, verbose=False)
    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=1347, epochs=args.epochs)
    tau_vals = np.linspace(args.tau_start, args.tau_end, args.epochs)

    ## Load checkpoint
    if args.ckpt_path is not None:
        ckpt_dict = torch.load(args.ckpt_path, map_location='cpu')
        unet.load_state_dict(ckpt_dict['unet'])
        ema_unet.load_state_dict(ckpt_dict['ema_unet'])
        querier.load_state_dict(ckpt_dict['querier'])
        # optimizer.load_state_dict(ckpt_dict['optimizer'])
        # scheduler.load_state_dict(ckpt_dict['scheduler'])
        print('Checkpoint Loaded!')

    ## Train
    for epoch in range(args.epochs):
        # evaluation
        # Training
        unet.train()
        ema_unet.train()
        querier.train()
        tau = tau_vals[epoch]
        for train_i, train_images in enumerate(tqdm(trainloader)):
            train_images = train_images.to(device)
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
                masked_image, S_v, S_ij, split = ops.get_patch_mask(mask, train_images, patch_size=PATCH_SIZE, null_val=null_val)

            querier_inputs = torch.cat([masked_image, train_images], dim=1)
            # querier_inputs = masked_image
            query_vec = querier(querier_inputs, mask)
            masked_image = ops.update_masked_image(masked_image, train_images, query_vec, patch_size=PATCH_SIZE)

            t = diffusion.sample_timesteps(train_images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(train_images, t)
            if np.random.random() < 0.1:
                x_t = torch.cat([torch.zeros_like(masked_image) + null_val, x_t], dim=1)
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

            if (train_i % 2000 == 0 or epoch == args.epochs - 1) and epoch % 3 == 0:

                unet.eval()
                ema_unet.eval()
                querier.eval()
                sample_model = ema_unet

                for test_images in tqdm(testloader):


                    test_images = test_images.to(device)[:args.num_viz]
                    N, H, C, W = test_images.shape

                    # Compute logits for all queries
                    test_inputs = torch.zeros_like(test_images).to(device) + null_val
                    mask = torch.zeros(N, QUERY_ALL).to(device)

                    # save gt
                    if epoch == 0:
                        gt_plot = (test_images.clamp(-1, 1) + 1) / 2
                        gt_plot = torch.repeat_interleave((gt_plot * 255).type(torch.uint8), args.num_viz, 0)
                        utils.save_images(gt_plot, os.path.join(args.save_dir, args.run_name, f"0-gt.png"), nrow=args.num_viz)

                    for i in range(args.max_queries_test):
                        with torch.no_grad():
                            querier_inputs = torch.cat([test_inputs, test_images], dim=1)
                            query_vec = querier(querier_inputs, mask)

                            # Random queries for test
                            query_vec = torch.zeros_like(query_vec)
                            rand_int = torch.randint(query_vec.shape[1],(query_vec.shape[0],))
                            query_vec[torch.arange(query_vec.shape[0]), rand_int] = 1

                            suffix = 'rand_'
                            if i == 0 or i == args.max_queries_test - 1 or args.sample_all:
                                sampled_images = diffusion.sample(sample_model, n=args.num_viz, labels=[test_inputs], cfg_scale=args.cfg_scale)
                                utils.save_images(sampled_images, os.path.join(args.save_dir, args.run_name, f"{suffix}{epoch}-nq{i}.png"), nrow=args.num_viz)
                                # if i == args.max_queries_test - 1:
                                #     sampled_images = diffusion.sample(ema_unet, n=args.num_viz, labels=[test_inputs], cfg_scale=args.cfg_scale)
                                #     utils.save_images(sampled_images, os.path.join(args.save_dir, args.run_name, f"{epoch}-nq{i}_ema.png"), nrow=args.num_viz)
                                test_inputs_plot = test_inputs.clone()
                                test_inputs_plot[test_inputs_plot == null_val] = 0
                                gt_plot = (test_inputs_plot.clamp(-1, 1) + 1) / 2
                                gt_plot = torch.repeat_interleave((gt_plot * 255).type(torch.uint8), args.num_viz, 0)
                                utils.save_images(gt_plot, os.path.join(args.save_dir, args.run_name, f"{suffix}{epoch}-nq{i}-qs.png"), nrow=args.num_viz)

                                test_inputs_mask_plot = test_inputs.clone()
                                test_inputs_mask_plot[test_inputs_mask_plot != null_val] = 1
                                test_inputs_mask_plot[test_inputs_mask_plot == null_val] = 0
                                test_inputs_mask_plot = torch.repeat_interleave(test_inputs_mask_plot, args.num_viz, 0)
                                utils.save_images((sampled_images*test_inputs_mask_plot).type(torch.uint8), os.path.join(args.save_dir, args.run_name, f"{suffix}{epoch}-nq{i}_masked.png"), nrow=args.num_viz)

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
        if epoch % 5 == 0 or epoch == args.epochs - 1:
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
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    main(args)


