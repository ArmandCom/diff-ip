import os
import json
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt, matplotlib.image as mpimg, io

def ltplot(img):
    'Plot an lt image array such as x.chans, x.rgb, x.plt'
    fp = io.BytesIO(img._repr_png_())
    with fp: img = mpimg.imread(fp, format='png')
    plt.imshow(img)
    plt.show()

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

import numpy as np
def save_images(images, path, range =None, **kwargs):
    if range is not None:
        images = 255 * (images - range[0])/(range[1]-range[0])
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr.astype(np.uint8))
    im.save(path)

def log_images(images, logger, name='image', range =None, **kwargs):
    if range is not None:
        images = 255 * (images - range[0])/(range[1]-range[0])
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr.astype(np.uint8))
    images = logger.Image(im)
    logger.log({name: images})

def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        curr_lr = param_group['lr']
        return curr_lr


def get_grad_norm(model, names=None):
    total_norm = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                #print('NOgrad', name)
                continue

            else:
                if names is not None:
                    if name not in names:
                        continue
                # print('has grad', name)
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def save_params(model_dir, params, name='params'):
    """Save params to a .json file. Params is a dictionary of parameters."""
    path = os.path.join(model_dir, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)
