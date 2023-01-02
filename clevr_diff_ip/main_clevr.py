import argparse
import random
import time
import glob
from tqdm import tqdm   
import os
import copy

import numpy as np
import torch
# import torch.nn as nn
# import torchvision.datasets as datasets
import torchvision.transforms as transforms
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

# from arch.modules import UNet_conditional, UNet_conditional_celeba, UNet_conditional_cifar10, EMA
# from arch.ddpm_conditional import Diffusion
# from arch.models import Querier
import ops
import utils
import wandb
# from PIL import Image
# from imagen_pytorch.imagen_pytorch import Unet, Imagen, SRUnet256
from imagen_pytorch import Unet, Imagen, SRUnet256, ImagenTrainer, resize_image_to
# os.environ['WANDB_DISABLED'] = 'true'
from dataloader import Clevr_with_masks
import torchvision as tv
import lovely_tensors as lt
lt.monkey_patch()

# conda activate imagen; cd projects/clevr_diff_ip/; accelerate launch main_clevr.py --batch_size 16 --sampling biased --test --name querier128SAUnet_S3000biased_and_unbiased_withmask_hard

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--data', type=str, default='clevr')
    parser.add_argument('--batch_size', type=int, default=320) # 128
    parser.add_argument('--max_queries', type=int, default=50)
    parser.add_argument('--max_queries_test', type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--tau_start', type=float, default=1.0)
    parser.add_argument('--tau_end', type=float, default=1.0)
    parser.add_argument('--lambd', type=float, default=0.5)
    parser.add_argument('--sampling', type=str, default='random')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='celeba')
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--tail', type=str, default='', help='tail message')
    parser.add_argument('--ckpt_path', type=str, default=None, help='load checkpoint')
    parser.add_argument('--save_dir', type=str, default='./saved/', help='save directory')
    parser.add_argument('--run_name', type=str, default='debug_clevr_biased_random', help='save directory')
    parser.add_argument('--data_dir', type=str, default='/cis/home/acomas/data/clevr-dataset-gen/output/images/', help='save directory')
    parser.add_argument('--num_viz', type=int, default=4)
    parser.add_argument('--cfg_scale', type=float, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_sampling_steps', type=int, default=300)
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--load_epoch', type=int, default=0)
    parser.add_argument('--test', type=bool, default=False)

    args = parser.parse_args()
    if args.num_viz > args.batch_size:
        args.num_viz = args.batch_size
    return args


def main(args):
    ## Setup
    # wandb
    run = wandb.init(project="Diff-IP-Clevr", name=args.name, mode=args.mode)
    # model_dir = os.path.join(args.save_dir, f'{run.id}')
    # os.makedirs(model_dir, exist_ok=True)

    model_dir_ckpt = os.path.join(args.save_dir, args.run_name)
    os.makedirs(os.path.join(model_dir_ckpt, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.run_name), exist_ok=True)

    utils.save_params(model_dir_ckpt, vars(args)) #TODO: or model_dir_ckpt?
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
    H = W = 128
    patch_size = 5
    qh = qw = 128 - patch_size + 1
    C = 3

    ## Data
    transform = transforms.Compose([transforms.ToTensor(),  
                                    transforms.Lambda(lambda x: x * 2 - 1),
                                    transforms.Resize(size=(H, W))
                                    ])
    trainset = Clevr_with_masks(args.data_dir, split='train', transform=transform)
    testset = Clevr_with_masks(args.data_dir, split='test', transform=transform)

    # trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=4, num_workers=args.num_workers, pin_memory=True)

    iters_per_epoch = len(trainset) // args.batch_size

    sampling_mode = 'biased'


    unet1 = Unet( # TODO: Increase dim to 128 at the expense of other hyperparams.
        dim = 32,
        patch_size = patch_size,
        image_size = (W, H),
        sampling_mode = sampling_mode,
        max_rand_queries = 100,
        image_embed_dim = 1024,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, False, False, True),
        layer_cross_attns = (False, True, True, True),
        cond_images_channels = 1 + C # S + GT Image
    )

    imagen = Imagen(
        condition_on_text = False,   # this must be set to False for unconditional Imagen
        unets = (unet1),
        image_sizes = (H),
        timesteps = 1000
    )

    trainer = ImagenTrainer(imagen,
                            warmup_steps = 1 * iters_per_epoch,
                            cosine_decay_max_steps = (args.epochs - 1) * iters_per_epoch,
                            split_valid_from_train = True,
                            dl_tuple_output_keywords_names = ('images', 'cond_images', 'text_embeds', 'text_masks')).cuda()

    trainer.add_train_dataset(trainset, batch_size = args.batch_size)


    tau_vals = np.linspace(args.tau_start, args.tau_end, args.epochs)


    def sample_with_querier(trainer, sample_ids, num_queries=5, num_samples=1, query_size=(128, 128), epoch=0, max_query=100, log=False, full_queries=False):

        q_list = []
        qx_list = []
        attn_list = []
        im_list = []

        _, (input_x, input_mask) = next(enumerate(testloader))
        input_mask = input_mask[:, :1]
        N, C, W, H = input_mask.shape
        NULL_VAL = 0 #-10


        if not log:
            utils.save_images(input_x, os.path.join(model_dir_ckpt, 'GT_sample.png'), range=(-1,1))
            utils.save_images(input_mask[:, :1] * torch.ones_like(input_x),
                              os.path.join(model_dir_ckpt, 'GT_mask.png'), range=(0,1))
        else:
            pass


        if full_queries:
            sample_ids = [0]
            for j in range(num_samples):
                images = trainer.sample(cond_images=torch.cat([input_mask, input_x], dim=1), batch_size=input_mask.shape[0])
                im_list.append(images.cpu())
        else:
            querier = trainer.imagen.unets[0].querier # Put to eval or torch nograd
            if sample_ids == 'all':
                sample_ids = list(np.arange(num_queries))
            elif -1 in sample_ids:
                sample_ids.remove(-1)
                sample_ids.append(num_queries-1)

            hq, wq = query_size
            input_mask = resize_image_to(input_mask, hq + patch_size - 1)
            masked_x = torch.zeros_like(input_mask) + NULL_VAL
            mask = torch.zeros(N, hq*wq).to(device)

            for i in range(num_queries):
                # Save queries
                # if i == 0:
                #     plot_mask = ops.random_sampling(max_query, query_size[0]*query_size[1], N).reshape(N, 1, *query_size).clone()
                # else:
                #     plot_mask = mask.reshape(N, 1, *query_size).clone().cpu()

                plot_mask = masked_x.clone()


                plot_mask[plot_mask != NULL_VAL] = 1
                plot_mask[plot_mask == NULL_VAL] = 0
                # plot_mask_raw = plot_mask.clone()

                plot_queries = resize_image_to(plot_mask * input_mask, input_x.shape[-2:])
                plot_mask = resize_image_to(plot_mask, input_x.shape[-2:])
                gaussian_blur = tv.transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

                plot_queries = gaussian_blur(plot_queries)
                plot_mask = gaussian_blur(plot_mask) # Gaussian blob

                plot_queries[plot_queries > 0.02] = 1
                plot_queries[plot_queries <= 0.02] = 0

                plot_mask[plot_mask > 0.02] = 1
                plot_mask[plot_mask <= 0.02] = 0

                drawn_queries_x = torch.ones_like(plot_mask) * plot_queries + input_x * (1-plot_queries)
                drawn_queries = torch.ones_like(plot_mask) * plot_mask + input_x * (1-plot_mask)

                if not log:
                    utils.save_images(drawn_queries, os.path.join(model_dir_ckpt, 'E{}_S{}_queries.png'.format(epoch,i)), range=(-1,1))
                else:
                    q_list.append(drawn_queries)
                    qx_list.append(drawn_queries_x)

                gt_input = resize_image_to(input_x, hq + patch_size - 1)
                # gt_input = torch.cat([input_mask, resize_image_to(input_x, query_size)], dim=1)

                # Save images
                if i in sample_ids:

                    for j in range(num_samples):
                        images = trainer.sample(cond_images=torch.cat([masked_x, gt_input], dim=1), batch_size=input_mask.shape[0])
                        if not log:
                            utils.save_images(torch.ones_like(plot_mask) * plot_mask + images.cpu() * (1-plot_mask),
                                              os.path.join(model_dir_ckpt, 'E{}_S{}_{}_sample_gen_wS.png'.format(epoch,i,j)), range=(-1,1))
                        else:
                            im_list.append(torch.ones_like(plot_mask) * plot_mask + images.cpu() * (1-plot_mask))

                # Querier update
                querier_inputs = torch.cat([masked_x, gt_input], dim=1).to(device)
                with torch.no_grad():
                    query_vec, attn = querier(querier_inputs, mask, return_attn=True)
                mask[np.arange(N), query_vec.argmax(dim=1)] = 1.0
                masked_x = ops.update_masked_image(masked_x, input_mask, query_vec.cpu(), patch_size=patch_size)
                if log:
                    attn_plot = attn.reshape(N, 1, *query_size).clone().cpu()
                    attn_plot = resize_image_to(attn_plot, input_x.shape[-2:])
                    attn_list.append(attn_plot)

        if log:
            if len(im_list)>0:
                utils.log_images(torch.stack(im_list, dim=1).flatten(0,1), wandb,
                                 name='S_sample_gen_wS', range=(-1,1), nrow=len(sample_ids) * num_samples)
            if len(q_list)>0:
                utils.log_images(torch.stack(q_list, dim=1).flatten(0,1), wandb, name='S_q', range=(-1,1), nrow=num_queries)
                utils.log_images(torch.stack(qx_list, dim=1).flatten(0,1), wandb, name='S_q(X)', range=(-1,1), nrow=num_queries)

            if len(attn_list)>0:
                utils.log_images(torch.stack(attn_list, dim=1).flatten(0,1), wandb, name='Q_attention', range=(0,1), nrow=num_queries)

    # Load epoch
    load_epoch = 0
    if args.load_epoch > 0:
        load_epoch = args.load_epoch
        if hasattr(trainer.imagen.unets[0], 'querier'):
            trainer.imagen.unets[0].querier.tau = tau_vals[load_epoch]
        trainer.load(os.path.join(model_dir_ckpt, 'ckpt', f'epoch{load_epoch}.pt'))


    # Test
    if args.test:
        sample_with_querier(trainer, sample_ids=[-1], num_samples=5, num_queries=20, query_size=(qh, qw), epoch=load_epoch, log=True, full_queries=False)
        exit()

    # Train
    trainer.imagen.unets[0].sampling = args.sampling
    for epoch in range(load_epoch, args.epochs, 1):
        if hasattr(trainer.imagen.unets[0], 'querier'):
            trainer.imagen.unets[0].querier.tau = tau_vals[epoch]
            if epoch % 1 == 0:
                sample_ids = []
                sample_with_querier(trainer, sample_ids=sample_ids, num_queries=20, query_size=(qh, qw), epoch=epoch, log=True)
                wandb.log({'lr':trainer.get_lr(1)})
        for _ in tqdm(range(iters_per_epoch)):
            dict_out = trainer.train_step(unet_number=1, max_batch_size=100)
            wandb.log(dict_out)
        print('Epoch {} done.'.format(epoch+1))
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print('Save ckpt')
            trainer.save(os.path.join(model_dir_ckpt, 'ckpt', f'epoch{epoch}.pt'))
            # exit()

    print('Training Done')
    exit()

if __name__ == '__main__':
    args = parseargs()
    # os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    main(args)


