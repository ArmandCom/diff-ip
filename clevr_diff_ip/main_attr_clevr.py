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

from arch.vector_querier_models import answer_queries, answer_single_query

import ops
import utils
import wandb
# from PIL import Image
# from imagen_pytorch.imagen_pytorch import Unet, Imagen, SRUnet256
from imagen_pytorch import Unet, Imagen, SRUnet256, ImagenTrainer, resize_image_to
# os.environ['WANDB_DISABLED'] = 'true'
from dataloader import Clevr_with_masks, Clevr_with_attr
import torchvision as tv
import lovely_tensors as lt
lt.monkey_patch()

# ps -up `nvidia-smi -q -x | grep pid | sed -e 's/<pid>//g' -e 's/<\/pid>//g' -e 's/^[[:space:]]*//'`
# conda activate imagen; cd projects/clevr_diff_ip/; accelerate launch main_clevr.py --batch_size 16 --sampling biased --test --name querier128SAUnet_S3000biased_and_unbiased_withmask_hard

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--data', type=str, default='clevr')
    parser.add_argument('--batch_size', type=int, default=20) # 128
    parser.add_argument('--max_queries', type=int, default=50)
    parser.add_argument('--max_queries_test', type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--tau_start', type=float, default=5.0)
    parser.add_argument('--tau_end', type=float, default=1.0)
    parser.add_argument('--lambd', type=float, default=0.5)
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
    parser.add_argument('--cond_scale', type=float, default=3)
    parser.add_argument('--attribute', type=str, default="color")
    parser.add_argument('--num_samples', type=int, default=1)

    parser.add_argument('--freeze_unet', type=bool, default=False)
    parser.add_argument('--train_querier', type=bool, default=True) # Doesn't work?
    parser.add_argument('--all_queries', type=bool, default=False)
    parser.add_argument('--sampling', type=str, default='random')
    parser.add_argument('--max_queries_biased', type=int, default=10)
    parser.add_argument('--embed_dim', type=int, default=8)
    parser.add_argument('--experiment_type', type=str, default='patches')
    parser.add_argument('--null_val', type=float, default=-10)
    args = parser.parse_args()

    args.train_querier = False
    args.all_queries = False
    if args.num_viz > args.batch_size:
        args.num_viz = args.batch_size
    return args


def main(args):
    ## Setup
    # wandb
    run = wandb.init(project="Diff-IP-Clevr-attr", name=args.name, mode=args.mode)
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
    print('\n Look at the TODO list! \n')

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
    cond_scale = args.cond_scale
    attribute = args.attribute
    max_num_objects = 5
    if args.attribute == 'color':
        max_num_attributes = 10
    elif args.attribute == 'shape':
        max_num_attributes = 5
    else: raise NotImplementedError
    max_num_queries = max_num_attributes * max_num_objects
    embed_dim = args.embed_dim
    attr_size = embed_dim * max_num_queries # Depending on if we use answers or embeddings for conditioning.
    args.max_queries_biased = 10 # max_num_queries


    ## Data
    transform = transforms.Compose([transforms.ToTensor(),  
                                    transforms.Lambda(lambda x: x * 2 - 1),
                                    transforms.Resize(size=(H, W))
                                    ])

    trainset = Clevr_with_attr(args.data_dir, split='train', transform=transform, attribute=attribute, max_attributes=max_num_objects)
    testset = Clevr_with_attr(args.data_dir, split='test', transform=transform, attribute=attribute, max_attributes=max_num_objects)

    # trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=4, num_workers=args.num_workers, pin_memory=True)

    iters_per_epoch = len(trainset) // args.batch_size


    unet1 = Unet( # TODO: Increase dim to 128 at the expense of other hyperparams.
        dim = 64,
        image_size = (W, H),
        sampling_mode = args.sampling,
        max_num_queries = 20,
        max_rand_queries = 100,
        max_num_attributes = max_num_attributes,
        max_num_objects = 5,
        image_embed_dim = 1024,
        cond_on_text = True, # Needed (?)
        text_embed_dim = attr_size, # Not sure if it's necessary, but a bunch of things are done with this
        cond_dim = attr_size,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, False, False, True),
        layer_cross_attns = (True, True, True, True), # TODO: Checked all true
        max_text_len = max_num_attributes,
        cond_images_channels = 3, # S + GT Image,
        FLAGS = args,
    )
    # cond_images_channels = 1 + C # S + GT Image

    imagen = Imagen(
        unets = (unet1),
        image_sizes = (H),
        timesteps = 1000,
        text_embed_dim = attr_size,
        cond_drop_prob=0.1
    )

    trainer = ImagenTrainer(imagen,
                            warmup_steps = 1 * iters_per_epoch,
                            cosine_decay_max_steps = (args.epochs - 1) * iters_per_epoch,
                            split_valid_from_train = True,
                            dl_tuple_output_keywords_names = ('images', 'text_embeds', 'cond_images')).cuda()

    trainer.add_train_dataset(trainset, batch_size = args.batch_size)

    # Freeze network parameters to train querier
    if args.freeze_unet and args.train_querier:
        for param_name, param in trainer.imagen.named_parameters():
            if "querier" not in param_name:
                param.requires_grad = False
            print(param.requires_grad)
        print('Frozen unet')
    tau_vals = np.linspace(args.tau_start, args.tau_end, args.epochs)



    def sample_with_querier(trainer, sample_ids, num_queries=5, num_samples=1, query_size=(128, 128), epoch=0, max_query=100, log=False, full_queries=False):

        q_list = []
        qx_list = []
        attn_list = []
        im_list = []


        _, (input_x, attrs, cond_images) = next(enumerate(testloader))
        N, C, W, H = input_x.shape

        text_table = wandb.Table(columns=["Attribute IDs"])
        for i in range(N):
            text_table.add_data(' '.join(str(e) for e in list(attrs[i].numpy())))
        wandb.log({"Attributes": text_table})

        if not log:
            utils.save_images(input_x, os.path.join(model_dir_ckpt, 'GT_sample.png'), range=(-1, 1))
        else:
            im_list.append(input_x)
            utils.log_images(input_x, wandb,
                             name='GT images', range=(-1, 1), nrow=1)

        querier = trainer.imagen.unets[0].querier # Put to eval or torch nograd
        if sample_ids == 'all':
            sample_ids = list(np.arange(num_queries))
        elif -1 in sample_ids:
            sample_ids.remove(-1)
            sample_ids.append(num_queries-1)


        q_mask = torch.ones((N, max_num_attributes, max_num_objects), device=input_x.device, dtype=attrs.dtype).to(device)
        q_all = torch.linspace(1, max_num_attributes, max_num_attributes, device=input_x.device, dtype=attrs.dtype)[None, :, None].to(device)
        q = (q_mask * q_all).reshape(N, -1)

        attr_embeds = trainer.imagen.unets[0].cond_embedding(q)
        attr_embeds_neg = trainer.imagen.unets[0].cond_embedding_neg(q)
        attr_embeds_unasked = trainer.imagen.unets[0].cond_embedding(torch.zeros_like(q))


        # Select the asked queries in their embeddings according to the answers.
        ans_neg = torch.zeros_like(q)[..., None]
        ans_pos = ans_neg.clone()
        ans_unasked = ans_neg.clone()
        cond_pos, cond_neg, cond_unasked = (attr_embeds * ans_pos), \
                             (attr_embeds_neg * ans_neg), (attr_embeds_unasked * ans_unasked)

        q_mask = q_mask * 0
        gt_attrs_rem = attrs.to(device)
        input_x = input_x.to(device)
        queried_attrs = []
        sampled_im_list = []
        attn_list = []
        for i in range(num_queries):

            # Get new query with querier and grads
            in_embeds = torch.stack([cond_pos, cond_neg, cond_unasked], dim=-1)
            in_ans = torch.stack([ans_pos, ans_neg, ans_unasked], dim=-1)

            with torch.no_grad():
                q_new, attn = querier(cond=in_embeds, ans=in_ans,
                                      image=input_x, mask=q_mask.reshape(N, -1),
                                      return_attn=True)

            q_mask = torch.clamp(q_mask + q_new.reshape(*q_mask.shape), 0, 1)
            q_new = q_new.reshape(N, max_num_attributes, max_num_objects)

            # answer new query
            # ans_new, chosen_attr, gt_attrs_rem  = \
            #     answer_single_query(q_new.reshape(N, max_num_attributes, max_num_objects),
            #                         gt_attrs_rem)
            ans_new, ans_all = \
                answer_queries(q_new, gt_attrs_rem)


            chosen_attr = []
            # TODO: make sure the +1 makes sense
            chosen_attr.append(q_new.max(1)[1].sum(1).cpu().numpy() + 1)
            chosen_attr.append(q_new.max(2)[1].sum(1).cpu().numpy() + 1)
            chosen_attr.append(ans_new[ans_new != 0].cpu().numpy())

            queried_attrs.append(np.stack(chosen_attr, axis=1))

            # select the asked queries in their embeddings according to the answers.
            cond_pos = torch.where(ans_new == 1, attr_embeds * q_new.reshape(N, -1, 1), cond_pos)
            cond_neg = torch.where(ans_new == -1, attr_embeds_neg * q_new.reshape(N, -1, 1), cond_neg)
            cond_unasked = torch.where(ans_new != 0, torch.zeros_like(cond_unasked), cond_unasked)

            ans_pos = torch.where(ans_new == 1, torch.ones_like(ans_pos), ans_pos)
            ans_neg = torch.where(ans_new == -1, -torch.ones_like(ans_pos), ans_neg)
            ans_unasked = torch.where(ans_new != 0, torch.zeros_like(ans_unasked), ans_unasked)

            if i in sample_ids:
                condition = torch.stack([cond_pos, cond_neg], dim=-1)
                answers = torch.stack([ans_pos, ans_neg], dim=-1)
                condition = torch.cat([condition, answers], dim=2)
                if num_samples > 0:
                    for j in range(num_samples):
                        images = trainer.sample(text_embeds=condition, cond_images=cond_images, batch_size=N, cond_scale=cond_scale)
                        sampled_im_list.append(images.cpu())
            attn_plot = attn.reshape(N, 1, max_num_attributes, max_num_objects).clone().cpu()
            attn_list.append(attn_plot)

        # condition = torch.stack([cond_pos, cond_neg], dim=-1)

        try:
            queried_attrs = np.stack(queried_attrs, axis=1)
            # Report selected queries
            text_table = wandb.Table(columns=["Queried Attributes IDs"])
            for i in range(N):
                text_table.add_data('_  _'.join(str(int(e[0]))+'|'+
                                                  str(int(e[1]))+'|'+
                                                  str(int(e[2]))
                                                  for e in list(queried_attrs[i])))
            wandb.log({"Queried Attributes": text_table})
        except: print('Chosen attributes didn\'t have the same shape.')

        if log:
            if len(sampled_im_list)>0:
                utils.log_images(torch.stack(sampled_im_list, dim=1).flatten(0,1), wandb,
                                 name='S_sample_gen_wS', range=(-1,1), nrow=len(sample_ids) * num_samples)
            if len(attn_list)>0:
                utils.log_images(torch.stack(attn_list, dim=1).flatten(0,1), wandb, name='Q_attention', range=(0,1), nrow=num_queries)

    def sample_with_gt(trainer, num_samples=1, cond_scale=3., epoch=0, log=True):

        im_list = []

        _, (input_x, attr_embeds, cond_images) = next(enumerate(testloader))
        N, C, W, H = input_x.shape

        text_table = wandb.Table(columns=["Attribute IDs"])
        for i in range(N):
            text_table.add_data(' '.join(str(e) for e in list(attr_embeds[i].numpy())))
        wandb.log({"Attributes" : text_table})

        if not log:
            utils.save_images(input_x, os.path.join(model_dir_ckpt, 'GT_sample.png'), range=(-1,1))
        else:
            im_list.append(input_x)
            utils.log_images(input_x, wandb,
                             name='GT images', range=(-1,1), nrow=1)

        sample_ids = [0]
        for j in range(num_samples):
            images = trainer.sample(text_embeds=attr_embeds, cond_images=cond_images, batch_size=input_x.shape[0], cond_scale = cond_scale)
            im_list.append(images.cpu())

        if log:
            if len(im_list)>0:
                utils.log_images(torch.stack(im_list, dim=1).flatten(0,1), wandb,
                                 name='Generated Samples with S', range=(-1,1), nrow=(len(sample_ids) * num_samples) + 1)


    # Load epoch
    load_epoch = 0
    if args.load_epoch > 0:
        load_epoch = args.load_epoch
        if hasattr(trainer.imagen.unets[0], 'querier'):
            trainer.imagen.unets[0].querier.tau = tau_vals[load_epoch]
        trainer.load(os.path.join(model_dir_ckpt, 'ckpt', f'epoch{load_epoch}.pt'))


    # Test
    if args.test:
        sample_with_querier(trainer, sample_ids=[-1], num_samples=1, num_queries=10, query_size=(qh, qw), epoch=load_epoch, log=True, full_queries=False)
        if args.all_queries:
            # TODO: implement num_samples to be sampled in parallel.
            sample_with_gt(trainer, num_samples=args.num_samples, cond_scale=cond_scale, epoch=load_epoch, log=True)
        exit()

    # Train
    trainer.imagen.unets[0].args.train_querier = args.train_querier
    trainer.imagen.unets[0].sampling = args.sampling
    for epoch in range(load_epoch, args.epochs, 1):
        if hasattr(trainer.imagen.unets[0], 'querier'):
            trainer.imagen.unets[0].querier.tau = tau_vals[epoch]
            # if epoch % 1 == 0 and epoch > load_epoch:
            #     sample_ids = [-1]
            #     num_samples = 1 #if epoch > 0 else 0
            #     if epoch % 5 == 0:
            #         trainer.imagen.unets[0].args.all_queries = True
            #         sample_with_gt(trainer, num_samples=num_samples, cond_scale=cond_scale, epoch=load_epoch, log=True)
            #         trainer.imagen.unets[0].args.all_queries = False
            #     sample_with_querier(trainer, sample_ids=sample_ids, num_queries=10,
            #                         num_samples=0, query_size=(qh, qw), epoch=epoch, log=True)
            #     wandb.log({'lr':trainer.get_lr(1)})
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
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    main(args)


