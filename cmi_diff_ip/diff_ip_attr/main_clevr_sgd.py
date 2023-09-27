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
from dataloader import Clevr, Clevr_with_masks, Clevr_with_attr, Cub
import torchvision as tv
import lovely_tensors as lt
lt.monkey_patch()

# TODO: Change directly the input to the querier when we have new answers.
# ps -up `nvidia-smi -q -x | grep pid | sed -e 's/<pid>//g' -e 's/<\/pid>//g' -e 's/^[[:space:]]*//'`
# conda activate imagen; cd projects/clevr_diff_ip/; accelerate launch main_clevr_attr.py --batch_size 16 --sampling biased --test --name querier128SAUnet_S3000biased_and_unbiased_withmask_hard

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--data', type=str, default='clevr')
    parser.add_argument('--batch_size', type=int, default=24) # 128
    parser.add_argument('--max_queries', type=int, default=50)
    parser.add_argument('--max_queries_test', type=int, default=25)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--tau_start', type=float, default=1.0)
    parser.add_argument('--tau_end', type=float, default=1.0)
    parser.add_argument('--lambd', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--tail', type=str, default='', help='tail message')
    parser.add_argument('--ckpt_path', type=str, default=None, help='load checkpoint')
    parser.add_argument('--save_dir', type=str, default='./saved/', help='save directory')
    parser.add_argument('--run_name', type=str, default='debug_clevr_biased_random', help='save directory')
    parser.add_argument('--data_dir', type=str, default='/cis/home/acomas/data/', help='save directory')
    parser.add_argument('--num_viz', type=int, default=4)
    parser.add_argument('--cfg_scale', type=float, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_sampling_steps', type=int, default=300)
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--load_epoch', type=int, default=0)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--cond_scale', type=float, default=5)
    parser.add_argument('--attribute', type=str, default="color")
    parser.add_argument('--num_samples', type=int, default=1)

    parser.add_argument('--freeze_unet', type=bool, default=False)
    parser.add_argument('--train_querier', type=bool, default=True) # Doesn't work?
    parser.add_argument('--all_queries', type=bool, default=False)
    parser.add_argument('--sampling', type=str, default='random')
    parser.add_argument('--max_queries_biased', type=int, default=10)
    parser.add_argument('--embed_dim', type=int, default=8)
    parser.add_argument('--cond_dim', type=int, default=8)
    parser.add_argument('--experiment_type', type=str, default='attributes')
    parser.add_argument('--null_val', type=float, default=-10)
    parser.add_argument('--query_mode', type=str, default='single_queries') # other option is 'encoder-decoder'

    args = parser.parse_args()
    args.train_querier = False #True
    args.freeze_unet = False
    args.all_queries = False
    args.embed_dim = 16 # 16
    args.cond_dim = 256
    # args.experiment_type = 'attributes'
    if args.num_viz > args.batch_size:
        args.num_viz = args.batch_size
    for k, v in args.__dict__.items():
        print(k, v)
    # print(args)
    return args
# CUDA_VISIBLE_DEVICES=7 python main_clevr.py --batch_size 16 --name embed_32_flatten --embed_dim 32 --experiment_type attributes --load_epoch 599  --sampling random --attribute color --num_samples 5 --cond_scale 2 --train_querier=True
# CUDA_VISIBLE_DEVICES=5 python main_clevr.py --batch_size 9 --num_samples 3 --test True --load_epoch 410 --cond_scale 10 --name x3_attn_context_bottleneck --query_mode flatten_obj
# CUDA_VISIBLE_DEVICES=1 python main_clevr.py --batch_size 10 --experiment_type attributes --sampling random --run_name query_w_qk_soft_tau_train_all_realrandom --name retrain_querier_scale_soft --load_epoch 475
def main(args):
    ## Setup
    # wandb
    if args.name is None:
        args.name = args.run_name
    if args.run_name == 'debug':
        run = wandb.init(project="Diff-IP-debug", name=args.name, mode=args.mode)
    else:
        run = wandb.init(project="Diff-IP-Clevr-attr", name=args.name, mode=args.mode)
    # model_dir = os.path.join(args.save_dir, f'{run.id}')
    # os.makedirs(model_dir, exist_ok=True)
    # torch.set_num_threads(1)

    model_dir_ckpt = os.path.join(args.save_dir, args.name)
    model_dir_ckpt_load = os.path.join(args.save_dir, args.run_name)
    # print(model_dir_ckpt_load, model_dir_ckpt)
    os.makedirs(os.path.join(model_dir_ckpt, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(model_dir_ckpt_load, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.name), exist_ok=True)
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

    pred_objectives = ['noise', 'x_start','v']
    pred_objective = pred_objectives[0]

    H = W = 128
    C = 3
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x * 2 - 1),
                                    transforms.Resize(size=(H, W))
                                    ])

    main_dim = 32

    ## Constants
    if args.experiment_type == 'patches':
        patch_size = 5
        qh = qw = H - patch_size + 1
        cond_images_channels = C
        attr_size = max_num_attributes = max_num_objects = None
        use_attr = cond_on_text = False
        dl_keynames = ('images', 'cond_images')

        ## Data
        trainset = Clevr(args.data_dir, split='train', transform=transform)
        testset = Clevr(args.data_dir, split='test', transform=transform)

    elif args.experiment_type == 'attributes':
        attribute = args.attribute
        max_num_objects = 5
        if args.attribute == 'color':
            max_num_attributes = 8 + 1 # num attrs + num_obj dimension
        elif args.attribute == 'shape':
            max_num_attributes = 3 + 1 # num attrs + num_obj dimension
        else: raise NotImplementedError
        qh, qw = max_num_objects, max_num_attributes
        max_num_queries = max_num_attributes * max_num_objects
        patch_size = None
        # if args.query_mode == 'flatten': # Note: uncomment for flatten_attr_process_token
        #     attr_size = max_num_queries * args.embed_dim
        # elif args.query_mode == 'flatten_obj':
        #     attr_size = max_num_objects * args.embed_dim
        # elif args.query_mode == 'encoder-decoder':
        #     attr_size = args.cond_dim

        attr_size = args.cond_dim
        args.max_queries_biased = 10 # max_num_queries
        cond_images_channels = C
        use_attr = cond_on_text = True
        dl_keynames = ('images', 'text_embeds', 'cond_images')

        ## Data
        trainset = Clevr_with_attr(args.data_dir, split='train', transform=transform, attribute=attribute, max_attributes=max_num_objects)
        testset = Clevr_with_attr(args.data_dir, split='test', transform=transform, attribute=attribute, max_attributes=max_num_objects)

    elif args.experiment_type == 'cub':

        patch_size = None
        max_num_attributes = 312
        max_num_objects = 1
        qh, qw = max_num_objects, max_num_attributes
        max_num_queries = max_num_attributes * max_num_objects

        attr_size = args.cond_dim
        args.max_queries_biased = 10  # max_num_queries
        cond_images_channels = C
        use_attr = cond_on_text = True
        dl_keynames = ('images', 'text_embeds', 'cond_images')

        trainset = Cub(args.data_dir, train=True, transform=transform, download=False)
        testset = Cub(args.data_dir, train=False, transform=transform, download=False)

    cond_scale = args.cond_scale

    testloader = DataLoader(testset, batch_size=4, num_workers=args.num_workers, pin_memory=True)

    iters_per_epoch = len(trainset) // args.batch_size

    unet1 = Unet( # TODO: Increase dim to 128 at the expense of other hyperparams.
        dim = main_dim,
        image_size = (W, H),
        patch_size = patch_size,
        sampling_mode = args.sampling,
        max_num_queries = 20,
        max_rand_queries = 100,
        max_num_attributes = max_num_attributes,
        max_num_objects = max_num_objects,
        image_embed_dim = 1024,
        cond_on_text = cond_on_text, # Needed (?)
        text_embed_dim = attr_size, # Not sure if it's necessary, but a bunch of things are done with this
        cond_dim = attr_size,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, False, True, True),
        layer_cross_attns = (True, True, True, True), # TODO: Checked all true
        max_text_len = max_num_attributes,
        cond_images_channels = cond_images_channels, # S + GT Image,
        FLAGS = args,
    )

    imagen = Imagen(
        unets = (unet1),
        image_sizes = (H),
        timesteps = 1000,
        text_embed_dim = attr_size,
        cond_drop_prob=0.1,
        condition_on_text = use_attr,
        pred_objectives = pred_objective
    )

    trainer = ImagenTrainer(imagen,
                            warmup_steps = 1 * iters_per_epoch,
                            cosine_decay_max_steps = (args.epochs - 1) * iters_per_epoch,
                            split_valid_from_train = True,
                            dl_tuple_output_keywords_names = dl_keynames).cuda()

    trainer.add_train_dataset(trainset, batch_size = args.batch_size)

    tau_vals = np.linspace(args.tau_start, args.tau_end, args.epochs)

    # Load epoch
    load_epoch = 0
    if args.load_epoch > 0:
        load_epoch = args.load_epoch
        if hasattr(trainer.imagen.unets[0], 'querier') and not args.train_querier:
            trainer.imagen.unets[0].querier.tau = tau_vals[load_epoch]
        trainer.load(os.path.join(model_dir_ckpt_load, 'ckpt', f'epoch{load_epoch}.pt'))
        if args.train_querier:
            load_epoch = 0
            print('New starting epoch {}'.format(load_epoch))

    # Freeze network parameters to train querier
    if args.freeze_unet and args.train_querier:
        for param_name, param in trainer.imagen.named_parameters():
            if "querier" not in param_name:
                param.requires_grad = False
            # print(param.requires_grad)
        print('Frozen unet!!\n\n')

    # Train querier
    trainer.imagen.unets[0].args = args
    trainer.imagen.unets[0].querier.tau = tau_vals[load_epoch]

    def sample_with_querier(trainer, sample_ids, num_queries=5, num_samples=1, epoch=0, max_query=100, log=False, full_queries=False):

        im_list = []
        soft = False
        if args.experiment_type == 'patches':
            _, (input_x, input_cond) = next(enumerate(testloader))
            input_x, input_cond = input_x.to(device), input_cond.to(device)
            N, C, W, H = input_cond.shape

        elif args.experiment_type == 'attributes':
            _, (input_x, attrs, cond_images) = next(enumerate(testloader))
            N, C, W, H = input_x.shape

            text_table = wandb.Table(columns=["Attribute IDs"])
            for i in range(N):
                text_table.add_data(' '.join(str(e) for e in list(attrs[i].numpy())))
            wandb.log({"Attributes": text_table})
        elif args.experiment_type == 'cub':
            _, (input_x, attrs, cond_images) = next(enumerate(testloader))
            N, C, W, H = input_x.shape

        # N, device = input_x.shape[0], input_x.device
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

        if args.experiment_type == 'patches':

            input_cond = resize_image_to(input_cond, qh + patch_size - 1)
            masked_x = torch.zeros_like(input_cond) + args.null_val
            mask = torch.zeros(N, qh * qw).to(device)

        elif args.experiment_type == 'attributes' or args.experiment_type == 'cub':

            q_mask = torch.ones((N, max_num_attributes, max_num_objects), device=input_x.device, dtype=attrs.dtype).to(device)
            q_all = torch.linspace(1, max_num_attributes, max_num_attributes, device=input_x.device, dtype=attrs.dtype)[None, :, None].to(device)
            q = (q_mask * q_all).reshape(N, -1)

            attr_embeds = trainer.imagen.unets[0].cond_embedding(q).detach()
            attr_embeds_neg = trainer.imagen.unets[0].cond_embedding_neg(q).detach()
            attr_embeds_unasked = trainer.imagen.unets[0].cond_embedding(torch.zeros_like(q)).detach()
            obj_embed = trainer.imagen.unets[0].query_encoder.object_embedding(N, device).detach()

            ans_all = answer_queries(q_mask.clone(),  # Note: Q is not really used
                                     attrs.clone())  # q: [q x num_attr x max_obj], binary - gt_attrs: [b x max_obj]
            ans = ans_all  # * q_mask.reshape(*ans_all.shape)

            ans_neg = torch.zeros_like(q)[..., None]
            ans_pos = ans_neg.clone()
            ans_unasked = ans_neg.clone()
            cond_pos, cond_neg, cond_unasked = (attr_embeds * ans_pos), \
                (attr_embeds_neg * ans_neg), (attr_embeds_unasked * ans_unasked)

            ans_neg_, ans_pos_, ans_unasked_ = ans_neg.clone(), ans_pos.clone(), ans_unasked.clone()
            ans_neg_, ans_pos_ = ans_neg_ +1, ans_pos_ +1 # Evaluate all
            # ans_neg_[ans == -1] = 1
            # ans_pos_[ans == 1] = 1

            cond_pos_all, cond_neg_all = \
                (attr_embeds * ans_pos_), \
                (attr_embeds_neg * ans_neg_)

            cond_pos_all_o, cond_neg_all_o = [torch.cat([c, obj_embed], dim=-1) * m for c, m in
                                                                  zip((cond_pos_all, cond_neg_all),
                                                                      (ans_pos_, ans_neg_))]
            # cond_pos_all, cond_neg_all, cond_unasked_all = [torch.cat([c, obj_embed], dim=1) for c in (cond_pos_all, cond_neg_all, cond_unasked_all)]
            # Select the asked queries in their embeddings according to the answers.


            q_mask = q_mask * 0
            gt_attrs_rem = attrs.to(device)

        queried_attrs = []
        queried_patches = []
        sampled_im_list = []
        attn_list = []
        ans_all = None
        for i in range(num_queries):

            if args.experiment_type == 'patches':

                plot_cond = masked_x.clone()
                queried_patches.append(plot_cond)

                with torch.no_grad():
                    querier_inputs = torch.cat([masked_x, input_cond], dim=1).to(device)
                    query_vec, attn = querier(querier_inputs, mask, return_attn=True)
                    mask[torch.arange(N), query_vec.argmax(dim=1)] = 1.0
                    masked_x = ops.update_masked_image(masked_x, input_cond, query_vec, patch_size=patch_size)

                if i in sample_ids:
                    if num_samples > 0:
                        for j in range(num_samples):
                            cond_images = masked_x
                            images = trainer.sample(cond_images=cond_images, batch_size=N, cond_scale=cond_scale) # Activates EMA, we need to change args to ema_model.
                            sampled_im_list.append(images.cpu())
                attn_plot = attn.reshape(N, 1, qw, qh).clone().cpu()
                attn_list.append(attn_plot)


            elif args.experiment_type == 'attributes' or args.experiment_type == 'cub':
                # Get new query with querier and grads
                # , cond_unasked, ans_unasked TODO: unused for now
                in_embeds = torch.stack([cond_pos, cond_neg], dim=-1)
                in_embeds_all = torch.stack([cond_pos_all_o, cond_neg_all_o], dim=-1)
                in_ans = torch.stack([ans_pos, ans_neg], dim=-1)
                input_x = input_x.to(device)

                with torch.no_grad():
                    q_new_flat, q_soft_flat, attn = querier(cond=in_embeds, cond_all=in_embeds_all, ans=in_ans,
                                          image=input_x, mask=q_mask.reshape(N, -1),
                                          return_attn=True)


                q_new_hard_flat = q_new_flat
                q_new_hard = q_new_flat.reshape(N, max_num_attributes, max_num_objects)

                soft = False
                if soft:
                    q_new_flat = q_soft_flat
                else: q_new_flat = q_new_flat
                q_new = q_new_flat.reshape(N, max_num_attributes, max_num_objects)

                q_mask = torch.clamp(q_mask + q_new.reshape(*q_mask.shape), 0, 1)

                # answer new query
                # ans_new, chosen_attr, gt_attrs_rem  = \
                #     answer_single_query(q_new.reshape(N, max_num_attributes, max_num_objects),
                #                         gt_attrs_rem)
                if args.experiment_type == 'attributes':
                    ans_all = \
                        answer_queries(q_new, gt_attrs_rem, ans_all)
                else: ans_all = gt_attrs_rem

                ans_new = ans_all * q_new_flat
                ans_new_hard = ans_all * q_new_hard_flat

                chosen_attr = []
                # TODO: make sure the +1 makes sense
                chosen_attr.append(q_new_hard.max(1)[1].sum(1).cpu().numpy() + 1)
                chosen_attr.append(q_new_hard.max(2)[1].sum(1).cpu().numpy() + 1)
                chosen_attr.append(ans_new_hard[ans_new_hard != 0].cpu().numpy())

                queried_attrs.append(np.stack(chosen_attr, axis=1))

                q_mask = q_mask.reshape(*ans_new.shape)
                bool_pos, bool_neg = (ans_new > 0), (ans_new < 0)


                # select the asked queries in their embeddings according to the answers.
                cond_pos = torch.where(bool_pos, attr_embeds, cond_pos) * q_mask # * q_new.reshape(N, -1, 1) or q_mask modulate the magnitude of the chosen embeds
                cond_neg = torch.where(bool_neg, attr_embeds_neg, cond_neg) * q_mask
                cond_unasked = torch.where(ans_new != 0, torch.zeros_like(cond_unasked), cond_unasked)

                ans_pos = torch.where(bool_pos, torch.ones_like(ans_pos), ans_pos) * q_mask
                ans_neg = torch.where(bool_neg, torch.ones_like(ans_pos), ans_neg) * q_mask
                ans_unasked = torch.where(ans_new != 0, torch.zeros_like(ans_unasked), ans_unasked)

                if i in sample_ids:
                    embeds = torch.stack([cond_pos, cond_neg], dim=-1)
                    answers = torch.stack([ans_pos, ans_neg], dim=-1)
                    condition = torch.cat([embeds, answers], dim=2)
                    if num_samples > 0:
                        for j in range(num_samples):
                            images = trainer.sample(text_embeds=condition, cond_images=cond_images, batch_size=N, cond_scale=cond_scale)
                            sampled_im_list.append(images.cpu())
                attn_plot = attn.reshape(N, 1, max_num_attributes, max_num_objects).clone().cpu()
                attn_list.append(attn_plot)

        if args.experiment_type == 'attributes':
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

            if len(queried_patches)>0:
                utils.log_images(torch.stack(queried_patches, dim=1).flatten(0,1), wandb,
                                 name='S', range=(-1,1), nrow=num_queries)
    def sample_with_gt(trainer, num_samples=1, cond_scale=3., epoch=0, log=True):

        im_list = []

        _, (input_x, attr_embeds, cond_images) = next(enumerate(testloader))
        N, C, W, H = input_x.shape

        if args.experiment_type == 'attributes':
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

    # Test
    if args.test:
        sample_with_querier(trainer, sample_ids=[-1], num_samples=0, num_queries=10, epoch=load_epoch, log=True, full_queries=False)
        if args.all_queries:
            # TODO: implement num_samples to be sampled in parallel.
            sample_with_gt(trainer, num_samples=args.num_samples, cond_scale=cond_scale, epoch=load_epoch, log=True)
        exit()

    # Train
    trainer.imagen.unets[0].args.train_querier = args.train_querier
    trainer.imagen.unets[0].sampling = args.sampling
    for epoch in range(load_epoch, args.epochs, 1):
        print(epoch)
        if hasattr(trainer.imagen.unets[0], 'querier'):
            trainer.imagen.unets[0].querier.tau = tau_vals[epoch]
            if epoch % 1 == 0 and epoch > load_epoch:
                sample_ids = [-1]
                num_samples = 1 if epoch > 0 else 0
                if epoch % 15 == 0 and args.experiment_type != 'patches':
                    trainer.imagen.unets[0].args.all_queries = True
                    trainer.unets[0].args.all_queries = True
                    sample_with_gt(trainer, num_samples=num_samples, cond_scale=cond_scale, epoch=load_epoch, log=True)
                    trainer.imagen.unets[0].args.all_queries = False
                    trainer.unets[0].args.all_queries = False

                sample_with_querier(trainer, sample_ids=sample_ids, num_queries=30,
                                    num_samples=0, epoch=epoch, log=True)
                wandb.log({'lr':trainer.get_lr(1)})
        for _ in tqdm(range(iters_per_epoch)):
            dict_out = trainer.train_step(unet_number=1, max_batch_size=100)
            wandb.log(dict_out)
        print('Epoch {} done.'.format(epoch+1))
        if (epoch % 5 == 0 or epoch == args.epochs - 1) and epoch > load_epoch:
            print('Save ckpt')
            trainer.save(os.path.join(model_dir_ckpt, 'ckpt', f'epoch{epoch}.pt'))

    print('Training Done')
    exit()

if __name__ == '__main__':
    args = parseargs()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    main(args)


