"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os
# os.chdir('/cis/home/acomas/projects/guided-diffusion')
import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import wandb
import lpips
from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data_test
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import log_images, parse_resume_step_from_filename, log_loss_dict
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import torchvision
from PIL import Image
# pip install pytorch-ignite # conda install ignite -c pytorch
# python classifier_train.py --data_dir /cis/home/acomas/data/ --iterations 300000 --anneal_lr True --batch_size 10 --lr 3e-4 --save_interval 10000 --weight_decay 0.05 --image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True
# mpiexec -n 2 python scripts/classifier_train.py --data_dir /cis/home/acomas/data/ --iterations 300000 --anneal_lr True --batch_size 10 --lr 3e-4 --save_interval 10000 --weight_decay 0.05 --image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width=128 --classifier_pool=attention --classifier_resblock_updown=True --classifier_use_scale_shift_norm=True
def main():
    args = create_argparser().parse_args()

    metrics = ['attributes'] #,'mse', 'lpips', 'attributes', 'identity']
    experiment = 'attributes' #'patches'
    mask_type = 'all'
    save = False #True
    metric_mode = 'full'
    baselines = ['', 'random']

    run = wandb.init(project="Class-Test", name=args.name, mode='online')
    th.set_num_threads(1)

    dist_util.setup_dist()
    logger.configure()
    wandb.config.update(args)

    logger.log("creating model and diffusion...")
    num_attributes = args.num_attrs

    if 'lpips' in metrics:
        # lpips_fn = lpips.LPIPS(net='alex')  # best forward scores
        lpips_fn = lpips.LPIPS(net='vgg').to(dist_util.dev())

    if 'attributes' in metrics and experiment == 'attributes':
        model, diffusion = create_classifier_and_diffusion(
            **args_to_dict(args, classifier_and_diffusion_defaults().keys())
        , num_attributes=num_attributes)
        model.to(dist_util.dev())

        # if args.noised:
        #     schedule_sampler = create_named_schedule_sampler(
        #         args.schedule_sampler, diffusion
        #     )

        # logger.log("loading classifier...")
        # classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
        # classifier.load_state_dict(
        #     dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        # )
        # classifier.to(dist_util.dev())
        # if args.classifier_use_fp16:
        #     classifier.convert_to_fp16()
        # classifier.eval()

        classifier = model
        resume_step = 0
        if args.resume_checkpoint:
            resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            classifier.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location="cpu" #dist_util.dev()
                )
            )
        model.to(dist_util.dev())
        if args.classifier_use_fp16:
            model.convert_to_fp16()
        model.eval()

        # Needed for creating correct EMAs and fp16 parameters.
        dist_util.sync_params(model.parameters())

        # mp_trainer = MixedPrecisionTrainer(
        #     model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
        # )

        # model = DDP(
        #     model,
        #     device_ids=[dist_util.dev()],
        #     output_device=dist_util.dev(),
        #     broadcast_buffers=False,
        #     bucket_cap_mb=128,
        #     find_unused_parameters=False,
        # )

    def forward_log(data_loader, prefix="test"):

        accs, mses, lpipss, logit_list, label_list = [], [], [], [], []
        gt = True

        for idx, (batch, extra) in enumerate(data_loader):

            # if isinstance(extra, dict):
            labels = extra["y"].to(dist_util.dev())
            # else: labels = extra.to(dist_util.dev())

            N, NS, NQ, C, W, H = batch.shape
            if gt:
                batch = extra["gt"].reshape(N, 1, 1, C, W, H)
                labels = labels.reshape(N, 1, 1, -1)
                NS, NQ = 1, 1

            else:
                labels = labels[:, None, None, :] \
                    .repeat_interleave(NS, 1).repeat_interleave(NQ, 2)
            batch = batch.to(dist_util.dev()) # N, num_samples_per_dp, num_queries, img

            batch, labels = batch.flatten(0,2), labels.flatten(0,2)

            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
            losses, metrics_dict = {}, {}
            for i, (sub_batch, sub_labels, sub_t) in enumerate(
                split_microbatches(args.microbatch, batch, labels, t)
            ):
                B = batch.shape[0]

                sample_ids = extra["sample_q_id"].numpy()

                if experiment == 'clevr-mask':
                    batch, target = batch.reshape(N * NS, NQ, *batch.shape[-3:]), labels.reshape(N * NS, NQ,
                                                                                                *batch.shape[-3:])
                    # Compute foreground mask with hardcoding

                if experiment == 'attributes':
                    sub_labels[sub_labels == -1] = 0
                    with th.no_grad():
                        logits = model(sub_batch, timesteps=sub_t)
                    # loss = F.cross_entropy(logits, sub_labels, reduction="none"

                    if mask_type == 'all':
                        mask = th.ones_like(logits)
                    else: raise NotImplementedError

                    acc = compute_acc(
                        logits, sub_labels, reduction="none"
                    )
                    acc_mean = (acc * mask).reshape(B, -1).sum(1) / mask.reshape(B, -1).sum(1)
                    accs.append(acc_mean.reshape(N, NS, NQ))
                    # logit_list.append(logits.detach().cpu()); logit_list.append(sub_labels.detach().cpu())
                    # losses[f"total_acc"] = acc

                if experiment == 'patches':
                    #TODO: MSE no reduce, lpips, then reshape back to sequence.
                    pred, target = batch.reshape(B, -1),\
                                    labels.reshape(B, -1) #.reshape(N * NS, NQ, *batch.shape[-3:]),
                    mse = (pred - target)**2
                    mses.append(mse.mean(1).reshape(N, NS, NQ))

                    if 'lpips' in metrics:
                        # pss = []
                        # pred, target = batch.reshape(N*NS, NQ, *batch.shape[1:]), labels.reshape(N*NS, NQ, *batch.shape[1:])
                        # for ii in range(NQ):
                        #     pss.append(lpips_fn(pred[:, ii], target[:, ii]).squeeze())
                        pred, target = batch.reshape(N*NS*NQ, *batch.shape[1:]), labels.reshape(N*NS*NQ, *batch.shape[1:])
                        lpipss.append(lpips_fn(pred, target).squeeze().reshape(N, NS, NQ).detach())

                # [wandb.log({k + '_' + prefix: v.mean().item()}) for k, v in losses.items()]
                print('iter', idx)

                # MSE
                # Celeba Identity
        for k, v in zip(['mse', 'lpips', 'acc'],
                        [mses, lpipss, accs]):
            if len(v) > 0:
                metrics_dict[k] = th.cat(v)

        return metrics_dict, sample_ids[0] #(logit_list, label_list)

    def plot_data(data_loader, plot_idx=0, prefix="test"):

        for idx, (batch, extra) in enumerate(data_loader):
            if idx < plot_idx:
                continue
            N, NS, NQ, C, W, H = batch.shape

            B = N * NS

            batch = batch.reshape(N * NS, NQ, *batch.shape[3:])
            sample_ids = extra["sample_q_id"]

            if experiment == 'clevr-mask':
                a = 1
                # Compute foreground mask with hardcoding

            if experiment == 'attributes':
                target = extra["gt"].reshape(N, 1, 1, C, W, H)

            if experiment == 'patches':
                target = extra["y"].reshape(N, 1, 1, C, W, H)


            save_images(target.reshape(N, C, W, H), os.path.join(args.data_dir, f"figures/gt.png") , save=True, show=False, range=(-1, 1), nrow=1)
            for ii in range(NQ):
                path_gen = os.path.join(args.data_dir, f"figures/gen_Q{sample_ids[0][ii]}.png")
                save_images(batch[:, ii].reshape(B, C, W, H), path_gen, save=True, show=False, range=(-1, 1), nrow=NS)

            print('iter', idx)

            exit()

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    sns.set_theme(style="darkgrid")
    csfont = {'fontname': 'Times New Roman'}
    sns.set(font="Verdana")
    # sns.set(style="white")

    def get_metric_figure(d):
        if not d:
            raise ValueError
        # ys = []
        # for k in d.keys():
        #     if k != query_ids:
        #         ys.append(k)
        df = pd.DataFrame(d)
        fig = sns.relplot(
            data=df, kind="line",
            x="query_id", y="val", hue='experiment', errorbar="sd",
        )
        return fig

    os.makedirs(os.path.join(args.data_dir, f"figures"), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, f"metrics"), exist_ok=True)
    # plot_data(data, plot_idx=0)

    def update_dict(d, val, query_ids, suffix):
        N, NS, NQ = val.shape[:3]
        if not d:
            d = {'val': val.reshape(-1), 'query_id': query_ids[None].repeat(N * NS, 0).reshape(-1),
                     'experiment': ['exp' + suffix] * N * NS * NQ}
        else:
            d = {'val': np.concatenate([d['val'], val.reshape(-1)]),
                     'query_id': np.concatenate([d['query_id'], query_ids[None].repeat(N * NS, 0).reshape(-1)]),
                     'experiment': d['experiment']+(['exp' + suffix] * N * NS * NQ)
                     }
        return d

    d_mse, d_lpips, d_acc, d_id = {}, {}, {}, {}
    for suffix in baselines:

        if suffix != '':
            suffix = '_' + suffix

        logger.log("creating data loader...")
        data = load_data_test(
            data_dir=args.data_dir + suffix,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
            random_crop=False,  # TODO: Changed to false (used to be true)
            deterministic=True,
            split='test',
            experiment=experiment
        )

        dict, query_ids = forward_log(data)
        keys = dict.keys()

        if 'mse' in keys:
            mse_val = dict['mse'].cpu().numpy()
            d_mse = update_dict(d_mse, mse_val, query_ids, suffix)

            # if save:
            #     # fig.savefig(os.path.join(args.data_dir, f"figures/plot_mse.png"))
            #     np.save(os.path.join(args.data_dir, 'metrics/mse_' + metric_mode + suffix), mse_val)
            mean = mse_val.mean(0).mean(0)
            print("MSE: ", mean)
        if 'lpips' in keys:
            lpips_val = dict['lpips'].cpu().numpy()
            d_lpips = update_dict(d_lpips, lpips_val, query_ids, suffix)
            # if save:
            #     np.save(os.path.join(args.data_dir, 'metrics/lpips_' + metric_mode + suffix), lpips_val)
            mean = lpips_val.mean(0).mean(0)
            print("LPIPS: ", mean)
            plt.show(mean)
        if 'acc' in keys:
            acc_val = dict['acc'].cpu().numpy()
            d_acc = update_dict(d_acc, acc_val, query_ids, suffix)
            # if save:
            #     np.save(os.path.join(args.data_dir, 'metric_acc_' + metric_mode + suffix), acc_val)
            mean = acc_val.mean(0).mean(0)
            print("Accuracy: ", mean)

    for k, d in zip(['mse', 'lpips', 'acc'],[d_mse, d_lpips, d_acc]):
        if d and save:
            fig = get_metric_figure(d)
            fig.savefig(os.path.join(args.data_dir, f"figures/plot_{k}_{metric_mode}.png"))



    # for step in range(args.iterations - resume_step):
    #     logger.logkv("step", step + resume_step)
    #     logger.logkv(
    #         "samples",
    #         (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
    #     )
    #     wandb.log(
    #         {"samples":
    #         (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
    #          "step": step + resume_step
    #          },
    #     )

def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)

def compute_acc(logits, labels, reduction="mean"):

    bin_log = F.sigmoid(logits.clone())
    bin_log[bin_log < 0.5] = 0
    bin_log[bin_log >= 0.5] = 1
    if reduction == "mean":
        return (bin_log == labels).float().mean().item()
    elif reduction == "none":
        return (bin_log == labels).float()


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=10000,
        name='debug',
        num_attrs = 1,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def save_images(images, path, range =None, save=False, show=False, **kwargs):
    if range is not None:
        images = 255 * (images - range[0])/(range[1]-range[0])
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr.astype(np.uint8))
    if path is not None and save:
        im.save(path)
    if show:
        im.show()


# def log_images(images, logger, name='image', range =None, **kwargs):
#     if range is not None:
#         images = 255 * (images - range[0])/(range[1]-range[0])
#     grid = torchvision.utils.make_grid(images, **kwargs)
#     ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
#     im = Image.fromarray(ndarr.astype(np.uint8))
#     images = logger.Image(im)
#     logger.log({name: images})
#

if __name__ == "__main__":
    main()

import torch.nn as nn

# wrapper class as feature_extractor
# class WrapperInceptionV3(nn.Module):
#
#     def __init__(self, fid_incv3):
#         super().__init__()
#         self.fid_incv3 = fid_incv3
#
#     @torch.no_grad()
#     def forward(self, x):
#         y = self.fid_incv3(x)
#         y = y[0]
#         y = y[:, :, 0, 0]
#         return y
#
# # use cpu rather than cuda to get comparable results
# device = "cpu"
#
# # pytorch_fid model
# dims = 2048
# block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
# model = InceptionV3([block_idx]).to(device)
#
# # wrapper model to pytorch_fid model
# wrapper_model = WrapperInceptionV3(model)
# wrapper_model.eval();
#
# # comparable metric
# pytorch_fid_metric = FID(num_features=dims, feature_extractor=wrapper_model)

# Important, pytorch_fid results depend on the batch size if the device is cuda.
# https://pytorch.org/ignite/generated/ignite.metrics.FID.html
# https://github.com/mseitzer/pytorch-fid