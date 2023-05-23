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
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torchvision
from PIL import Image
# pip install pytorch-ignite # conda install ignite -c pytorch
# python classifier_train.py --data_dir /cis/home/acomas/data/ --iterations 300000 --anneal_lr True --batch_size 10 --lr 3e-4 --save_interval 10000 --weight_decay 0.05 --image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True
# mpiexec -n 2 python scripts/classifier_train.py --data_dir /cis/home/acomas/data/ --iterations 300000 --anneal_lr True --batch_size 10 --lr 3e-4 --save_interval 10000 --weight_decay 0.05 --image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width=128 --classifier_pool=attention --classifier_resblock_updown=True --classifier_use_scale_shift_norm=True
def main():
    args = create_argparser().parse_args()

    gt = False

    # [ samples_test_MNIST_32-3p_ep100_gt_new_good, samples_test_MNIST_32-3p_ep100_nogt_new_good,
    # test_CelebA_ep365rb_64p5_gt_more_2, samples_test_Clevr-masks-128p5_60, samples_Clevr-attr-color-test_7q]
    
    # exp_name = 'samples_test_CelebA_ep365rb_64p5_gt_more_2' # CELEBA
    # exp_name = 'samples_test_MNIST_32-3p_ep100_gt_new_good' # MNIST GT
    # exp_name = 'samples_test_MNIST_32-3p_ep100_nogt_new_good'  # MNIST noGT
    exp_name = 'samples_test_Clevr-masks-128p5_60'
    # exp_name = 'samples_Clevr-attr-color-test_7q'

    args.data_dir = os.path.join('/cis/home/acomas/data/', exp_name)

    load_npy = False
    compute_metrics = True
    # Option 1:
    # experiment = 'patches'

    # Option 2:
    experiment = 'patches-foreground'

    # Option 3:
    # experiment = 'attributes'

    # mask_type = 'all'

    if experiment == 'patches':
        metrics = ['mse', 'lpips']
    elif experiment == 'patches-foreground':
        metrics = ['mse']
    elif experiment == 'attributes':
        metrics = ['attributes']
    else: raise NotImplementedError

    save = True #True
    mask_types = ['all', 'asked', 'unasked']
    baselines = ['', 'random'] #, 'prior']#, 'prior']#, 'random']

    th.set_num_threads(1)

    if not save:
        run = wandb.init(project="Class-Test", name=args.name, mode='online')

        wandb.config.update(args)

    logger.configure()
    logger.log("creating model and diffusion...")
    num_attributes = args.num_attrs

    if 'lpips' in metrics and not load_npy:
        # lpips_fn = lpips.LPIPS(net='alex')  # best forward scores
        lpips_fn = lpips.LPIPS(net='vgg').to(dist_util.dev())

    if 'attributes' in metrics and experiment == 'attributes' and not load_npy:
        dist_util.setup_dist()

        model, diffusion = create_classifier_and_diffusion(
            **args_to_dict(args, classifier_and_diffusion_defaults().keys())
        , num_attributes=num_attributes)
        model.to(dist_util.dev())

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


    def forward_log(data_loader, prefix="test", mask_type='all'):

        accs, mses, lpipss, logit_list, label_list = [], [], [], [], []

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

                if experiment == 'attributes':
                    sub_labels[sub_labels == -1] = 0
                    # sample_ids[sample_ids == 45] = 44  # TODO: Hack
                    # print(sample_ids)
                    with th.no_grad():
                        logits = model(sub_batch, timesteps=sub_t)
                    # loss = F.cross_entropy(logits, sub_labels, reduction="none"

                    queries = extra['q'][:, None, None].repeat(1, NS, 1, 1, 1).type(th.long)
                    max_attrs, max_obj = 9, 5
                    if mask_type == 'all':
                        mask = th.ones_like(logits)
                    else:
                        mask = th.zeros((N * NS, max_attrs, max_obj)).to(logits.device)
                        queries = queries.reshape(N * NS, -1, 3)[..., :2] - 1
                        batch = th.arange(N*NS).type(th.long)
                        masks = []
                        for i in range(queries.shape[1]):
                            if i > 0:
                                mask[batch, queries[batch, i-1, 0], queries[batch, i-1, 1]] = 1
                            if i in list(sample_ids[0]):
                                masks.append(mask.clone())
                        mask = th.stack(masks, 1).reshape(*logits.shape)
                        # print([m1 == m2 for m1, m2 in zip(masks[:-1],masks[1:])])
                        if mask_type == 'asked':
                            pass
                        elif mask_type == 'unasked':
                            mask = 1-mask


                    acc = compute_acc(
                        logits, sub_labels, reduction="none"
                    )
                    acc_mean = (acc * mask).reshape(B, -1).sum(1) / mask.reshape(B, -1).sum(1)
                    accs.append(acc_mean.reshape(N, NS, NQ))
                    # logit_list.append(logits.detach().cpu()); logit_list.append(sub_labels.detach().cpu())
                    # losses[f"total_acc"] = acc

                if experiment.startswith('patches'):

                    if experiment.endswith('foreground'):
                        max, min = 0.5, -0.25
                        batch_rgb, labels_rgb = batch, labels
                        batch, labels = th.zeros_like(batch_rgb[..., :1, :, :]), th.zeros_like(labels_rgb[..., :1, :, :])
                        fg_batch = (batch_rgb < max) * (batch_rgb > min )
                        fg_batch = fg_batch[:, :1] * fg_batch[:, 1:2] * fg_batch[:, 2:3]
                        batch[~fg_batch] = 1

                        fg_labels = (labels_rgb < max) * (labels_rgb > min )
                        fg_labels = fg_labels[:, :1] * fg_labels[:, 1:2] * fg_labels[:, 2:3]
                        labels[~fg_labels] = 1
                        # GT = labels.reshape(N, 1, batch_rgb.shape[-1], batch_rgb.shape[-1])
                        # save_images(GT, 'gt_mask.png', save=True, show=False, range=(0, 1), nrow=1)

                        # Compute foreground mask with hardcoding

                    if mask_type == 'all':
                        mask = th.ones_like(batch)
                    elif 'q' in extra.keys():
                        queries = extra['q'].repeat(1, NS, 1, 1, 1, 1).flatten(0, 1)
                        queries[queries != -10] = 1
                        queries[queries == -10] = 0
                        # print(queries.shape, sample_ids)
                        sel_queries = queries[:, sample_ids[0]].to(batch.device)
                        mask = sel_queries.clone()
                        # perc_asked = th.ones_like(sel_queries.reshape(B, -1)).sum(1) / sel_queries.reshape(B, -1).sum(1)

                        # masks = []
                        # for i in range(queries.shape[1]):
                        #     if i not in list(sample_ids[0]):
                        #         continue
                        #     masks.append(queries[:, i].clone())
                        # mask = th.stack(masks, 1).reshape(*logits.shape)
                        if mask_type == 'asked':
                            pass
                        elif mask_type == 'unasked':
                            mask = 1-mask
                    else: print('No masks!'); return {}, None
                    mask = mask.reshape(B, -1)
                    #TODO: MSE no reduce, lpips, then reshape back to sequence.
                    pred, target = batch.reshape(B, -1),\
                                    labels.reshape(B, -1) #.reshape(N * NS, NQ, *batch.shape[-3:]),
                    mse = (pred - target)**2
                    mean_mse = (mse * mask).sum(1) / mask.sum(1)
                    mses.append(mean_mse.reshape(N, NS, NQ))

                    if 'lpips' in metrics and mask_type == 'all':

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

    def plot_data(data_loader, plot_idx=0, suffix='', prefix="test"):

        for idx, (batch, extra) in enumerate(data_loader):
            if idx < plot_idx:
                continue
            N, NS, NQ, C, W, H = batch.shape

            B = N * NS

            batch = batch.reshape(N * NS, NQ, *batch.shape[3:])
            sample_ids = extra["sample_q_id"]

            if experiment == 'attributes':
                target = extra["gt"].reshape(N, 1, 1, C, W, H)

            if experiment.startswith('patches'):
                target = extra["y"].reshape(N, 1, 1, C, W, H)

            save_images(target.reshape(N, C, W, H), os.path.join(args.data_dir, f"figures/gt.png") , save=True, show=False, range=(-1, 1), nrow=1)
            for ii in range(NQ):
                path_gen = os.path.join(args.data_dir, f"figures/gen_Q{sample_ids[0][ii]}{suffix}.png")
                save_images(batch[:, ii].reshape(B, C, W, H), path_gen, save=True, show=False, range=(-1, 1), nrow=NS)

            if experiment == 'patches':
                if 'q' in extra.keys():
                    c, w, h = extra['q'].shape[-3:]
                    path_gen = os.path.join(args.data_dir, f"figures/q_Q0-{sample_ids[0][-1]}{suffix}.png")
                    q = extra['q'].reshape(N, -1, c, w, h)[:, :sample_ids[0][-1] + 1].reshape(N*(sample_ids[0][-1] + 1), c, w, h)
                    q[ q == -10 ] = 0
                    save_images(q, path_gen, save=True, show=False, range=(-1, 1), nrow=(sample_ids[0][-1] + 1))

                if 'attn' in extra.keys():
                    path_gen = os.path.join(args.data_dir, f"figures/att_Q0-{sample_ids[0][-1]}{suffix}.png")
                    qs = extra['attn'].shape[-1]
                    attn = extra['attn'].reshape(N, -1, 1, qs, qs)[:, :sample_ids[0][-1] + 1].reshape(N*(sample_ids[0][-1] + 1), 1, qs, qs)
                    save_images(attn, path_gen, save=True, show=False, range=(0, 1), nrow=(sample_ids[0][-1] + 1))

            print('iter', idx)

            break

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    sns.set_theme(style="darkgrid")
    sns.set(font_scale=1.5)
    csfont = {'fontname': 'Times New Roman'}
    # sns.set(font="Times New Roman")
    # sns.set(style="white")

    def get_metric_figure(d, name):
        if not d:
            raise ValueError
        df = pd.DataFrame(d)
        # a4_dims = (11, 8)
        fig, ax = plt.subplots()
        fig = sns.relplot(data=df, kind="line",
            x="Query number", y=name, hue='baseline', style='experiment',#, errorbar="sd",
           markers = ['o', '<', 's'], markersize=6, ax=ax
        )
        fig._legend.remove()
        ax.set_xlim(0, df['Query number'].max())
        if name == 'Test Accuracy':
            ax.set_ylim(0.86, 1)
        return fig

    os.makedirs(os.path.join(args.data_dir, f"figures"), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, f"metrics"), exist_ok=True)

    def update_dict(d, name, val, query_ids, exp, baseline):
        N, NS, NQ = val.shape[:3]
        if not d:
            d = {name: val.reshape(-1), 'Query number': query_ids[None].repeat(N * NS, 0).reshape(-1),
                     'experiment': [exp] * N * NS * NQ,
                     'baseline': ['model' + baseline] * N * NS * NQ}
        else:
            d = {name: np.concatenate([d[name], val.reshape(-1)]),
                     'Query number': np.concatenate([d['Query number'], query_ids[None].repeat(N * NS, 0).reshape(-1)]),
                     'experiment': d['experiment']+([exp] * N * NS * NQ),
                     'baseline': d['baseline']+(['model' + baseline] * N * NS * NQ)}

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
        if load_npy:
            files = os.listdir(os.path.join(args.data_dir, 'metrics'))
            files.sort()
            if len(files) > 0:
                for file in files:
                    case_name = file.split('/')[-1].strip('.npy')
                    if case_name == 'query_ids':
                        continue
                    # file_dict.update({case_name: np.load(file)})
                    query_ids = np.load(os.path.join(args.data_dir, 'metrics/query_ids.npy'))
                    l = case_name.split('_')
                    if len(l) == 2:
                        met, mask_type = l; suffix = ''
                    else: met, mask_type, suffix = l; suffix = '_' + suffix
                    if met == 'mse':
                        name = 'Mean-Squared Error'
                        val = np.load(os.path.join(args.data_dir, 'metrics', file))
                        d_mse = update_dict(d_mse, name, val, query_ids, mask_type, suffix)
                    if met == 'lpips':
                        name = 'LPIPS'
                        val = np.load(os.path.join(args.data_dir, 'metrics', file))
                        d_lpips = update_dict(d_lpips, name, val, query_ids, mask_type, suffix)
                    if met == 'acc':
                        name = 'Test Accuracy'
                        val = np.load(os.path.join(args.data_dir, 'metrics', file))
                        d_acc = update_dict(d_acc, name, val, query_ids, mask_type, suffix)


                    # + mask_type + suffix
        #     else: raise ValueError

        # check that theres files in it.
        else:
            if save:
                plot_data(data, plot_idx=5, suffix=suffix)
                # exit()
            if compute_metrics:
                for mask_type in mask_types:
                    dict, query_ids = forward_log(data, mask_type=mask_type)
                    keys = dict.keys()

                    if 'mse' in keys:
                        mse_val = dict['mse'].cpu().numpy()
                        d_mse = update_dict(d_mse, 'Mean-Squared Error', mse_val, query_ids, mask_type, suffix)

                        if save:
                            # fig.savefig(os.path.join(args.data_dir, f"figures/plot_mse.png"))
                            np.save(os.path.join(args.data_dir, 'metrics/mse_' + mask_type + suffix), mse_val)
                            np.save(os.path.join(args.data_dir, 'metrics/query_ids'), query_ids)
                        mean = mse_val.mean(0).mean(0)
                        print("MSE: ", mean)
                    if 'lpips' in keys:
                        lpips_val = dict['lpips'].cpu().numpy()
                        d_lpips = update_dict(d_lpips, 'LPIPS', lpips_val, query_ids, mask_type, suffix)
                        if save:
                            np.save(os.path.join(args.data_dir, 'metrics/lpips_' + mask_type + suffix), lpips_val)
                            np.save(os.path.join(args.data_dir, 'metrics/query_ids'), query_ids)
                        mean = lpips_val.mean(0).mean(0)
                        print("LPIPS: ", mean)
                    if 'acc' in keys:
                        acc_val = dict['acc'].cpu().numpy()
                        d_acc = update_dict(d_acc, 'Test Accuracy', acc_val, query_ids, mask_type, suffix)
                        if save:
                            np.save(os.path.join(args.data_dir, 'metrics', 'acc_' + mask_type + suffix), acc_val)
                            np.save(os.path.join(args.data_dir, 'metrics/query_ids'), query_ids)
                        mean = acc_val.mean(0).mean(0)
                        print("Accuracy: ", mean)

    for k, d in zip(['mse', 'lpips', 'acc'],[d_mse, d_lpips, d_acc]):
        if d and save:
            if k == 'mse': name = 'Mean-Squared Error'
            if k == 'lpips': name = 'LPIPS'
            if k == 'acc': name = 'Test Accuracy'
            fig = get_metric_figure(d, name)
            fig.savefig(os.path.join(args.data_dir, f"figures/plot_{k}.png"))



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



if __name__ == "__main__":
    main()
