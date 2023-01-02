"""
Train a diffusion model on images.
"""
import os
import argparse
import json
import torch as th
from composable_diffusion import dist_util, logger
from composable_diffusion.download import download_model
from composable_diffusion.image_datasets import load_data
from composable_diffusion.resample import create_named_schedule_sampler
from model_creation import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_querier_model,
    args_to_dict,
    add_dict_to_argparser,
)
import wandb
from train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    use_captions = args.use_captions

    # run = wandb.init(project="Pre-trained-diff-IP", mode='online')
    # wandb.config.update(args)

    # dist_util.setup_dist()
    log_folder = f'./logs_debug_{args.dataset}_{args.image_size}'
    os.makedirs(log_folder, exist_ok=True)
    logger.configure(log_folder) #, ["tensorboard"]

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    querier_model = create_querier_model(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    json.dump(args_to_dict(args, model_and_diffusion_defaults().keys()),
              open(os.path.join(log_folder, 'arguments.json'), "w"), sort_keys=True, indent=4)

    # print(os.environ["CUDA_VISIBLE_DEVICES"], dist_util.dev())
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    # print(os.environ["CUDA_VISIBLE_DEVICES"], dist_util.dev())

    model.to(dist_util.dev())
    querier_model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)



    # # Load diffusion pre-trained model and freeze.
    model.load_state_dict(th.load(download_model('clevr_pos'), dist_util.dev()))
    for param in model.parameters():
        param.requires_grad = False

    logger.log("creating data loader...")
    data = load_data(
        root=args.data_dir,
        split='train',
        dataset_type=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        use_captions=use_captions,
        deterministic=False,
        random_crop=False,
        random_flip=False
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        querier_model=querier_model,
        diffusion=diffusion,
        dataset=args.dataset,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop_vip()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=500,
        save_interval=1000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset="",
        use_captions=False
    )
    defaults.update(model_and_diffusion_defaults())

    # flags = {
    #     "image_size": 128,
    #     "num_channels": 192,
    #     "num_res_blocks": 2,
    #     "learn_sigma": True,
    #     "use_scale_shift_norm": False,
    #     "raw_unet": True,
    #     "noise_schedule": "squaredcos_cap_v2",
    #     "rescale_learned_sigmas": False,
    #     "rescale_timesteps": False,
    #     "num_classes": '2',
    #     "dataset": "clevr_pos",
    #     "use_fp16": has_cuda,
    #     "timestep_respacing": str(timestep_respacing)
    # }
    #
    # for key, val in flags.items():
    #     defaults[key] = val

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
