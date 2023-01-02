import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from composable_diffusion import dist_util, logger
from composable_diffusion.fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from composable_diffusion.logger import plot_image_with_label
from composable_diffusion.nn import update_ema
from composable_diffusion.resample import LossAwareSampler, UniformSampler
import time
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
# 20 - imageNet
INITIAL_LOG_LOSS_SCALE = 20
import torch

class TrainLoop:
    def __init__(
            self,
            *,
            model,
            querier_model,
            diffusion,
            dataset,
            data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
    ):
        self.dataset = dataset

        self.model = model
        self.querier = querier_model
        # print(self.querier)
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size #* dist.get_world_size()

        # list(self.model.parameters()) +
        self.model_params = list(self.querier.parameters()) + list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            # print(dist.get_world_size(), th.cuda.device_count())
            self.use_ddp = True
            # self.ddp_model = DDP(
            #     self.model,
            #     device_ids=[dist_util.dev()],
            #     output_device=dist_util.dev(),
            #     broadcast_buffers=False,
            #     bucket_cap_mb=128,
            # )
            self.ddp_model = self.model
            self.ddp_querier = self.querier
            # self.ddp_querier = DDP(
            #     self.querier,
            #     device_ids=[dist_util.dev()],
            #     output_device=dist_util.dev(),
            #     broadcast_buffers=False,
            #     bucket_cap_mb=128,
            # )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
            self.ddp_querier = self.querier

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        # TODO: do the same for querier later
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        # dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        # dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()
        self.querier.convert_to_fp16() # Note: Added. Not clear if it works

    def run_loop(self):
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_loop_vip(self):
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)

            # TODO: Generate random set of conditions.
            cond_set = [cond]
            img, pred_y, true_y = self.run_step_vip(batch, cond_set)
            if self.step % self.log_interval == 0:
                plot_image_with_label(list(img), list(true_y), save=True, name='sample_gt')
                plot_image_with_label(list(img), list(pred_y), save=True, name='sample_pred')
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def run_step_vip(self, batch, cond_set):
        img, pred_y, true_y = self.forward_backward_vip(batch, cond_set)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()
        return img, pred_y, true_y

    def _get_tokens_masks(self, captions, p=0.05):
        # provide a way to mask out some captions under a threshold of p
        tokens_list = [self.model.tokenizer.encode(prompt) for prompt in captions]
        # within some probability, no captions are provided
        outputs = [self.model.tokenizer.padded_tokens_and_mask(
            tokens if np.random.rand() > p else [], self.model.text_ctx
        ) for tokens in tokens_list]
        tokens, masks = zip(*outputs)
        return dict(tokens=th.tensor(tokens), mask=th.tensor(masks))

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i:i + self.microbatch].to(dist_util.dev())
            # cond contains captions
            micro_cond = dict()
            if 'caption' in cond:
                micro_cond.update({
                    k: v[i:i + self.microbatch].to(dist_util.dev())
                    for k, v in self._get_tokens_masks(cond['caption']).items()
                })

            if 'y' in cond:
                if self.dataset == 'clevr_pos' or self.dataset == 'clevr_pos_multiple':
                    dtype = th.float
                elif self.dataset == 'clevr_rel':
                    dtype = th.long
                else:
                    raise NotImplementedError()

                micro_cond['y'] = cond['y'][i:i + self.microbatch].type(dtype).to(dist_util.dev())
                micro_cond['masks'] = cond['masks'][i:i + self.microbatch].to(dist_util.dev())

            others = {k: v[i:i + self.microbatch].to(dist_util.dev())
                      for k, v in cond.items() if k not in ['caption', 'y']}

            micro_cond.update(others)
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            x_t = None
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                x_t=x_t,
                model_kwargs=micro_cond,
            )
            # print(compute_losses)
            # exit()
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def sample_pos(self, cond, data, diffusion, model, x_t, t, noise_t, weights, num_steps, step_lr=0.1):


        noise = torch.randn_like(cond['y']).detach()
        # condy = torch.clamp(torch.rand_like(cond['y']).requires_grad_(requires_grad=True), 0.2, 0.8)
        condy = (torch.ones_like(cond['y']) * 0.5).requires_grad_(requires_grad=True)
        cond = {'y': condy, 'masks': cond['masks']}

        t_0 = time.time()
        for i in range(num_steps):
            noise.normal_()

            compute_losses = functools.partial(
                diffusion,
                model,
                data,
                t,
                x_t=x_t,
                model_kwargs=cond,
                noise=noise_t,
            )

            energy = compute_losses()['loss'] * weights
            print(energy.mean().item(), '\n', cond['y'][:2].detach().numpy())
            # Get grad for current optimization iteration.
            grad, = torch.autograd.grad([energy.sum()], [condy], create_graph=False)

            # grad = torch.clamp(grad, min=-0.5, max=0.5) # TODO: Remove if useless
            print(condy.abs().mean())
            condy = condy - step_lr * grad + 0.005 * noise # GD computation
            print(grad.abs().mean())
            condy = torch.clamp(condy, 0, 1) # TODO: put back on

            condy = condy.detach()
            cond['y'] = condy.requires_grad_()  # Q: Why detaching and reataching? Is this to avoid backprop through this step?
            print(time.time() - t_0)

        return cond, energy

    def forward_backward_vip(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i:i + self.microbatch].to(dist_util.dev())
            # cond contains captions
            micro_cond = dict()
            # if cond is not None:
            if isinstance(cond, list):
                cond = cond[0]
            if 'caption' in cond:
                micro_cond.update({
                    k: v[i:i + self.microbatch].to(dist_util.dev())
                    for k, v in self._get_tokens_masks(cond['caption']).items()
                })

            if 'y' in cond:
                if self.dataset == 'clevr_pos' or self.dataset == 'clevr_pos_multiple':
                    dtype = th.float

                elif self.dataset == 'clevr_rel':
                    dtype = th.long
                else:
                    raise NotImplementedError()

                micro_cond['y'] = cond['y'][i:i + self.microbatch].type(dtype).to(dist_util.dev())
                micro_cond['masks'] = cond['masks'][i:i + self.microbatch].to(dist_util.dev())

            others = {k: v[i:i + self.microbatch].to(dist_util.dev())
                      for k, v in cond.items() if k not in ['caption', 'y']}

            micro_cond.update(others)
            micro_conds = [micro_cond]
            last_batch = (i + self.microbatch) >= batch.shape[0]
            true_label = micro_cond['y']
            t_0 = time.time()
            noise = torch.randn_like(micro)
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            x_t = None
            # t = torch.ones_like(t) * 200 # TODO: test at noise level 200
            true_tensor = th.tensor([True]*micro.shape[0], dtype=th.bool, device=micro.device)
            outs = []
            for cid, micro_cond_i in enumerate(micro_conds):
                query_out = self.ddp_querier(micro)
                micro_cond_i['y'] = query_out
                micro_cond_i['masks'] = true_tensor
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    x_t = x_t,
                    noise=noise,
                    model_kwargs=micro_cond_i,
                )
                out_dict = compute_losses()
                if cid == 0:
                    x_t = out_dict["x_t"]
                    target = out_dict["target"]
                output = out_dict["model_output"]
                outs.append(output)

            outs = torch.stack(outs, dim=1)
            # exit()
            def mean_flat(tensor):
                return tensor.mean(dim=list(range(1, len(tensor.shape))))

            mse_loss = mean_flat((target[:, None] - outs) ** 2)

            # energy = out_dict['mse'] * weights
            # print(energy.mean().item(), '\n', micro_cond['y'][:2].detach().numpy())
            # print(time.time() - t_0)

            # cond, energy = self.sample_pos(micro_cond, micro, self.diffusion.training_losses, self.ddp_model, out_dict['x_t'], t, noise, weights,
            #                                num_steps=20, step_lr=2e3)

            # exit()

            if last_batch or not self.use_ddp:
                # Enters here
                losses = out_dict #compute_losses()
            else:

                with self.ddp_model.no_sync():
                    losses = out_dict #compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                # Enters here

                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            # loss = (losses["loss"] * weights).mean()
            loss = mse_loss.mean()
            losses['mse_loss'] = mse_loss
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items() if v.shape == weights.shape}
                # self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:

                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()
        return micro, query_out, true_label # TODO: This is the last of both, should be more systematic
    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            # if p.grad is not None:
            if p.grad is None:
                pass
            else:
                sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            # if dist.get_rank() == 0: # TODO: Modified
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step + self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # if dist.get_rank() == 0: # TODO: Modified
        with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"),
                "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

        # dist.barrier() # TODO: Modified

    # def _master_params_to_state_dict(self, master_params):
    #     if self.use_fp16:
    #         master_params = unflatten_master_params(
    #             self.model.parameters(), master_params
    #         )
    #     state_dict = self.model.state_dict()
    #     for i, (name, _value) in enumerate(self.model.named_parameters()):
    #         assert name in state_dict
    #         state_dict[name] = master_params[i]
    #     return state_dict
    #
    # def _state_dict_to_master_params(self, state_dict):
    #     params = [state_dict[name] for name, _ in self.model.named_parameters()]
    #     if self.use_fp16:
    #         return make_master_params(params)
    #     else:
    #         return params
    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.querier.parameters(), master_params
            )
        state_dict = self.querier.state_dict()
        for i, (name, _value) in enumerate(self.querier.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.querier.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)