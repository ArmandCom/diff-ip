
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from flamingo_pytorch import PerceiverResampler, GatedCrossAttentionBlock
from arch.models import PositionalEncoding2D

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(bs, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t=None):
        x = self.maxpool_conv(x)
        if t is not None:
            emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
            x = x + emb
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t=None):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        if t is not None:
            emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
            x = x + emb
        return x

class Down_uc(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x

class Up_uc(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        return x

class SAUNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, size=28, patch_size=5, out_size=None, device="cuda"):
        super().__init__()
        self.device = device
        position_enc = True
        if out_size is None:
            out_size = size - patch_size + 1
        if position_enc:
            d_pe = 4
            if not isinstance(out_size, tuple):
                d_spa = (size, size)
            else: d_spa = size
            self.pos_encoder = PositionalEncoding2D(d_model=d_pe, d_spatial=d_spa, dropout=0.0)
            c_in += d_pe

        # c_in += 1

        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down_uc(64, 128)
        size = size//2
        # self.sa1 = SelfAttention(128, size)
        self.down2 = Down_uc(128, 128)
        size = size//2
        # self.sa2 = SelfAttention(128, size)
        self.down3 = Down_uc(128, 256)
        size = size//2
        self.sa3 = SelfAttention(256, size)

        self.bot1 = DoubleConv(256, 256)
        self.bot2 = DoubleConv(256, 256)
        self.bot3 = DoubleConv(256, 128)

        self.up1 = Up_uc(256, 128)
        size = size*2
        self.sa4 = SelfAttention(128, size)
        self.up2 = Up_uc(256, 64)
        size = size*2
        # self.sa5 = SelfAttention(64, size)
        self.up3 = Up_uc(128, 64)
        size = size*2
        # self.sa6 = SelfAttention(64, size)
        self.upout = nn.Upsample(size=out_size, mode='bilinear')
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)



        self.softmax = nn.Softmax(-1)

    def intp(self, x, ref, mode='bilinear'):
        return F.interpolate(x, ref.shape[-1], mode=mode, align_corners=True)
        # Mask layers
    def forward(self, x, mask, return_attn=False):
        # h = w = int(math.sqrt(mask.shape[-1]))
        # mask_img = mask.reshape(x.shape[0], 1, w, h)
        # nmask = 1 - mask_img

        x = self.pos_encoder(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        # x2 = x2 * self.intp(nmask, x2)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2)
        # x3 = x3 * self.intp(nmask, x3)
        # x3 = self.sa2(x3)
        x4 = self.down3(x3)
        # x4 = x4 * self.intp(nmask, x4)
        x4 = self.sa3(x4)
        x = x4

        x4 = self.bot1(x)
        x4 = self.bot2(x4)
        x = self.bot3(x4)

        x = self.up1(x, x3)
        # x = x * self.intp(nmask, x)
        x = self.sa4(x)
        x = self.up2(x, x2)
        # x = x * self.intp(nmask, x)
        # x = self.sa5(x)
        x = self.up3(x, x1)
        # x = x * self.intp(nmask, x)
        # x = self.sa6(x)
        x = self.upout(x)
        x = self.outc(x)

        query_logits_pre = x.view(x.shape[0], -1)
        query_mask = torch.where(mask == 1, query_logits_pre.min().detach(), torch.zeros((1,)).to(x.device)) # TODO: Check why.
        query_logits = query_logits_pre + query_mask #.to(x.device)

        # straight through softmax
        query = self.softmax(query_logits / self.tau)
        _, max_ind = (query).max(1)
        query_onehot = F.one_hot(max_ind, query.shape[1]).type(query.dtype)
        query_out = (query_onehot - query).detach() + query

        if return_attn:
            # TODO: Check if this being soft is essential.
            # query = self.dropout(query)
            query_logits_pre = query_logits_pre - torch.min(query_logits_pre, dim=1, keepdim=True)[0]
            query_logits_pre = query_logits_pre / torch.max(query_logits_pre, dim=1, keepdim=True)[0]
            return query_out, query_logits_pre
        else: query_out = query
        return query_out

# class SAUNet(nn.Module):
#     def __init__(self, c_in=1, c_out=1, size=28, patch_size=5, out_size=None, device="cuda"):
#         super().__init__()
#         self.device = device
#         if out_size is None:
#             out_size = size - patch_size + 1
#
#         # c_in += 1
#
#         self.inc = DoubleConv(c_in, 64)
#         self.down1 = Down_uc(64, 128)
#         size = size//2
#         self.sa1 = SelfAttention(128, size)
#         # self.down2 = Down_uc(128, 128) #
#         # size = size//2 #
#         # # self.sa2 = SelfAttention(128, size)
#         # self.down3 = Down_uc(128, 256) #
#         # size = size//2 #
#         # self.sa3 = SelfAttention(256, size) #
#
#         self.bot1 = DoubleConv(128, 256)
#         self.bot2 = DoubleConv(256, 256)
#         self.bot3 = DoubleConv(256, 128)
#
#         # self.up1 = Up_uc(256, 128) #
#         # size = size*2 #
#         # self.sa4 = SelfAttention(128, size) #
#         # self.up2 = Up_uc(256, 64) #
#         # size = size*2 #
#         # self.sa5 = SelfAttention(128, size)
#         self.up3 = Up_uc(192, 64)
#         size = size*2
#         # self.sa6 = SelfAttention(64, size)
#         self.upout = nn.Upsample(size=out_size, mode='bilinear')
#         self.outc = nn.Conv2d(64, c_out, kernel_size=1)
#
#         self.softmax = nn.Softmax(-1)
#
#     def intp(self, x, ref, mode='bilinear'):
#         return F.interpolate(x, ref.shape[-1], mode=mode, align_corners=True)
#         # Mask layers
#     def forward(self, x, mask, return_attn=False):
#         # h = w = int(math.sqrt(mask.shape[-1]))
#         # mask_img = mask.reshape(x.shape[0], 1, w, h)
#         # nmask = 1 - mask_img
#
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         # x2 = x2 * self.intp(nmask, x2)
#         # x2 = self.sa1(x2)
#         # x3 = self.down2(x2) #
#         # x3 = x3 * self.intp(nmask, x3)
#         # x3 = self.sa2(x3)
#         # x4 = self.down3(x3) #
#         # x4 = x4 * self.intp(nmask, x4)
#         # x4 = self.sa3(x4) #
#         x = x2 #x4 #
#
#         x4 = self.bot1(x)
#         x4 = self.bot2(x4)
#         x = self.bot3(x4)
#
#         # x = self.up1(x, x3) #
#         # x = x * self.intp(nmask, x)
#         # x = self.sa4(x) #
#         # x = self.up2(x, x2) #
#         # x = x * self.intp(nmask, x)
#         # x = self.sa5(x) #
#         x = self.up3(x, x1)
#         # x = x * self.intp(nmask, x)
#         # x = self.sa6(x)
#         x = self.upout(x)
#         x = self.outc(x)
#
#         query_logits_pre = x.view(x.shape[0], -1)
#         query_mask = torch.where(mask == 1, query_logits_pre.min().detach(), torch.zeros((1,)).to(x.device)) # TODO: Check why.
#         query_logits = query_logits_pre + query_mask #.to(x.device)
#
#         # straight through softmax
#         query = self.softmax(query_logits / self.tau)
#         _, max_ind = (query).max(1)
#         query_onehot = F.one_hot(max_ind, query.shape[1]).type(query.dtype)
#         query_out = (query_onehot - query).detach() + query
#
#         if return_attn:
#             # TODO: Check if this being soft is essential.
#             # query = self.dropout(query)
#             query_logits_pre = query_logits_pre - torch.min(query_logits_pre, dim=1, keepdim=True)[0]
#             query_logits_pre = query_logits_pre / torch.max(query_logits_pre, dim=1, keepdim=True)[0]
#             return query_out, query_logits_pre
#         else: query_out = query
#         return query_out