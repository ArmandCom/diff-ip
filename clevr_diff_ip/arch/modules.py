
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
        if not isinstance(size, tuple):
            self.size = (size, size)
        else: self.size = size
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
        # Note: Changed self.size to admit tuples
        x = x.view(bs, self.channels, self.size[0] * self.size[1]).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(bs, self.channels, *self.size)


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

class DoubleConv_wK(nn.Module):
    def __init__(self, in_channels, out_channels, kernel = (3, 3), padding = (1, 1), mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, padding=padding, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel, padding=padding, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_linear = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.LayerNorm(mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, out_channels, bias=False),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # if self.residual:
        #     return F.gelu(x + self.double_linear(x))
        # else:
        return self.double_linear(x)

class MLPLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_linear = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.LayerNorm(mid_channels),
            nn.LeakyReLU(negative_slope=0.3), #nn.ReLU(),
            nn.Linear(mid_channels, out_channels, bias=False),
            nn.LayerNorm(out_channels),
            nn.LeakyReLU(negative_slope=0.3), #nn.ReLU(),
        )

    def forward(self, x):
        # if self.residual:
        #     return F.gelu(x + self.double_linear(x))
        # else:
        return self.double_linear(x)
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

class QuerierImageAttr(nn.Module):
    def __init__(self, embed_dim=5, num_obj=5, num_attr=5, image_size=128, hidden_dim=256, encode_image=True, use_latent=True, use_answers=False, pos_enc_fn = None):
        super().__init__()
        # self.device = device
        #TODO: pass positional encoding as input with Imagenet function. LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        position_enc = True
        # self.latent
        self.tau = 1
        bottleneck_dim = 0
        hidden_dim = embed_dim

        c_in = 3 #embed_dim * 2  # (pos and neg)
        self.embeds_dim, self.num_obj, self.num_attr = embed_dim, num_obj, num_attr
        if position_enc and encode_image:
            d_pe = 4
            if not isinstance(image_size, tuple):
                d_spa = (image_size, image_size)
            else: d_spa = image_size
            # d_spa = (num_obj, num_attr)
            self.pos_encoder_im = PositionalEncoding2D(d_model=d_pe, d_spatial=d_spa, dropout=0.0)
            c_in += d_pe

        if position_enc:
            d_pe_q = 4
            d_spa_q = (num_attr, num_obj)
            self.pos_encoder_q = PositionalEncoding2D(d_model=d_pe_q, d_spatial=d_spa_q, dropout=0.0)
            bottleneck_dim += d_pe_q


        ## Image Encoder
        if encode_image:
            self.inc = DoubleConv(c_in, 64)
            self.down1 = Down_uc(64, 128)
            size = image_size//2
            # self.sa1 = SelfAttention(128, size)
            self.down2 = Down_uc(128, 128)
            size = size//2
            # self.sa2 = SelfAttention(128, size)
            self.down3 = Down_uc(128, 256)
            size = size//2
            self.sa3 = SelfAttention(256, size)
            self.bot1 = DoubleConv(256, 256)
            self.bot2 = DoubleConv(256, hidden_dim * num_obj * num_attr)
            self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

            self.image_encoder = nn.Sequential(
                self.inc,
                self.down1,
                self.down2,
                self.down3,
                self.sa3,
                self.bot1,
                self.bot2,
                self.max_pool
            )
            bottleneck_dim += hidden_dim

            # create a parameter
            self.alpha = nn.Parameter(torch.zeros(1,))
        self.feat_dim = hidden_dim
        self.encode_image = encode_image

        # Query encoder
        bottleneck_dim += embed_dim
        self.inc = DoubleConv(bottleneck_dim, hidden_dim)
        # self.conv1d = nn.Conv2d(bottleneck_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.sa4 = SelfAttention(hidden_dim, (num_attr, num_obj))
        self.sa5 = SelfAttention(hidden_dim, (num_attr, num_obj))
        self.conv_end = nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1, padding=0)
        self.query_encoder = nn.Sequential(
            self.inc,
            self.sa4,
            self.sa5,
            self.conv_end,
        )

        self.softmax = nn.Softmax(-1)

    def intp(self, x, ref, mode='bilinear'):
        return F.interpolate(x, ref.shape[-1], mode=mode, align_corners=True)
        # Mask layers

    def forward(self, cond, ans=None, image=None, mask=None, return_attn=False):
        x = cond
        x = x.reshape(x.shape[0], self.num_attr, self.num_obj, self.embeds_dim, 3).sum(-1)
        x = x.permute(0, 3, 1, 2)

        if self.encode_image:
            # TODO: Just do resnet with classification head, don't reinvent the wheel
            image = self.pos_encoder_im(image)
            h = self.image_encoder(image)
            h = h.reshape(-1, self.feat_dim, self.num_attr, self.num_obj)
            # x = x + self.alpha * h
            x = torch.cat([x, h], dim=1)

        x = self.pos_encoder_q(x)
        x = self.query_encoder(x)

        # remove elements in mask
        query_logits_pre = x.view(x.shape[0], -1)
        query_mask = torch.where(mask == 1, -1e8, torch.zeros((1,)).to(x.device))
        query_logits = query_logits_pre + query_mask  # .to(x.device)

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
            return query_out.reshape(-1, self.num_attr * self.num_obj, 1), \
                query_logits_pre
        # else: query_out = query
        return query_out.reshape(-1, self.num_attr * self.num_obj, 1)
class QuerierConvImageAttr(nn.Module):
    def __init__(self, embed_dim=5, num_obj=5, num_attr=5, image_size=128, hidden_dim=256, encode_image=True, use_latent=True, use_answers=False, pos_enc_fn = None):
        super().__init__()
        # self.device = device
        #TODO: pass positional encoding as input with Imagenet function. LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        position_enc = False
        self.position_enc = position_enc
        # self.latent
        self.tau = 1
        bottleneck_dim = 0
        hidden_dim = embed_dim

        c_in = 3 #embed_dim * 2  # (pos and neg)
        self.embeds_dim, self.num_obj, self.num_attr = embed_dim, num_obj, num_attr
        if position_enc and encode_image:
            d_pe = 4
            if not isinstance(image_size, tuple):
                d_spa = (image_size, image_size)
            else: d_spa = image_size
            # d_spa = (num_obj, num_attr)
            self.pos_encoder_im = PositionalEncoding2D(d_model=d_pe, d_spatial=d_spa, dropout=0.0)
            c_in += d_pe

        if position_enc:
            d_pe_q = 4
            d_spa_q = (num_attr, num_obj)
            self.pos_encoder_q = PositionalEncoding2D(d_model=d_pe_q, d_spatial=d_spa_q, dropout=0.0)
            bottleneck_dim += d_pe_q


        ## Image Encoder
        # if encode_image:
            # TODO: resnet
            # c_in --> hidden_dim
            # torchvision.models.resnet34
        self.feat_dim = hidden_dim
        self.encode_image = encode_image

        # Query encoder
        bottleneck_dim += embed_dim
        self.inc = DoubleConv_wK(bottleneck_dim, hidden_dim,
                                 kernel=(1, 5), padding=(0, 2), residual=True)
        self.conv_1 = DoubleConv_wK(hidden_dim, hidden_dim,
                                 kernel=(1, 3), padding=(0, 1))
        self.conv_2 = DoubleConv_wK(hidden_dim, hidden_dim,
                                 kernel=(1, 3), padding=(0, 1))
        self.conv_3 = DoubleConv_wK(hidden_dim, hidden_dim,
                                 kernel=(5, 1), padding=(2, 0))
        self.conv_4 = DoubleConv_wK(hidden_dim, hidden_dim,
                                 kernel=(5, 1), padding=(2, 0))
        self.conv_5 = DoubleConv_wK(hidden_dim, hidden_dim,
                                 kernel=(5, 1), padding=(2, 0))
        self.conv_out = nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1, padding=0)

        self.query_encoder = nn.Sequential(
            self.inc,
            self.conv_1,
            self.conv_2,
            self.conv_3,
            self.conv_4,
            self.conv_5,
            self.conv_out,
        )

        self.softmax = nn.Softmax(-1)

    def intp(self, x, ref, mode='bilinear'):
        return F.interpolate(x, ref.shape[-1], mode=mode, align_corners=True)
        # Mask layers

    def forward(self, cond, ans=None, image=None,
                image_features=None,
                query_features=None,mask=None, return_attn=False):
        x = cond
        x = x.reshape(x.shape[0], self.num_attr, self.num_obj, self.embeds_dim, 3).sum(-1)
        x = x.permute(0, 3, 1, 2)

        # if self.encode_image:
        #     # TODO: Just do resnet with classification head, don't reinvent the wheel
        #     image = self.pos_encoder_im(image)
        #     h = self.image_encoder(image)
        #     h = h.reshape(-1, self.feat_dim, self.num_attr, self.num_obj)
        #     # x = x + self.alpha * h
        #     x = torch.cat([x, h], dim=1)

        if self.position_enc:
            x = self.pos_encoder_q(x)
        x = self.query_encoder(x)

        # remove elements in mask
        query_logits_pre = x.view(x.shape[0], -1)
        query_mask = torch.where(mask == 1, -1e8, torch.zeros((1,)).to(x.device))
        query_logits = query_logits_pre + query_mask  # .to(x.device)

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
            return query_out.reshape(-1, self.num_attr * self.num_obj, 1), \
                query_logits_pre
        # else: query_out = query
        return query_out.reshape(-1, self.num_attr * self.num_obj, 1)

class SqueezeExciteLayer(nn.Module):
    """Squeeze and Exite layer from https://arxiv.org/abs/1709.01507."""
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.dense_1 = nn.Linear(num_channels, num_channels // reduction, bias=False)
        self.dense_2 = nn.Linear(num_channels // reduction, num_channels, bias=False)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        num_channels = x.shape[-3]
        y = x.reshape(x.shape[0], num_channels, -1).mean(-1)
        y = self.dense_1(y)
        y = self.act(y)
        y = self.dense_2(y)
        y = self.sigmoid(y)
        return x * y[:, :, None, None]

class Scorer(nn.Module):
    """Scorer function."""
    def __init__(self, c_in, c_out=1, size=60, out_size=None, patch_size=5, reduction=16):
        super().__init__()
        if out_size is None:
            out_size = size - patch_size + 1
        self.act = nn.ReLU() # Check for leaky
        # self.conv1 = nn.Conv2d(c_in, 1, kernel_size=(3, 3), padding=(1, 1))
        self.conv1 = nn.Conv2d(c_in, 8, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=(1, 1))
        self.sae2 = SqueezeExciteLayer(16, reduction=reduction)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.sae4 = SqueezeExciteLayer(64, reduction=reduction)
        self.conv_out = nn.Conv2d(64, c_out, kernel_size=(3, 3), padding=(1, 1))
        self.upout = nn.Upsample(size=out_size, mode='nearest')
        # self.maxpool = nn.MaxPool2d(kernel_size=(8, 8), stride=(8, 8))
        # self.conv_out.weight.data.fill_(0.01)

        self.softmax = nn.Softmax(-1)
        self.tau=1
    def encode(self, x, use_squeeze_excite=False):
        # x = self.act(self.conv1(x))
        x = self.conv1(x)
        x = self.act(self.conv2(x))
        if use_squeeze_excite:
            x = self.sae2(x)
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        if use_squeeze_excite:
            x = self.sae4(x)
        x = self.conv_out(x)
        x = self.upout(x)
        # x = self.maxpool(x)
        return x
    def forward(self, x, mask, return_attn=False):
        x = self.encode(x, use_squeeze_excite=False)

        query_logits_pre = x.view(x.shape[0], -1)
        n_q = query_logits_pre.shape[-1]

        query_mask = torch.where(mask == 1, -1e9, torch.zeros((1,)).to(x.device))  # TODO: Check why.
        # identity_mask = torch.zeros_like(mask).reshape(*x.shape)
        # identity_mask[..., x.shape[-2] // 2, x.shape[-1] // 2] = 1
        # query_mask += identity_mask.reshape(*query_mask.shape)

        query_logits = query_logits_pre + query_mask  # .to(x.device)

        # straight through softmax
        
        query = self.softmax(query_logits / (n_q * self.tau))
        query_out = F.gumbel_softmax(query_logits, tau=self.tau, hard=True)
        # _, max_ind = (query).max(1)
        # query_onehot = F.one_hot(max_ind, query.shape[1]).type(query.dtype)
        # query_out = (query_onehot - query).detach() + query

        if return_attn:
            # TODO: Check if this being soft is essential.
            # query = self.dropout(query)
            query_logits_pre = query_logits_pre - torch.min(query_logits_pre, dim=1, keepdim=True)[0]
            query_logits_pre = query_logits_pre / torch.max(query_logits_pre, dim=1, keepdim=True)[0]
            return query_out, query, query_logits_pre
        # else: query_out = query
        # print('Query_out is set to query!! Should be query_onehot')
        # print('We changed it to query_onehot')
        return query_out, query

    # @classmethod
    # def compute_output_size(cls, height, width):
    #     return ((height - 8) // 8, (width - 8) // 8)
class QuerierLinearImageAttr(nn.Module):
    def __init__(self, embed_dim=5, num_obj=5, num_attr=5, image_size=128, latent_dim=64, hidden_dim=128, encode_image=True,
                 use_image_features=False,
                 use_query_features=False,
                 add_object_embedding=True, use_latent=True, use_answers=False, cond_dim=256, pos_enc_fn=None):
        super().__init__()
        # TODO: pass positional encoding as input with Imagenet function. LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        position_enc = True
        self.position_enc = position_enc
        # self.latent
        self.tau = 1
        self.use_answers = use_answers
        self.only_image = False
        # hidden_dim = embed_dim
        embed_dim = 1 if use_answers else embed_dim
        embed_dim = cond_dim if use_query_features else embed_dim
        c_in = 3
        bottleneck_dim_conv = embed_dim
        bottleneck_dim = embed_dim * num_obj * num_attr
        if self.only_image:
            assert (use_image_features or encode_image)
            bottleneck_dim_conv = 0
            bottleneck_dim = 0

        self.embeds_dim, self.num_obj, self.num_attr = embed_dim, num_obj, num_attr
        if position_enc and encode_image:
            d_pe = 4
            if not isinstance(image_size, tuple):
                d_spa = (image_size, image_size)
            else:
                d_spa = image_size
            # d_spa = (num_obj, num_attr)
            self.pos_encoder_im = PositionalEncoding2D(d_model=d_pe, d_spatial=d_spa, dropout=0.0)
            c_in += d_pe

        if encode_image and not use_image_features:
            inc = DoubleConv(c_in, 64)
            down1 = Down_uc(64, 128)
            size = image_size // 2
            # self.sa1 = SelfAttention(128, size)
            down2 = Down_uc(128, 128)
            size = size // 2
            # self.sa2 = SelfAttention(128, size)
            down3 = Down_uc(128, 256)
            size = size // 2
            sa3 = SelfAttention(256, size)
            bot1 = DoubleConv(256, 256)
            bot2 = DoubleConv(256, hidden_dim * num_obj * num_attr)
            max_pool = nn.AdaptiveMaxPool2d((1, 1))

            self.image_encoder = nn.Sequential(
                inc,
                down1,
                down2,
                down3,
                sa3,
                bot1,
                bot2,
                max_pool
            )
            bottleneck_dim += hidden_dim * num_obj * num_attr
            bottleneck_dim_conv += hidden_dim
            ## Image Encoder
            # if encode_image:
            # TODO: resnet
            # c_in --> hidden_dim
            # torchvision.models.resnet34
            self.feat_dim = hidden_dim

        elif use_image_features:
            inc = DoubleConv(128, 128)
            down1 = Down_uc(128, hidden_dim * num_obj * num_attr)
            max_pool = nn.AdaptiveAvgPool2d((1, 1)) # Note: Average instead of max
            self.image_feature_encoder = nn.Sequential(
                inc,
                down1,
                max_pool
            )
            bottleneck_dim += hidden_dim * num_obj * num_attr
            bottleneck_dim_conv += hidden_dim
            self.feat_dim = hidden_dim
        self.encode_image = encode_image
        self.use_image_features = use_image_features
        self.use_query_features = use_query_features

        # # Query encoder
        # #TODO: add residual connections
        # fcd1 = MLP(bottleneck_dim, hidden_dim, hidden_dim*2)
        # fcd2 = MLP(hidden_dim, latent_dim, hidden_dim)
        # fcu1 = MLP(latent_dim, hidden_dim)
        # # fcu2 = MLP(hidden_dim, hidden_dim)
        # linear_out = nn.Linear(hidden_dim, num_obj * num_attr)
        #
        # self.query_encoder_ = nn.Sequential(
        #     fcd1,
        #     fcd2,
        #     fcu1,
        #     # fcu2,
        #     linear_out,
        # )

        conv_layers = nn.Sequential(
                nn.Conv2d(bottleneck_dim_conv, hidden_dim, kernel_size=5, padding=2, bias=True),
                nn.GroupNorm(2, hidden_dim),
                nn.LeakyReLU(0.3),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, bias=True),
                nn.GroupNorm(2, hidden_dim),
                nn.LeakyReLU(0.3),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, bias=True),
                nn.GroupNorm(2, hidden_dim),
                nn.LeakyReLU(0.3),
                nn.Conv2d(hidden_dim, 1, kernel_size=5, padding=2, bias=True),
                # nn.GroupNorm(1, hidden_dim),
                # nn.ReLU(),
        )
        self.query_encoder_ = conv_layers

        self.softmax = nn.Softmax(-1)

    def intp(self, x, ref, mode='bilinear'):
        return F.interpolate(x, ref.shape[-1], mode=mode, align_corners=True)
        # Mask layers

    def forward(self, cond, ans=None, image=None,
                image_features=None,
                query_features=None, mask=None, return_attn=False, cond_all=None):
        # (self, cond, ans=None, image=None, mask=None, return_attn=False):
        if not self.use_answers and not self.use_query_features:
            x = cond
            x = x.reshape(x.shape[0], self.num_attr, self.num_obj, self.embeds_dim, x.shape[-1]).sum(-1)
            x = x.permute(0, 3, 1, 2)
        elif self.use_answers and not self.use_query_features:
            x = ans
            x = x.reshape(x.shape[0], self.num_attr, self.num_obj, self.embeds_dim, x.shape[-1])
            x = x[..., 0].float() - x[..., 1].float()
            x = x.permute(0, 3, 1, 2)
        elif self.use_query_features:
            x = query_features.reshape(query_features.shape[0], self.num_attr, self.num_obj, self.embeds_dim)
            x = x.permute(0, 3, 1, 2)

        if self.encode_image and not self.use_image_features:
            # TODO: Just do resnet with classification head, don't reinvent the wheel
            image = self.pos_encoder_im(image)
            h = self.image_encoder(image)
            h = h.reshape(-1, self.feat_dim, self.num_attr, self.num_obj)
            # x = x + self.alpha * h
            if self.only_image:
                x = h
            else:
                x = torch.cat([x, h], dim=1)
        elif self.use_image_features:
            h = self.image_feature_encoder(image_features)
            h = h.reshape(-1, self.feat_dim, self.num_attr, self.num_obj)
            if self.only_image:
                x = h
            else:
                x = torch.cat([x, h], dim=1)

        # if self.position_enc:
        #     x = self.pos_encoder_q(x)
        # x = x.reshape(x.shape[0], -1)
        if not self.only_image:
            x = self.query_encoder_(x)

        # remove elements in mask
        query_logits_pre = x.view(x.shape[0], -1)
        query_mask = torch.where(mask == 1, -1e8, torch.zeros((1,)).to(x.device))
        query_logits = query_logits_pre + query_mask  # .to(x.device)

        # straight through softmax
        query = self.softmax(query_logits / self.tau)

        _, max_ind = (query).max(1)
        query_onehot = F.one_hot(max_ind, query.shape[1]).type(query.dtype)
        query_out = (query_onehot - query).detach() + query


        if return_attn:
            # query = self.dropout(query)
            query_logits_pre = query_logits_pre - torch.min(query_logits_pre, dim=1, keepdim=True)[0]
            query_logits_pre = query_logits_pre / torch.max(query_logits_pre, dim=1, keepdim=True)[0]
            return query_out.reshape(-1, self.num_attr * self.num_obj, 1), \
                query.reshape(-1, self.num_attr * self.num_obj, 1), \
                query_logits_pre
        # else: query_out = query
        return query_out.reshape(-1, self.num_attr * self.num_obj, 1), \
            query.reshape(-1, self.num_attr * self.num_obj, 1)

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce
from einops_exts import rearrange_many, repeat_many, check_shape
from torch import nn, einsum
def exists(val):
    return val is not None
class QuerierAttn(nn.Module):
    def __init__(self, embed_dim=5, num_obj=5, num_attr=5, image_size=128, latent_dim=64, hidden_dim=128, encode_image=True,
                 add_object_embedding=True, use_latent=True, use_answers=False, pos_enc_fn=None):
        super().__init__()
        # TODO: pass positional encoding as input with Imagenet function. LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        position_enc = True
        self.position_enc = position_enc
        # self.latent
        self.tau = 1
        self.use_answers = use_answers
        bottleneck_dim = 0
        # hidden_dim = embed_dim
        # embed_dim = 1 if use_answers else embed_dim
        c_in = 3
        bottleneck_dim_conv = 2 * embed_dim + 1 if self.use_answers else 4 * embed_dim # 2*embed because we have an object embedding too.
        bottleneck_dim = (2 * embed_dim + 1) * num_obj * num_attr
        # S_c_in = embed_dim * num_obj * num_attr
        self.embeds_dim, self.num_obj, self.num_attr = embed_dim, num_obj, num_attr
        if position_enc and encode_image:
            d_pe = 4
            if not isinstance(image_size, tuple):
                d_spa = (image_size, image_size)
            else:
                d_spa = image_size
            # d_spa = (num_obj, num_attr)
            self.pos_encoder_im = PositionalEncoding2D(d_model=d_pe, d_spatial=d_spa, dropout=0.0)
            c_in += d_pe

        if encode_image:
            inc = DoubleConv(c_in, 64)
            down1 = Down_uc(64, 128)
            size = image_size // 2
            # self.sa1 = SelfAttention(128, size)
            down2 = Down_uc(128, 128)
            size = size // 2
            # self.sa2 = SelfAttention(128, size)
            down3 = Down_uc(128, 256)
            size = size // 2
            # sa3 = SelfAttention(256, size)
            # bot1 = DoubleConv(256, 256)
            bot2 = DoubleConv(256, hidden_dim * num_obj * num_attr)
            max_pool = nn.AdaptiveMaxPool2d((1, 1))

            self.image_encoder = nn.Sequential(
                inc,
                down1,
                down2,
                down3,
                # sa3,
                # bot1,
                bot2,
                max_pool
            )
            bottleneck_dim += hidden_dim * num_obj * num_attr
            bottleneck_dim_conv += hidden_dim
            ## Image Encoder
            # if encode_image:
            # TODO: resnet
            # c_in --> hidden_dim
            # torchvision.models.resnet34
            self.feat_dim = hidden_dim
        self.encode_image = encode_image

        # Query encoder
        #TODO: add residual connections
        # bottleneck_dim += S_c_in
        fcd1 = MLP(bottleneck_dim, hidden_dim, hidden_dim*2)
        fcd2 = MLP(hidden_dim, latent_dim, hidden_dim)
        fcu1 = MLP(latent_dim, hidden_dim)
        # fcu2 = MLP(hidden_dim, hidden_dim)
        linear_out = nn.Linear(hidden_dim, latent_dim)

        self.query_encoder_ = nn.Sequential(
            fcd1,
            fcd2,
            fcu1,
            # fcu2,
            linear_out,
        )

        # conv_layers = nn.Sequential(
        #         nn.Conv2d(bottleneck_dim_conv, hidden_dim, kernel_size=5, padding=2, bias=True),
        #         # nn.GroupNorm(1, hidden_dim),
        #         nn.ReLU(),
        #         nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, bias=True),
        #         # nn.GroupNorm(1, hidden_dim),
        #         nn.ReLU(),
        #         nn.AvgPool2d(2),
        #         nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, bias=True),
        #         # nn.GroupNorm(1, hidden_dim),
        #         nn.ReLU(),
        #         nn.AvgPool2d(2),
        #         nn.Conv2d(hidden_dim, latent_dim, kernel_size=5, padding=2, bias=True),
        #         # nn.GroupNorm(1, hidden_dim),
        #         nn.ReLU(),
        # )
        #
        # self.query_encoder_ = conv_layers

        self.softmax = nn.Softmax(-1)

        dim_head = latent_dim
        self.scale = dim_head ** -0.5 #if not cosine_sim_attn else 1.
        # self.cosine_sim_attn = cosine_sim_attn
        # self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        inner_dim = dim_head

        self.norm = nn.LayerNorm(latent_dim)

        # self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(latent_dim, inner_dim, bias = False)
        self.to_k = nn.Linear(2*embed_dim, inner_dim, bias = False)

        # self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, latent_dim, bias = False),
            nn.LayerNorm(latent_dim)
        )
    def intp(self, x, ref, mode='bilinear'):
        return F.interpolate(x, ref.shape[-1], mode=mode, align_corners=True)
        # Mask layers

    def forward(self, cond, ans=None, image=None,
                image_features=None,
                query_features=None, mask=None, return_attn=False, cond_all=None):
        if not self.use_answers:
            x = cond
            x = x.reshape(x.shape[0], self.num_attr, self.num_obj, self.embeds_dim, x.shape[-1]).sum(-1)
            cond = x.reshape(x.shape[0], -1, self.embeds_dim)
            x = x.permute(0, 3, 1, 2)
        else:
            x = ans
            x = x.reshape(x.shape[0], self.num_attr, self.num_obj, 1, x.shape[-1])
            x = x[..., 0].float() - x[..., 1].float()
            x = x.permute(0, 3, 1, 2)

            scale = x.clone()
            scale[scale==0] = -10

        if cond_all is not None:
            cond_all_k = cond_all.sum(-1)
            cond_all = cond_all.reshape(x.shape[0], self.num_attr, self.num_obj, *cond_all.shape[-2:]).sum(-1).permute(0, 3, 1, 2)
            
            # Option 1
            cond_all = cond_all * scale
            # Option 2
            # cond = cond.reshape(x.shape[0], self.num_attr, self.num_obj, *cond.shape[-2:]).sum(-1).permute(0, 3, 1, 2)
            # mask_cond = mask.reshape(x.shape[0], 1, self.num_attr, self.num_obj)
            # # cond_all[:, cond.shape[1]:] = cond_all[dia-:, cond.shape[1]:] * (1-mask_cond) + cond * mask_cond
            # cond_all = cond_all * (1-mask_cond) + torch.cat([cond, torch.zeros_like(cond)], dim=1) * mask_cond

            cond_all_k = cond_all.permute(0, 2, 3, 1).reshape(x.shape[0], self.num_attr * self.num_obj, -1)
            x = torch.cat([x, cond_all], dim=1) # TODO: multiply by ans and put a token where 0.


        if self.encode_image:
            # TODO: Just do resnet with classification head, don't reinvent the wheel
            image = self.pos_encoder_im(image)
            h = self.image_encoder(image)
            h = h.reshape(-1, self.feat_dim, self.num_attr, self.num_obj)
            # x = x + self.alpha * h
            x = torch.cat([x, h], dim=1)
        #     print(x.shape, h.shape)
        # print(x.shape, self.query_encoder)

        # if self.position_enc:
        #     x = self.pos_encoder_q(x)
        # x = x.reshape(x.shape[0], -1)


        x = x.reshape(x.shape[0], -1)
        x = self.query_encoder_(x)
        x = x.reshape(x.shape[0], -1)
        # x = x.reshape(*x.shape[:2], -1).mean(-1)

        # remove elements in mask
        query_logits_pre = x.view(x.shape[0], -1)

        # b, n, device = *x.shape[:2], x.device

        # x = self.norm(x)

        q, k = self.to_q(x), self.to_k(cond_all_k) #TODO cond all includes all conditions + object embeddings, which is also part of the input.

        q = q * self.scale

        # calculate query / key similarities

        sim = einsum('b d, b j d -> b j', q, k) #* self.cosine_sim_scale

        query_logits_pre = sim
        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = mask == 1
            # mask = F.pad(mask, (0, feat_kv_dim), value=True)
            # mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(mask, max_neg_value)

        # attention

        attn = (sim / self.tau).softmax(dim=-1) # TODO: Add noise
        # attn = attn.to(sim.dtype)

        query = attn
        # query_out = query # einsum('b j, b j d -> b d', attn, cond)

        # query_mask = torch.where(mask == 1, -1e8, torch.zeros((1,)).to(x.device))
        # query_logits = query_logits_pre + query_mask  # .to(x.device)
        # # straight through softmax
        # query = self.softmax(query_logits / self.tau)
        #
        _, max_ind = (query).max(1)
        query_onehot = F.one_hot(max_ind, query.shape[1]).type(query.dtype)
        query_out = (query_onehot - query).detach() + query


        if return_attn:
            # query = self.dropout(query)
            query_logits_pre = query_logits_pre - torch.min(query_logits_pre, dim=1, keepdim=True)[0]
            query_logits_pre = query_logits_pre / torch.max(query_logits_pre, dim=1, keepdim=True)[0]
            return query_out.reshape(-1, self.num_attr * self.num_obj, 1), \
                query.reshape(-1, self.num_attr * self.num_obj, 1), \
                query_logits_pre
        # else: query_out = query
        return query_out.reshape(-1, self.num_attr * self.num_obj, 1), \
            query.reshape(-1, self.num_attr * self.num_obj, 1)

class QuerierFactorizedImageAttr(nn.Module):
    def __init__(self, embed_dim=5, num_obj=5, num_attr=5, image_size=128, latent_dim=64, hidden_dim=128, encode_image=True,
                 add_object_embedding=True, use_latent=True, use_answers=False, pos_enc_fn=None):
        super().__init__()
        # TODO: pass positional encoding as input with Imagenet function. LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        position_enc = True
        self.position_enc = position_enc
        # self.latent
        self.tau = 1
        self.use_answers = use_answers
        bottleneck_dim = 0
        # hidden_dim = embed_dim
        embed_dim = 1 if use_answers else embed_dim
        c_in = 3

        S_c_in = embed_dim * num_obj * num_attr
        self.embeds_dim, self.num_obj, self.num_attr = embed_dim, num_obj, num_attr
        if position_enc and encode_image:
            d_pe = 4
            if not isinstance(image_size, tuple):
                d_spa = (image_size, image_size)
            else:
                d_spa = image_size
            # d_spa = (num_obj, num_attr)
            self.pos_encoder_im = PositionalEncoding2D(d_model=d_pe, d_spatial=d_spa, dropout=0.0)
            c_in += d_pe

        if encode_image:
            inc = DoubleConv(c_in, 64)
            down1 = Down_uc(64, 128)
            size = image_size // 2
            # self.sa1 = SelfAttention(128, size)
            down2 = Down_uc(128, 128)
            size = size // 2
            # self.sa2 = SelfAttention(128, size)
            down3 = Down_uc(128, 256)
            size = size // 2
            sa3 = SelfAttention(256, size)
            bot1 = DoubleConv(256, 256)
            bot2 = DoubleConv(256, hidden_dim) # * num_obj * num_attr
            max_pool = nn.AdaptiveMaxPool2d((1, 1))

            self.image_encoder = nn.Sequential(
                inc,
                down1,
                down2,
                down3,
                sa3,
                bot1,
                bot2,
                max_pool
            )
            bottleneck_dim += hidden_dim #* num_obj * num_attr

        ## Image Encoder
        # if encode_image:
        # TODO: resnet
        # c_in --> hidden_dim
        # torchvision.models.resnet34
        self.feat_dim = hidden_dim
        self.encode_image = encode_image

        # Query encoder
        #TODO: add residual connections
        bottleneck_dim += S_c_in
        fcd1 = MLPLeaky(bottleneck_dim, hidden_dim, hidden_dim*2)
        fcd2 = MLPLeaky(hidden_dim, latent_dim, hidden_dim)
        fcu1 = MLPLeaky(latent_dim, hidden_dim)
        # self.fcu2 = MLP(hidden_dim, hidden_dim)
        # self.linear_out = nn.Linear(hidden_dim, num_obj * num_attr)

        self.query_encoder_ = nn.Sequential(
            fcd1,
            fcd2,
            fcu1,
            # self.fcu2,
        )

        self.proj_obj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.3),  # nn.ReLU(),
            nn.Linear(hidden_dim, num_obj, bias=False),
        )

        self.proj_attr = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.3),  # nn.ReLU(),
            nn.Linear(hidden_dim, num_attr, bias=False),
        )

        self.softmax = nn.Softmax(-1)

    def intp(self, x, ref, mode='bilinear'):
        return F.interpolate(x, ref.shape[-1], mode=mode, align_corners=True)
        # Mask layers

    def forward(self, cond, ans=None, image=None, mask=None, return_attn=False):
        if not self.use_answers:
            x = cond
            x = x.reshape(x.shape[0], self.num_attr, self.num_obj, self.embeds_dim, x.shape[-1]).sum(-1)
            x = x.permute(0, 3, 1, 2)
        else:
            x = ans
            x = x.reshape(x.shape[0], self.num_attr, self.num_obj, self.embeds_dim, x.shape[-1])
            x = x[..., 0].float() - x[..., 1].float()
            x = x.permute(0, 3, 1, 2)

        if self.encode_image:
            # TODO: Just do resnet with classification head, don't reinvent the wheel
            image = self.pos_encoder_im(image)
            h = self.image_encoder(image)
            h = h.reshape(-1, self.feat_dim, 1, 1).repeat(1, 1, self.num_attr, self.num_obj)
            # h = h.reshape(-1, self.feat_dim, self.num_attr, self.num_obj)
            # x = x + self.alpha * h
            x = torch.cat([x, h], dim=1)

        # if self.position_enc:
        #     x = self.pos_encoder_q(x)
        x = x.reshape(x.shape[0], -1)
        x = self.query_encoder_(x)
        x_obj = self.proj_obj(x).view(x.shape[0], -1)
        x_attr = self.proj_attr(x).view(x.shape[0], -1)

        x = x_attr[:, :, None] * x_obj[:, None, :] # Factorized query.

        # remove elements in mask
        query_logits_pre = x.view(x.shape[0], -1)
        query_mask = torch.where(mask == 1, -1e8, torch.zeros((1,)).to(x.device))
        query_logits = query_logits_pre + query_mask  # .to(x.device)
        # query_logits = query_logits_pre

        # straight through softmax
        query = self.softmax(query_logits / self.tau)

        _, max_ind = (query).max(1)
        query_onehot = F.one_hot(max_ind, query.shape[1]).type(query.dtype)
        query_out = (query_onehot - query).detach() + query


        if return_attn:
            # query = self.dropout(query)
            query_logits_pre = query_logits_pre - torch.min(query_logits_pre, dim=1, keepdim=True)[0]
            query_logits_pre = query_logits_pre / torch.max(query_logits_pre, dim=1, keepdim=True)[0]
            return query_out.reshape(-1, self.num_attr * self.num_obj, 1), \
                query.reshape(-1, self.num_attr * self.num_obj, 1), \
                query_logits_pre
        # else: query_out = query
        return query_out.reshape(-1, self.num_attr * self.num_obj, 1), \
            query.reshape(-1, self.num_attr * self.num_obj, 1)

class S_AE(nn.Module):
    def __init__(self, embed_dim, latent_dim=16, hidden_dim=64, num_obj=5, num_attr=5, use_answers=True):
        super().__init__()
        # self.device = device
        #TODO: pass positional encoding as input with Imagenet function. LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        position_enc = True
        self.use_answers = use_answers
        decoder_type = 'mlp'
        if use_answers:
            input_dim = 1
        else:
            input_dim = embed_dim
        first_hidden_dim = hidden_dim

        self.input_dim, self.num_obj, self.num_attr = input_dim, num_obj, num_attr

        if position_enc:
            d_pe_q = 4
            d_spa_q = (num_attr, num_obj)
            pos_encoder_q = PositionalEncoding2D(d_model=d_pe_q, d_spatial=d_spa_q, dropout=0.0)
            first_hidden_dim += d_pe_q


        ## Image Encoder
        self.feat_dim = hidden_dim

        # Query encoder
        # self.inc = DoubleConv(, hidden_dim)
        conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        gn1 = nn.GroupNorm(1, hidden_dim)
        act1 = nn.GELU()
        conv2 = nn.Conv2d(first_hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        sa4 = SelfAttention(hidden_dim, (num_attr, num_obj))
        sa5 = SelfAttention(hidden_dim, (num_attr, num_obj))
        conv_end = nn.Conv2d(hidden_dim, latent_dim, kernel_size=(1, num_obj), stride=(1, 1), padding=0)
        amax_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.query_encoder = nn.Sequential(
            conv1,
            gn1,
            act1,
            pos_encoder_q,
            conv2,
            sa4,
            sa5,
            conv_end,
            amax_pool
        )


        # Decoder
        if decoder_type == 'mlp':
            linear1 = nn.Linear(latent_dim, hidden_dim)
            gn1 = nn.GroupNorm(1, hidden_dim)
            act1 = nn.GELU()
            linear2 = nn.Linear(hidden_dim, hidden_dim)
            gn2 = nn.GroupNorm(1, hidden_dim)
            act2 = nn.GELU()
            linear3 = nn.Linear(hidden_dim, hidden_dim)
            gn3 = nn.GroupNorm(1, hidden_dim)
            act3 = nn.GELU()
            linear4 = nn.Linear(hidden_dim, num_attr * num_obj * self.input_dim)
            self.query_decoder = nn.Sequential(
               linear1,
                gn1,
                act1,
                linear2,
                gn2,
                act2,
                linear3,
                gn3,
                act3,
                linear4
            )

    def intp(self, x, ref, mode='bilinear'):
        return F.interpolate(x, ref.shape[-1], mode=mode, align_corners=True)
        # Mask layers

    def encode(self, x):
        z = self.query_encoder(x).squeeze(-1).squeeze(-1)
        return z

    def decode(self, z):
        x = self.query_decoder(z).reshape(z.shape[0], -1, self.num_attr, self.num_obj)
        return x
    def forward(self, cond=None, ans=None):
        if self.use_answers:
            x = ans
        else:
            x = cond
        x_pos, x_neg = x[0], x[1]
        x = x_pos + x_neg  # torch.cat([x_pos, x_neg], dim=1) # There shouldn't be overlap
        if len(x) == 3:
            x = x + x[2]
        x = x.reshape(-1, self.num_attr, self.num_obj, self.input_dim)
        x_in = x.permute(0, 3, 1, 2)
        z = self.encode(x_in)
        return x_in, self.decode(z), z

class S_ConvAE(nn.Module):
    def __init__(self, embed_dim, latent_dim=16, hidden_dim=64, num_obj=5, num_attr=5, use_answers=True):
        super().__init__()
        # self.device = device
        #TODO: pass positional encoding as input with Imagenet function. LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        self.use_answers = use_answers
        decoder_type = 'mlp'
        if use_answers:
            input_dim = 1
        else:
            input_dim = embed_dim
        first_hidden_dim = hidden_dim

        self.input_dim, self.num_obj, self.num_attr = input_dim, num_obj, num_attr


        ## Image Encoder
        self.feat_dim = hidden_dim

        self.inc = DoubleConv_wK(input_dim, hidden_dim//2,
                                 kernel=(1, 5), padding=(0, 2), residual=True)
        self.conv_1 = DoubleConv_wK(hidden_dim//2, hidden_dim//2,
                                 kernel=(1, 3), padding=(0, 1), residual=True)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(hidden_dim//2, hidden_dim, kernel_size=(1, num_obj), padding=(0, 0), bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU()
        )
        self.conv_3 = DoubleConv_wK(hidden_dim, hidden_dim,
                                 kernel=(5, 1), padding=(2, 0), residual=True)
        # self.conv_4 = DoubleConv_wK(hidden_dim, hidden_dim,
        #                          kernel=(5, 1), padding=(2, 0), residual=True)
        self.conv_4 = DoubleConv_wK(hidden_dim, hidden_dim,
                                 kernel=(5, 1), padding=(2, 0))
        self.conv_out = nn.Conv2d(hidden_dim, latent_dim, kernel_size=1, stride=1, padding=0)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.query_encoder = nn.Sequential(
            self.inc,
            self.conv_1,
            self.conv_2,
            self.conv_3,
            self.conv_4,
            # self.conv_5,
            self.conv_out,
            self.adaptive_pool
        )


        # Decoder
        if decoder_type == 'mlp':
            linear1 = nn.Linear(latent_dim, hidden_dim)
            gn1 = nn.GroupNorm(1, hidden_dim)
            act1 = nn.GELU()
            linear2 = nn.Linear(hidden_dim, hidden_dim)
            gn2 = nn.GroupNorm(1, hidden_dim)
            act2 = nn.GELU()
            linear3 = nn.Linear(hidden_dim, hidden_dim)
            gn3 = nn.GroupNorm(1, hidden_dim)
            act3 = nn.GELU()
            linear4 = nn.Linear(hidden_dim, num_attr * num_obj * self.input_dim)
            self.query_decoder = nn.Sequential(
               linear1,
                gn1,
                act1,
                linear2,
                gn2,
                act2,
                linear3,
                gn3,
                act3,
                linear4
            )

    def intp(self, x, ref, mode='bilinear'):
        return F.interpolate(x, ref.shape[-1], mode=mode, align_corners=True)
        # Mask layers

    def encode(self, x):
        # print(x.shape)
        # x = self.inc(x)
        # print(x.shape)
        # x = self.conv_1(x)
        # print(x.shape)
        # x = self.conv_2(x)
        # print(x.shape)
        # x = self.conv_3(x)
        # print(x.shape)
        # x = self.conv_4(x)
        # print(x.shape)
        # # x = self.conv_5(x)
        # # print(x.shape)
        # x = self.conv_out(x)
        # print(x.shape)
        # z = self.adaptive_pool(x)
        # print(z.shape)
        # exit()
        z = self.query_encoder(x)
        return z.squeeze(-1).squeeze(-1)

    def decode(self, z):
        x = self.query_decoder(z).reshape(z.shape[0], -1, self.num_attr, self.num_obj)
        return x
    def forward(self, cond=None, ans=None):
        if self.use_answers:
            x = ans
        else:
            x = cond
        x_pos, x_neg, x_unasked = x[0], x[1], x[2]
        x = x_pos + x_neg + x_unasked  # torch.cat([x_pos, x_neg], dim=1) # There shouldn't be overlap
        x = x.reshape(-1, self.num_attr, self.num_obj, self.input_dim)
        x_in = x.permute(0, 3, 1, 2)
        z = self.encode(x_in)
        return x_in, self.decode(z), z

class S_PoolMLP(nn.Module):
    def __init__(self, embed_dim, latent_dim=16, hidden_dim=64, num_obj=5, num_attr=5, embeds=None, use_answers=True):
        super().__init__()
        # self.device = device
        #TODO: pass positional encoding as input with Imagenet function. LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        self.use_answers = use_answers
        decoder_type = 'mlp'

        input_dim = embed_dim * 2

        self.input_dim, self.num_obj, self.num_attr = input_dim, num_obj, num_attr

        self.pos_emb, self.neg_emb, self.obj_emb = embeds

        ## Image Encoder
        self.feat_dim = hidden_dim

        self.inc = DoubleConv_wK(input_dim, hidden_dim,
                                 kernel=1, padding=0)
        self.conv_out = nn.Conv2d(hidden_dim, latent_dim,
                                 kernel_size=1, padding=0)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((num_attr, 1))
        self.query_encoder = nn.Sequential(
            self.inc,
            self.conv_out,
            self.adaptive_pool
        )

        # Decoder
        if decoder_type == 'mlp':
            linear1 = nn.Linear(latent_dim * num_attr, hidden_dim)
            gn1 = nn.GroupNorm(1, hidden_dim)
            act1 = nn.GELU()
            linear2 = nn.Linear(hidden_dim, hidden_dim)
            gn2 = nn.GroupNorm(1, hidden_dim)
            act2 = nn.GELU()
            linear4 = nn.Linear(hidden_dim, num_attr * num_obj * 1)
            self.query_decoder = nn.Sequential(
               linear1,
                gn1,
                act1,
                linear2,
                gn2,
                act2,
                # linear3,
                # gn3,
                # act3,
                linear4
            )

    def intp(self, x, ref, mode='bilinear'):
        return F.interpolate(x, ref.shape[-1], mode=mode, align_corners=True)
        # Mask layers

    def encode(self, x):
        z = self.query_encoder(x)
        return z.squeeze(-1).permute(0, 2, 1) #.squeeze(-1)

    def decode(self, z):
        z = z.reshape(z.shape[0], -1)
        x = self.query_decoder(z).reshape(z.shape[0], -1, self.num_attr, self.num_obj)
        return x
    def forward(self, cond=None, ans=None):
        x = ans
        x_pos, x_neg, x_unasked = x[0].reshape(-1, self.num_attr, self.num_obj, 1), x[1].reshape(-1, self.num_attr, self.num_obj, 1), x[2].reshape(-1, self.num_attr, self.num_obj, 1)
        x = x_pos + x_neg + x_unasked
        target = x.permute(0, 3, 1, 2)
        all = torch.ones_like(x[..., 0], dtype=torch.int32)
        attr_ls = torch.linspace(1, self.num_attr, self.num_attr, device=x.device, dtype=torch.int32)[None, :, None]
        obj_ls  = torch.linspace(0, self.num_obj-1, self.num_obj, device=x.device, dtype=torch.int32)[None, None, :]
        all_attr, all_obj = all * attr_ls, all * obj_ls
        emb_pos_attr, emb_neg_attr, emb_ua_attr, emb_obj = \
            self.pos_emb(all_attr), self.neg_emb(all_attr), self.pos_emb(all * 0), self.obj_emb(all_obj)

        emb_attr = emb_pos_attr * (x_pos == 1) + \
                   emb_neg_attr * (x_neg == -1) + \
                   emb_ua_attr * (x_unasked != 0)

        emb = torch.cat([emb_attr, emb_obj], dim=-1)

        x = emb.reshape(-1, self.num_attr, self.num_obj, self.input_dim)
        x_enc = x.permute(0, 3, 1, 2)
        z = self.encode(x_enc)
        return target, self.decode(z), z

# class QuerierImageAnswersAttr(nn.Module):
#     def __init__(self, embed_dim=5, num_obj=5, num_attr=5, image_size=128, hidden_dim=256, encode_image=True):
#         super().__init__()
#         # self.device = device
#         position_enc = True
#         self.tau = 1
#         bottleneck_dim = 0
#         # hidden_dim = embed_dim
#
#         c_in = 3 #embed_dim * 2  # (pos and neg)
#         self.embeds_dim, self.num_obj, self.num_attr = embed_dim, num_obj, num_attr
#         if position_enc and encode_image:
#             d_pe = 4
#             if not isinstance(image_size, tuple):
#                 d_spa = (image_size, image_size)
#             else: d_spa = image_size
#             # d_spa = (num_obj, num_attr)
#             self.pos_encoder_im = PositionalEncoding2D(d_model=d_pe, d_spatial=d_spa, dropout=0.0)
#             c_in += d_pe
#
#         if position_enc:
#             d_pe_q = 4
#             d_spa_q = (num_attr, num_obj)
#             self.pos_encoder_q = PositionalEncoding2D(d_model=d_pe_q, d_spatial=d_spa_q, dropout=0.0)
#             bottleneck_dim += d_pe_q
#
#
#         ## Image Encoder
#         if encode_image:
#             self.inc = DoubleConv(c_in, 64)
#             self.down1 = Down_uc(64, 128)
#             size = image_size//2
#             # self.sa1 = SelfAttention(128, size)
#             self.down2 = Down_uc(128, 128)
#             size = size//2
#             # self.sa2 = SelfAttention(128, size)
#             self.down3 = Down_uc(128, 256)
#             size = size//2
#             self.sa3 = SelfAttention(256, size)
#             self.bot1 = DoubleConv(256, 256)
#             self.bot2 = DoubleConv(256, hidden_dim * num_obj * num_attr)
#             self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
#
#             self.image_encoder = nn.Sequential(
#                 self.inc,
#                 self.down1,
#                 self.down2,
#                 self.down3,
#                 self.sa3,
#                 self.bot1,
#                 self.bot2,
#                 self.max_pool
#             )
#             bottleneck_dim += hidden_dim
#
#             # create a parameter
#             self.alpha = nn.Parameter(torch.zeros(1,))
#         self.feat_dim = hidden_dim
#         self.encode_image = encode_image
#
#         # Query encoder
#         bottleneck_dim += 1 # embed_dim
#         self.inc = DoubleConv(bottleneck_dim, hidden_dim)
#         # self.conv1d = nn.Conv2d(bottleneck_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
#         self.sa4 = SelfAttention(hidden_dim, (num_attr, num_obj))
#         self.sa5 = SelfAttention(hidden_dim, (num_attr, num_obj))
#         self.conv_end = nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1, padding=0)
#         self.query_encoder = nn.Sequential(
#             self.inc,
#             self.sa4,
#             self.sa5,
#             self.conv_end,
#         )
#
#         self.softmax = nn.Softmax(-1)
#
#     def intp(self, x, ref, mode='bilinear'):
#         return F.interpolate(x, ref.shape[-1], mode=mode, align_corners=True)
#         # Mask layers
#
#     def forward(self, cond, ans=None, image=None, mask=None, return_attn=False):
#         x = cond
#         x = x.reshape(x.shape[0], self.num_attr, self.num_obj, 1, 2).sum(-1)
#         x = x.permute(0, 3, 1, 2)
#
#         if self.encode_image:
#             image = self.pos_encoder_im(image)
#             h = self.image_encoder(image)
#             h = h.reshape(-1, self.feat_dim, self.num_attr, self.num_obj)
#             x = torch.cat([x, self.alpha * h], dim=1)
#
#         x = self.pos_encoder_q(x)
#         x = self.query_encoder(x)
#
#         # remove elements in mask
#         query_logits_pre = x.view(x.shape[0], -1)
#         query_mask = torch.where(mask == 1, -1e8, # query_logits_pre.min().detach()
#                                  torch.zeros((1,)).to(x.device))
#         query_logits = query_logits_pre + query_mask  # .to(x.device)
#
#         # straight through softmax
#         query = self.softmax(query_logits / self.tau)
#         # query = query_logits / (torch.sum(query_logits, dim=-1, keepdim=True) + 1e-8)
#         _, max_ind = (query).max(1)
#         query_onehot = F.one_hot(max_ind, query.shape[1]).type(query.dtype)
#         query_out = (query_onehot - query).detach() + query
#
#         if return_attn:
#             # TODO: Check if this being soft is essential.
#             # query = self.dropout(query)
#             query_logits_pre = query_logits_pre - torch.min(query_logits_pre, dim=1, keepdim=True)[0]
#             query_logits_pre = query_logits_pre / torch.max(query_logits_pre, dim=1, keepdim=True)[0]
#             return query_out.reshape(-1, self.num_attr * self.num_obj, 1), \
#                 query_logits_pre
#         # else: query_out = query
#         return query_out.reshape(-1, self.num_attr * self.num_obj, 1)


class SAUNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, size=28, patch_size=5, out_size=None, multi_resolution=False, device="cuda"):
        super().__init__()
        self.device = device
        position_enc = False
        if out_size is None:
            out_size = size - patch_size + 1
        if position_enc:
            d_pe = 4
            if not isinstance(out_size, tuple):
                d_spa = (size, size)
            else: d_spa = size
            self.pos_encoder = PositionalEncoding2D(d_model=d_pe, d_spatial=d_spa, dropout=0.0)
            c_in += d_pe
        self.tau = 1.0
        # c_in += 1

        self.inc_ = DoubleConv(c_in, 64)
        self.down1_ = Down_uc(64, 128)
        size = size//2
        # self.sa1 = SelfAttention(128, size)
        self.down2_ = Down_uc(128, 128)
        size = size//2
        # self.sa2 = SelfAttention(128, size)
        self.down3_ = Down_uc(128, 256)
        size = size//2
        self.sa3_ = SelfAttention(256, size)

        self.bot1_ = DoubleConv(256, 256)
        self.bot2_ = DoubleConv(256, 256)
        self.bot3_ = DoubleConv(256, 128)

        self.up1_ = Up_uc(256, 128)
        size = size*2
        self.sa4_ = SelfAttention(128, size)
        self.up2_ = Up_uc(256, 64)
        size = size*2
        # self.sa5 = SelfAttention(64, size)
        self.up3_ = Up_uc(128, 64)
        size = size*2
        # self.sa6 = SelfAttention(64, size)
        self.upout_ = nn.Upsample(size=out_size, mode='nearest')
        self.outc_ = nn.Conv2d(64, c_out, kernel_size=1, bias=False)

        self.multi_resolution = multi_resolution
        if self.multi_resolution:
            self.out_1 = nn.Sequential(
                nn.Conv2d(128, 64, 1, bias=False),
                nn.GroupNorm(1, 64),
                nn.ReLU(),
                nn.Conv2d(64, 1, 1, bias=False),
            )

            self.out_2 = nn.Sequential(
                nn.Conv2d(128, 64, 1, bias=False),
                nn.GroupNorm(1, 64),
                nn.ReLU(),
                nn.Conv2d(64, 1, 1, bias=False),
            )

            self.out_3 = nn.Sequential(
                nn.Conv2d(64, 32, 1, bias=True),
                nn.GroupNorm(1, 32),
                nn.ReLU(),
                nn.Conv2d(32, 1, 1, bias=False),
            )
            self.out_1.conv_2.weight.data.fill_(0.0)
            self.out_2.conv_2.weight.data.fill_(0.0)
            self.out_3.conv_2.weight.data.fill_(0.0)
            print('We should initialize it as 0, not train it and then train it after.')

        # self.outc_.weight.data.fill_(0.01)

        self.softmax = nn.Softmax(-1)
        self.softmax2d = nn.Softmax2d()

    def intp(self, x, ref, mode='bilinear'):
        return F.interpolate(x, ref.shape[-1], mode=mode, align_corners=True)
        # Mask layers
    def forward(self, x, mask, return_attn=False):
        # h = w = int(math.sqrt(mask.shape[-1]))
        # mask_img = mask.reshape(x.shape[0], 1, w, h)
        # nmask = 1 - mask_img

        # x = self.pos_encoder(x)

        x1 = self.inc_(x)
        x2 = self.down1_(x1)
        # x2 = x2 * self.intp(nmask, x2)
        # x2 = self.sa1_(x2)
        x3 = self.down2_(x2)
        # x3 = x3 * self.intp(nmask, x3)
        # x3 = self.sa2_(x3)
        x4 = self.down3_(x3)
        # x4 = x4 * self.intp(nmask, x4)
        x4 = self.sa3_(x4)
        x = x4

        x4 = self.bot1_(x)
        x4 = self.bot2_(x4)
        x = self.bot3_(x4)

        res1 = x
        x = self.up1_(x, x3)
        # x = x * self.intp(nmask, x)
        x = self.sa4_(x)
        res2 = x
        x = self.up2_(x, x2)
        # x = x * self.intp(nmask, x)
        # x = self.sa5_(x)
        res3 = x
        x = self.up3_(x, x1)
        # x = x * self.intp(nmask, x)
        # x = self.sa6_(x)
        # print(x.shape)
        # print(x.shape)
        x = self.outc_(x)
        x = self.upout_(x)

        N = x.shape[0]
        query_logits_pre = x.view(N, -1)
        n_q = query_logits_pre.shape[-1]

        query_mask = torch.where(mask == 1, -1e9, torch.zeros((1,)).to(x.device))  # TODO: Check why.
        # identity_mask = torch.zeros_like(mask).reshape(*x.shape)
        # identity_mask[..., x.shape[-2] // 2, x.shape[-1] // 2] = 1
        # query_mask += identity_mask.reshape(*query_mask.shape)

        query_logits = query_logits_pre + query_mask   # .to(x.device)

        # add_noise = True
        # if add_noise:
        #     query_logits = query_logits + (torch.randn_like(query_logits_pre) * self.tau)

        # straight through softmax
        query = self.softmax(query_logits / (self.tau))

        if self.multi_resolution:
            # print(res1.shape, self.out_1)
            if res1.shape[-1]>1:
                query = query * self.upout_(self.softmax2d(self.out_1(res1) / self.tau)).view(N, -1)
            query = query * self.upout_(self.softmax2d(self.out_2(res2) / self.tau)).view(N, -1)
            query = query * self.upout_(self.softmax2d(self.out_3(res3) / self.tau)).view(N, -1)

        _, max_ind = (query).max(1)
        query_onehot = F.one_hot(max_ind, query.shape[1]).type(query.dtype)
        query_out = (query_onehot - query).detach() + query

        if return_attn:
            # TODO: Check if this being soft is essential.
            # query = self.dropout(query)
            query_logits_pre = query_logits_pre - torch.min(query_logits_pre, dim=1, keepdim=True)[0]
            query_logits_pre = query_logits_pre / torch.max(query_logits_pre, dim=1, keepdim=True)[0]
            return query_out, query, query_logits_pre
        # else: query_out = query
        # print('Query_out is set to query!! Should be query_onehot')
        # print('We changed it to query_onehot')
        return query_out, query


# class SAUNet(nn.Module):
#     def __init__(self, c_in=1, c_out=1, size=28, patch_size=5, out_size=None, device="cuda"):
#         super().__init__()
#         self.device = device
#         position_enc = False
#         if out_size is None:
#             out_size = size - patch_size + 1
#         if position_enc:
#             d_pe = 4
#             if not isinstance(out_size, tuple):
#                 d_spa = (size, size)
#             else: d_spa = size
#             self.pos_encoder = PositionalEncoding2D(d_model=d_pe, d_spatial=d_spa, dropout=0.0)
#             c_in += d_pe
#         self.tau = 1.0
#         # c_in += 1
#
#         self.inc_ = DoubleConv(c_in, 64)
#         self.down1_ = Down_uc(64, 128)
#         size = size//2
#         # self.sa1 = SelfAttention(128, size)
#         self.down2_ = Down_uc(128, 128)
#         size = size//2
#         # self.sa2 = SelfAttention(128, size)
#         self.down3_ = Down_uc(128, 256)
#         size = size//2
#         # self.sa3_ = SelfAttention(256, size)
#
#         # self.bot1_ = DoubleConv(256, 256)
#         # self.bot2_ = DoubleConv(256, 256)
#         self.bot3_ = DoubleConv(256, 128)
#
#         self.up1_ = Up_uc(256, 128)
#         size = size*2
#         # self.sa4_ = SelfAttention(128, size)
#         self.up2_ = Up_uc(256, 64)
#         size = size*2
#         # self.sa5 = SelfAttention(64, size)
#         self.up3_ = Up_uc(128, 64)
#         size = size*2
#         # self.sa6 = SelfAttention(64, size)
#         self.upout_ = nn.Upsample(size=out_size, mode='bilinear')
#         self.outc_ = nn.Conv2d(64, c_out, kernel_size=1, bias=False)
#
#         self.outc_.weight.data.fill_(0.01)
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
#         # x = self.pos_encoder(x)
#
#         x1 = self.inc_(x)
#         x2 = self.down1_(x1)
#         # x2 = x2 * self.intp(nmask, x2)
#         # x2 = self.sa1_(x2)
#         x3 = self.down2_(x2)
#         # x3 = x3 * self.intp(nmask, x3)
#         # x3 = self.sa2_(x3)
#         x4 = self.down3_(x3)
#         # x4 = x4 * self.intp(nmask, x4)
#         # x4 = self.sa3_(x4)
#         x = x4
#
#         # x4 = self.bot1_(x)
#         # x4 = self.bot2_(x4)
#         x = self.bot3_(x4)
#
#         x = self.up1_(x, x3)
#         # x = x * self.intp(nmask, x)
#         # x = self.sa4_(x)
#         x = self.up2_(x, x2)
#         # x = x * self.intp(nmask, x)
#         # x = self.sa5_(x)
#         x = self.up3_(x, x1)
#         # x = x * self.intp(nmask, x)
#         # x = self.sa6_(x)
#         # print(x.shape)
#         x = self.upout_(x)
#         # print(x.shape)
#         x = self.outc_(x)
#
#         query_logits_pre = x.view(x.shape[0], -1)
#         n_q = query_logits_pre.shape[-1]
#
#         query_mask = torch.where(mask == 1, -1e9, torch.zeros((1,)).to(x.device))  # TODO: Check why.
#         identity_mask = torch.zeros_like(mask).reshape(*x.shape)
#         identity_mask[..., x.shape[-2] // 2, x.shape[-1] // 2] = 1
#         query_mask += identity_mask.reshape(*query_mask.shape)
#
#         query_logits = query_logits_pre + query_mask   # .to(x.device)
#
#         # add_noise = True
#         # if add_noise:
#         #     query_logits = query_logits + (torch.randn_like(query_logits_pre) * self.tau)
#
#         # straight through softmax
#         query = self.softmax(query_logits / (n_q * self.tau))
#         _, max_ind = (query).max(1)
#         query_onehot = F.one_hot(max_ind, query.shape[1]).type(query.dtype)
#         query_out = (query_onehot - query).detach() + query
#
#         if return_attn:
#             # TODO: Check if this being soft is essential.
#             # query = self.dropout(query)
#             query_logits_pre = query_logits_pre - torch.min(query_logits_pre, dim=1, keepdim=True)[0]
#             query_logits_pre = query_logits_pre / torch.max(query_logits_pre, dim=1, keepdim=True)[0]
#             return query_out, query, query_logits_pre
#         # else: query_out = query
#         # print('Query_out is set to query!! Should be query_onehot')
#         # print('We changed it to query_onehot')
#         return query_out, query


# class SAUNet(nn.Module):
#     def __init__(self, c_in=1, c_out=1, size=28, patch_size=5, out_size=None, device="cuda"):
#         super().__init__()
#         self.device = device
#         if out_size is None:s
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