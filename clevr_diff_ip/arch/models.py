import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import pdb
import utils
import math
# from modules import UNet_conditional, SAUnet

# encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
# transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
# src = torch.rand(10, 32, 512)
# out = transformer_encoder(src)
import random
import numpy as np
class QuerierEncoder(nn.Module):
    def __init__(self, tau=1.0, num_queries=100, query_dim=64, in_channels=1, num_layers=6, num_heads=8, dropout=0.0):
        super().__init__()
        self.tau = tau

        # ENCODER
        self.conv1 = nn.Conv2d(in_channels, 32, 3)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bnorm4 = nn.BatchNorm2d(256)
        self.conv5 = torch.nn.Sequential(nn.Upsample(size=(8, 8), mode='nearest'),
                                           nn.Conv2d(256, 128, kernel_size=3))
        self.bnorm5 = nn.BatchNorm2d(128)
        self.conv6 = torch.nn.Sequential(nn.Upsample(size=(1, 1), mode='nearest'),
                                         nn.Conv2d(128, out_channels=num_queries, kernel_size=1))

        # activations
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu = nn.LeakyReLU(negative_slope=0.3)
        self.softmax = nn.Softmax(dim=-1)

        # Positional embedding
        self.pos_encoder = PositionalEncoding(query_dim, dropout=dropout)

        # Embeddings
        self.embedding = nn.Embedding(num_queries, query_dim) # max_norm=True ?

        # Transformer (group embeddings)
        encoder_layer = nn.TransformerEncoderLayer(d_model=query_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        # src = torch.rand(10, 32, 512)
        # out = self.transformer_encoder(src)

        self.query_dim = query_dim

    def encode(self, x):
        x = self.relu(self.bnorm1(self.conv1(x)))
        x = self.maxpool1(self.relu(self.bnorm2(self.conv2(x))))
        x = self.relu(self.bnorm3(self.conv3(x)))
        x = self.maxpool2(self.relu(self.bnorm4(self.conv4(x))))
        x = self.relu(self.bnorm5(self.conv5(x)))
        x = self.conv6(x)
        return x

    def update_tau(self, tau):
        self.tau = tau

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def forward(self, x, mask):
        device = x.device

        # TODO: Mask should be added to the encoding process as side information.
        #  We need to know what has been selected aside from removing the option from the prediction

        x = self.encode(x)

        query_logits = x.view(x.shape[0], -1)
        query_mask = torch.where(mask == 1, -1e9, 0.) # TODO: Check why.
        query_logits = query_logits + query_mask.to(device)

        # straight through softmax
        query = self.softmax(query_logits / self.tau)
        _, max_ind = (query).max(1)
        query_onehot = F.one_hot(max_ind, query.shape[1]).type(query.dtype)
        query = (query_onehot - query).detach() + query

        x = self.embedding(query) * math.sqrt(self.query_dim)

        return query

class QueryAggregator(nn.Module):
    def __init__(self, tau=1.0, num_queries=100, query_dim=64, in_channels=1, num_layers=6, num_heads=8, dropout=0.0):
        super().__init__()

        # Positional embedding
        self.pos_encoder = PositionalEncoding(query_dim, dropout=dropout)

        # Embeddings
        self.embedding = nn.Embedding(num_queries, query_dim) # max_norm=True ?

        # Transformer (group embeddings)
        encoder_layer = nn.TransformerEncoderLayer(d_model=query_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        # src = torch.rand(10, 32, 512)
        # out = self.transformer_encoder(src)

        self.query_dim = query_dim

    def forward(self, updated_mask, true_labels):

        x = self.embedding(updated_mask) * math.sqrt(self.query_dim)
        # Append answers with true_labels.
        # x = self.pos_encoder(x) #Note: Only if we care about the order, but we might not as the objects have no order. It is a Set
        S = self.transformer_encoder(x, updated_mask)

        return S

# class Querier128(nn.Module):
#     def __init__(self, tau=1.0, query_size = (128, 128), in_channels=1):
#         super().__init__()
#         self.tau = tau
#
#         # ENCODER
#         self.conv1 = nn.Conv2d(in_channels, 32, 7)
#         self.bnorm1 = nn.GroupNorm(4, 32)
#         self.conv2 = nn.Conv2d(32, 64, 5)
#         self.bnorm2 = nn.GroupNorm(4, 64)
#         self.conv3 = nn.Conv2d(64, 128, 3)
#         self.bnorm3 = nn.GroupNorm(8, 128)
#         self.conv4 = nn.Conv2d(128, 256, 3)
#         self.bnorm4 = nn.GroupNorm(8, 256)
#         self.conv5 = nn.Conv2d(256, 256, 3)
#         self.bnorm5 = nn.GroupNorm(8, 256)
#
#         # interp
#         interp = 'nearest'
#         interp_unm = 'nearest'
#
#         # Decoder
#         self.deconv1 = torch.nn.Sequential(nn.Upsample(size=(query_size[0]//4,  query_size[1]//4), mode=interp),
#                                            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
#         self.bnorm6 = nn.GroupNorm(8, 128)
#         self.deconv2 = torch.nn.Sequential(nn.Upsample(size=(query_size[0]//3,  query_size[1]//3),  mode=interp),
#                                            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1))
#         self.bnorm7 = nn.GroupNorm(4, 64)
#         self.deconv3 = torch.nn.Sequential(nn.Upsample(size=(query_size[0],     query_size[1]),     mode=interp),
#                                            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1))
#         self.bnorm8 = nn.GroupNorm(4, 32)
#         self.decoded_image_pixels = torch.nn.Conv2d(32, 1, 1)
#         torch.nn.init.xavier_normal(self.decoded_image_pixels.weight)
#         self.unmaxpool1 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode=interp_unm),
#                                               nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
#         self.unmaxpool2 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode=interp_unm),
#                                               nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
#         self.unmaxpool3 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode=interp_unm),
#                                               nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
#
#         # activations
#         self.maxpool1 = nn.MaxPool2d(2)
#         self.maxpool2 = nn.MaxPool2d(2)
#         self.maxpool3 = nn.MaxPool2d(2)
#         self.maxpool4 = nn.MaxPool2d(2)
#
#         self.act = nn.LeakyReLU(negative_slope=0.3)
#         # self.act = nn.ELU()
#         self.softmax = nn.Softmax(dim=-1)
#
#     def encode(self, x):
#         x = self.act(self.bnorm1(self.conv1(x)))
#         x = self.maxpool1(self.act(self.bnorm2(self.conv2(x))))
#         x = self.maxpool2(self.act(self.bnorm3(self.conv3(x))))
#         x = self.maxpool3(self.act(self.bnorm4(self.conv4(x))))
#         x = self.maxpool4(self.act(self.bnorm5(self.conv5(x))))
#         return x
#
#     def decode(self, x):
#         x = self.act((self.unmaxpool1(x)))
#         x = self.act(self.bnorm6(self.deconv1(x)))
#         x = self.act(self.bnorm7(self.deconv2(x)))
#         x = self.act(self.unmaxpool2(x))
#         x = self.act(self.bnorm8(self.deconv3(x)))
#         return self.decoded_image_pixels(x)
#
#     def update_tau(self, tau):
#         self.tau = tau
#
#     def embed_code(self, embed_id):
#         return F.embedding(embed_id, self.embed.transpose(0, 1))
#
#     def forward(self, x, mask):
#         device = x.device
#
#         x = self.encode(x)
#         x = self.decode(x)
#
#         query_logits = x.view(x.shape[0], -1)
#         query_mask = torch.where(mask == 1, -1e9, 0.) # TODO: Check why.
#         query_logits = query_logits + query_mask.to(device)
#
#         # straight through softmax
#         query = self.softmax(query_logits / self.tau)
#         _, max_ind = (query).max(1)
#         query_onehot = F.one_hot(max_ind, query.shape[1]).type(query.dtype)
#         query = (query_onehot - query).detach() + query
#         return query

class Querier128Flat(nn.Module):
    def __init__(self, tau=1.0, query_size = (26, 26), in_channels=1):
        super().__init__()
        self.tau = tau
        # in_channels += 4
        # in_channels = 1

        # ENCODER
        self.conv1 = nn.Conv2d(in_channels, 32, 3)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bnorm4 = nn.BatchNorm2d(256)

        self.conv5_pool = torch.nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                              nn.Upsample(size=(1, 1), mode='nearest'))
        self.bnorm5 = nn.BatchNorm2d(256)
        self.conv_last = nn.Conv2d(256, query_size[0]*query_size[1], kernel_size=1)

        # activations
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu = nn.LeakyReLU(negative_slope=0.3)
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(0.1)
        # self.register_buffer('pos_enc', self.positionalencoding2d(4, 128, 128, 1))

    def encode(self, x):
        x = self.relu(self.bnorm1(self.conv1(x)))
        x = self.maxpool1(self.relu(self.bnorm2(self.conv2(x))))
        x = self.relu(self.bnorm3(self.conv3(x)))
        x = self.maxpool2(self.relu(self.bnorm4(self.conv4(x))))
        x = self.relu(self.bnorm5(self.conv5_pool(x)))
        return x

    def decode(self, x):
        return self.conv_last(x)

    def update_tau(self, tau):
        self.tau = tau

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def positionalencoding2d(self, d_model, height, width, batch=1):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe = pe.unsqueeze(0)#.repeat_interleave(batch, dim=0)
        return pe

    def forward(self, x, mask, return_attn=False):
        device = x.device

        # x = x[:, :1]
        # x = torch.cat([self.pos_enc.repeat_interleave(x.shape[0], dim=0), x], dim=1)

        # TODO: Add positional encoding.
        x = self.encode(x)
        x = self.decode(x)

        query_logits_pre = x.view(x.shape[0], -1)
        query_mask = torch.where(mask == 1, -1e9, 0.) # TODO: Check why.
        query_logits = query_logits_pre + query_mask.to(device)

        # straight through softmax
        query = self.softmax((query_logits) / self.tau)

        # TODO: Dropout?

        if return_attn:
            query = self.dropout(query)
            _, max_ind = (query).max(1)
            query_onehot = F.one_hot(max_ind, query.shape[1]).type(query.dtype)
            query_out = (query_onehot - query).detach() + query
            query_logits_pre = query_logits_pre - torch.min(query_logits_pre, dim=1, keepdim=True)[0]
            query_logits_pre = query_logits_pre / torch.max(query_logits_pre, dim=1, keepdim=True)[0]
            return query_out, query_logits_pre
        else: query_out = query
        return query_out

class Querier128(nn.Module):
    def __init__(self, tau=1.0, query_size = (26, 26), in_channels=1):
        super().__init__()
        self.tau = tau
        # in_channels += 4
        # in_channels = 1

        # ENCODER
        self.conv1 = nn.Conv2d(in_channels, 32, 3)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bnorm4 = nn.BatchNorm2d(256)

        # Decoder
        self.deconv1 = torch.nn.Sequential(nn.Upsample(size=(5, 5), mode='nearest'),
                                           nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.bnorm5 = nn.BatchNorm2d(256)
        self.deconv2 = torch.nn.Sequential(nn.Upsample(size=(query_size[0]//4,  query_size[1]//4),  mode='nearest'),
                                           nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1))
        self.bnorm6 = nn.BatchNorm2d(64)
        self.deconv3 = torch.nn.Sequential(nn.Upsample(size=(query_size[0],     query_size[1]),     mode='nearest'),
                                           nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1))
        self.bnorm7 = nn.BatchNorm2d(32)
        self.decoded_image_pixels = torch.nn.Conv2d(32, 1, 1)
        self.unmaxpool1 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                              nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.unmaxpool2 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))

        # activations
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu = nn.LeakyReLU(negative_slope=0.3)
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(0.3)
        # self.register_buffer('pos_enc', self.positionalencoding2d(4, 128, 128, 1))

    def encode(self, x):
        x = self.relu(self.bnorm1(self.conv1(x)))
        x = self.maxpool1(self.relu(self.bnorm2(self.conv2(x))))
        x = self.relu(self.bnorm3(self.conv3(x)))
        x = self.maxpool2(self.relu(self.bnorm4(self.conv4(x))))
        return x

    def decode(self, x):
        x = self.relu(self.bnorm5(self.deconv1(x)))
        x = self.relu((self.unmaxpool1(x)))
        x = self.relu(self.bnorm6(self.deconv2(x)))
        x = self.relu(self.unmaxpool2(x))
        x = self.relu(self.bnorm7(self.deconv3(x)))
        return self.decoded_image_pixels(x)

    def update_tau(self, tau):
        self.tau = tau

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def positionalencoding2d(self, d_model, height, width, batch=1):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe = pe.unsqueeze(0)#.repeat_interleave(batch, dim=0)
        return pe

    def forward(self, x, mask, return_attn=False):
        device = x.device

        # x = x[:, :1]
        # x = torch.cat([self.pos_enc.repeat_interleave(x.shape[0], dim=0), x], dim=1)

        # TODO: Add positional encoding.
        x = self.encode(x)
        x = self.decode(x)

        query_logits_pre = x.view(x.shape[0], -1)
        query_mask = torch.where(mask == 1, -1e9, 0.) # TODO: Check why.
        query_logits = query_logits_pre + query_mask.to(device)

        # straight through softmax
        query = self.softmax((query_logits) / self.tau)

        # TODO: Dropout?

        if return_attn:
            query = self.dropout(query)
            _, max_ind = (query).max(1)
            query_onehot = F.one_hot(max_ind, query.shape[1]).type(query.dtype)
            query_out = (query_onehot - query).detach() + query
            query_logits_pre = query_logits_pre - torch.min(query_logits_pre, dim=1, keepdim=True)[0]
            query_logits_pre = query_logits_pre / torch.max(query_logits_pre, dim=1, keepdim=True)[0]
            return query_out, query_logits_pre
        else: query_out = query
        return query_out

class Querier128_with_mask(nn.Module):
    def __init__(self, tau=1.0, query_size = (26, 26), in_channels=1):
        super().__init__()
        self.tau = tau
        mode = 'nearest'


        self.down_block1 = nn.Sequential(
            # nn.Conv2d(1, 32, 3, 1, 1),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # nn.LeakyReLU(negative_slope=0.3)
        )
        self.gain1, self.bias1 = nn.Conv2d(1, 64, 1), nn.Conv2d(1, 64, 1)
        nn.init.zeros_(self.gain1.weight); nn.init.zeros_(self.bias1.weight)
        # self.gain1, self.bias1 = nn.Conv2d(32, 64, 1), nn.Conv2d(32, 64, 1)

        self.down_block2 = nn.Sequential(
            # nn.Conv2d(32, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # nn.LeakyReLU(negative_slope=0.3)
        )
        self.gain2, self.bias2 = nn.Conv2d(1, 256, 1), nn.Conv2d(1, 256, 1)
        nn.init.zeros_(self.gain2.weight); nn.init.zeros_(self.bias2.weight)
        # self.gain2, self.bias2 = nn.Conv2d(64, 256, 1), nn.Conv2d(64, 256, 1)

        self.up_block1 = nn.Sequential(
            nn.Upsample(size=(query_size[0]//4,  query_size[1]//4), mode='bilinear'),
            # nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(negative_slope=0.3)
        )
        self.gain3, self.bias3 = nn.Conv2d(1, 64, 1), nn.Conv2d(1, 64, 1)
        nn.init.zeros_(self.gain3.weight); nn.init.zeros_(self.bias3.weight)
        # self.gain3, self.bias3 = nn.Conv2d(32, 64, 1), nn.Conv2d(32, 64, 1)

        self.up_block2 = nn.Sequential(
            nn.Upsample(size=(query_size[0],  query_size[1]), mode='bilinear'),
            # nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            # nn.LeakyReLU(negative_slope=0.3)
        )
        self.gain4, self.bias4 = nn.Conv2d(1, 32, 1), nn.Conv2d(1, 32, 1)
        nn.init.zeros_(self.gain4.weight); nn.init.zeros_(self.bias4.weight)
        # self.gain4, self.bias4 = nn.Conv2d(16, 32, 1), nn.Conv2d(16, 32, 1)

        # ENCODER
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1, 1)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bnorm4 = nn.BatchNorm2d(256)

        # Decoder
        self.deconv1 = torch.nn.Sequential(nn.Upsample(size=(5, 5), mode=mode),
                                           nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.bnorm5 = nn.BatchNorm2d(256)
        self.deconv2 = torch.nn.Sequential(nn.Upsample(size=(query_size[0]//4,  query_size[1]//4),  mode=mode),
                                           nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1))
        self.bnorm6 = nn.BatchNorm2d(64)
        self.deconv3 = torch.nn.Sequential(nn.Upsample(size=(query_size[0],     query_size[1]),     mode=mode),
                                           nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1))
        self.bnorm7 = nn.BatchNorm2d(32)
        self.decoded_image_pixels = torch.nn.Conv2d(32, 1, 1)
        self.unmaxpool1 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode=mode),
                                              nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.unmaxpool2 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode=mode),
                                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))

        # activations
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu = nn.LeakyReLU(negative_slope=0.3)
        self.softmax = nn.Softmax(dim=-1)

    def condition_layer(self, x, cond, fns=None, mode='masking'):
        if mode == 'modulation' and fns is not None:
            x = x + (x * fns[0](cond) + fns[1](cond))
        elif mode == 'addition' and fns is not None:
            x = x + fns[0](cond)
        elif mode == 'masking' and fns is not None:
            x = x * cond
        else:
            pass
        return x
    # self.out_layers = nn.Sequential(
    #     normalization(self.out_channels, swish=0.0 if use_scale_shift_norm else 1.0),
    #     nn.SiLU() if use_scale_shift_norm else nn.Identity(),
    #     nn.Dropout(p=dropout),
    #     zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
    # )
    #     if emb is not None:
    #     emb_out = self.emb_layers(emb).type(h.dtype)
    #     while len(emb_out.shape) < len(h.shape):
    #         emb_out = emb_out[..., None]
    #     if self.use_scale_shift_norm:
    #         out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
    #         scale, shift = th.chunk(emb_out, 2, dim=1)
    #         h = out_norm(h) * (1 + scale) + shift
    #         h = out_rest(h)
    #     else:
    #         h = h + emb_out
    #         h = self.out_layers(h)
    # else: h = self.out_layers(h)
    # return self.skip_connection(x) + h

    # print(h.shape)
    def encode(self, x, mask):
        mask_1 = self.down_block1(mask)
        mask_2 = self.down_block2(mask_1)

        x = self.relu(self.bnorm1(self.conv1(x)))
        x = self.maxpool1(self.relu(self.bnorm2(self.conv2(x))))
        x = self.condition_layer(x, mask_1, (self.gain1, self.bias1))
        x = self.relu(self.bnorm3(self.conv3(x)))
        x = self.maxpool2(self.relu(self.bnorm4(self.conv4(x))))
        x = self.condition_layer(x, mask_2, (self.gain2, self.bias2))
        return x, mask_2

    def decode(self, x, mask):
        mask_1 = self.up_block1(mask)
        mask_2 = self.up_block2(mask_1)

        x = self.relu(self.bnorm5(self.deconv1(x)))
        x = self.relu((self.unmaxpool1(x)))
        x = self.relu(self.bnorm6(self.deconv2(x)))
        # x = self.condition_layer(x, mask_1, (self.gain3, self.bias3))
        x = self.relu(self.unmaxpool2(x))
        x = self.relu(self.bnorm7(self.deconv3(x)))
        # x = self.condition_layer(x, mask_2, (self.gain4, self.bias4))

        return self.decoded_image_pixels(x)

    def update_tau(self, tau):
        self.tau = tau

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def forward(self, x, mask, return_attn=False):
        device = x.device

        enc_mask = 1 - mask.reshape(-1, 1, *x.shape[-2:])
        # TODO: Add positional encoding.
        x, enc_mask = self.encode(x, enc_mask)
        x = self.decode(x, enc_mask)

        query_logits_pre = x.view(x.shape[0], -1)
        query_mask = torch.where(mask == 1, -1e9, 0.) # TODO: Check why.
        query_logits = query_logits_pre + query_mask.to(device)

        # straight through softmax
        query = self.softmax((query_logits) / self.tau)

        # TODO: Dropout?

        _, max_ind = (query).max(1)
        query_onehot = F.one_hot(max_ind, query.shape[1]).type(query.dtype)
        query_out = (query_onehot - query).detach() + query
        if return_attn:
            query_logits_pre = query_logits_pre - torch.min(query_logits_pre, dim=1, keepdim=True)[0]
            query_logits_pre = query_logits_pre / torch.max(query_logits_pre, dim=1, keepdim=True)[0]
            return query_out, query_logits_pre
        return query_out

class Querier32(nn.Module):
    def __init__(self, tau=1.0, query_size = (26, 26), in_channels=1):
        super().__init__()
        self.tau = tau

        # ENCODER
        self.conv1 = nn.Conv2d(in_channels, 32, 3)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bnorm4 = nn.BatchNorm2d(256)

        # Decoder
        # self.deconv1 = torch.nn.Sequential(nn.Upsample(size=(5, 5), mode='nearest'),
        #                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        # self.bnorm5 = nn.BatchNorm2d(256)
        mode = 'bilinear'
        self.deconv2 = torch.nn.Sequential(nn.Upsample(size=(query_size[0]//4,  query_size[1]//4),  mode=mode),
                                           nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1))
        self.bnorm6 = nn.BatchNorm2d(64)
        self.deconv3 = torch.nn.Sequential(nn.Upsample(size=(query_size[0],     query_size[1]),     mode=mode),
                                           nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1))
        self.bnorm7 = nn.BatchNorm2d(32)
        self.decoded_image_pixels = torch.nn.Conv2d(32, 1, 1)
        self.unmaxpool1 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode=mode),
                                              nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.unmaxpool2 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode=mode),
                                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))

        # activations
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu = nn.LeakyReLU(negative_slope=0.3)
        self.softmax = nn.Softmax(dim=-1)

    def encode(self, x):
        x = self.relu(self.bnorm1(self.conv1(x)))
        x = self.maxpool1(self.relu(self.bnorm2(self.conv2(x))))
        x = self.relu(self.bnorm3(self.conv3(x)))
        x = self.maxpool2(self.relu(self.bnorm4(self.conv4(x))))
        return x

    def decode(self, x):
        # x = self.relu(self.bnorm5(self.deconv1(x)))
        x = self.relu((self.unmaxpool1(x)))
        x = self.relu(self.bnorm6(self.deconv2(x)))
        x = self.relu(self.unmaxpool2(x))
        x = self.relu(self.bnorm7(self.deconv3(x)))
        return self.decoded_image_pixels(x)

    def update_tau(self, tau):
        self.tau = tau

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def forward(self, x, mask, return_attn=False):
        device = x.device

        # TODO: Add positional encoding.
        x = self.encode(x)
        x = self.decode(x)

        query_logits_pre = x.view(x.shape[0], -1)
        query_mask = torch.where(mask == 1, -1e9, 0.) # TODO: Check why.
        query_logits = query_logits_pre + query_mask.to(device)

        # straight through softmax
        query = self.softmax((query_logits) / self.tau)

        # TODO: Dropout?

        _, max_ind = (query).max(1)
        query_onehot = F.one_hot(max_ind, query.shape[1]).type(query.dtype)
        query_out = (query_onehot - query).detach() + query
        if return_attn:
            query_logits_pre = query_logits_pre - torch.min(query_logits_pre, dim=1, keepdim=True)[0]
            query_logits_pre = query_logits_pre / torch.max(query_logits_pre, dim=1, keepdim=True)[0]
            return query_out, query_logits_pre
        return query_out

class QuerierUnet(nn.Module):
    def __init__(self, num_classes=10, tau=1.0, q_size = (26, 26), in_channels=1):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau

        # ENCODER
        self.conv1 = nn.Conv2d(in_channels, 32, 3)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bnorm4 = nn.BatchNorm2d(256)

        # Decoder
        self.deconv1 = torch.nn.Sequential(nn.Upsample(size=(10, 10), mode='nearest'),
                                           nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.bnorm5 = nn.BatchNorm2d(128)
        self.deconv2 = torch.nn.Sequential(nn.Upsample(size=(12, 12), mode='nearest'),
                                           nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1))
        self.bnorm6 = nn.BatchNorm2d(64)
        self.deconv3 = torch.nn.Sequential(nn.Upsample(size=q_size, mode='nearest'),
                                           nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1))
        self.bnorm7 = nn.BatchNorm2d(32)
        self.decoded_image_pixels = torch.nn.Conv2d(32, 1, 1)
        self.unmaxpool1 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                              nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.unmaxpool2 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))

        # activations
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu = nn.LeakyReLU(negative_slope=0.3)
        self.softmax = nn.Softmax(dim=-1)

    def encode(self, x):
        x = self.relu(self.bnorm1(self.conv1(x)))
        x = self.maxpool1(self.relu(self.bnorm2(self.conv2(x))))
        x = self.relu(self.bnorm3(self.conv3(x)))
        x = self.maxpool2(self.relu(self.bnorm4(self.conv4(x))))
        return x

    def decode(self, x):
        x = self.relu((self.unmaxpool1(x)))
        x = self.relu(self.bnorm5(self.deconv1(x)))
        x = self.relu(self.bnorm6(self.deconv2(x)))
        x = self.relu(self.unmaxpool2(x))
        x = self.relu(self.bnorm7(self.deconv3(x)))
        return self.decoded_image_pixels(x)

    def update_tau(self, tau):
        self.tau = tau

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def forward(self, x, mask):
        device = x.device

        # TODO: Add positional encoding.
        x = self.encode(x)
        x = self.decode(x)

        query_logits = x.view(x.shape[0], -1)
        query_mask = torch.where(mask==1, -1e9, 0.)
        query_logits = query_logits + query_mask.to(device)

        # straight through softmax
        query = self.softmax(query_logits / self.tau)
        _, max_ind = (query).max(1)
        query_onehot = F.one_hot(max_ind, query.shape[1]).type(query.dtype)
        query = (query_onehot - query).detach() + query
        return query

# class Querier(nn.Module):
#     def __init__(self, tau=1.0, query_size = (26, 26), in_channels=1):
#         super().__init__()
#         self.tau = tau
#
#         # ENCODER
#         self.conv1 = nn.Conv2d(in_channels, 32, 3)
#         self.bnorm1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, 3)
#         self.bnorm2 = nn.BatchNorm2d(64)
#
#         self.conv3 = nn.Conv2d(64, 128, 3)
#         self.bnorm3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 256, 3)
#         self.bnorm4 = nn.BatchNorm2d(256)
#
#         # Decoder
#         self.deconv1 = torch.nn.Sequential(nn.Upsample(size=(10, 10), mode='nearest'),
#                                            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
#         self.bnorm5 = nn.BatchNorm2d(128)
#         self.deconv2 = torch.nn.Sequential(nn.Upsample(size=(query_size[0]//4,  query_size[1]//4),  mode='nearest'),
#                                            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1))
#         self.bnorm6 = nn.BatchNorm2d(64)
#         self.deconv3 = torch.nn.Sequential(nn.Upsample(size=(query_size[0],     query_size[1]),     mode='nearest'),
#                                            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1))
#         self.bnorm7 = nn.BatchNorm2d(32)
#         self.decoded_image_pixels = torch.nn.Conv2d(32, 1, 1)
#         self.unmaxpool1 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
#                                               nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
#         self.unmaxpool2 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
#                                               nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
#
#         # activations
#         self.maxpool1 = nn.MaxPool2d(2)
#         self.maxpool2 = nn.MaxPool2d(2)
#         self.relu = nn.LeakyReLU(negative_slope=0.3)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def encode(self, x):
#         x = self.relu(self.bnorm1(self.conv1(x)))
#         x = self.maxpool1(self.relu(self.bnorm2(self.conv2(x))))
#         x = self.relu(self.bnorm3(self.conv3(x)))
#         x = self.maxpool2(self.relu(self.bnorm4(self.conv4(x))))
#         return x
#
#     def decode(self, x):
#         x = self.relu((self.unmaxpool1(x)))
#         x = self.relu(self.bnorm5(self.deconv1(x)))
#         x = self.relu(self.bnorm6(self.deconv2(x)))
#         x = self.relu(self.unmaxpool2(x))
#         x = self.relu(self.bnorm7(self.deconv3(x)))
#         return self.decoded_image_pixels(x)
#
#     def update_tau(self, tau):
#         self.tau = tau
#
#     def embed_code(self, embed_id):
#         return F.embedding(embed_id, self.embed.transpose(0, 1))
#
#     def forward(self, x, mask):
#         device = x.device
#
#         # TODO: Add positional encoding.
#         x = self.encode(x)
#         x = self.decode(x)
#
#         query_logits = x.view(x.shape[0], -1)
#         query_mask = torch.where(mask == 1, -1e9, 0.) # TODO: Check why.
#         query_logits = query_logits + query_mask.to(device)
#
#         # straight through softmax
#         query = self.softmax(query_logits / self.tau)
#         _, max_ind = (query).max(1)
#         query_onehot = F.one_hot(max_ind, query.shape[1]).type(query.dtype)
#         query = (query_onehot - query).detach() + query
#         return query

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PositionalEncoding2D(nn.Module):

    def __init__(self, d_model: int, d_spatial: tuple, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.w, self.h = d_spatial
        self.register_buffer('pe', self.positionalencoding2d(d_model, *d_spatial)[None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = torch.cat([x, self.pe.repeat_interleave(x.shape[0], dim=0)], dim=1)
        return self.dropout(x)

    def positionalencoding2d(self, d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe