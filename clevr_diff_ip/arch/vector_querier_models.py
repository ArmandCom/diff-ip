import torch
import torch.utils.data
import torchvision as tv
from torch import nn
from torch.nn import functional as F
import pdb
import utils
import math
from flamingo_pytorch import PerceiverResampler, GatedCrossAttentionBlock
# from vit_pytorch.vit import ViT
# from vit_pytorch.extractor import Extractor
from transformers import ViTModel, ViTFeatureExtractor

# from modules import UNet_conditional, SAUnet

# encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
# transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
# src = torch.rand(10, 32, 512)
# out = transformer_encoder(src)
import random
import numpy as np

# tv.models.vit_b_16(*, weights: Optional[ViT_B_16_Weights] = None, progress: bool = True, **kwargs: Any) → VisionTransformer
# def __init__(
#         self,
#         image_size: int,
#         patch_size: int,
#         num_layers: int,
#         num_heads: int,
#         hidden_dim: int,
#         mlp_dim: int,
#         dropout: float = 0.0,
#         attention_dropout: float = 0.0,
#         num_classes: int = 1000,
#         representation_size: Optional[int] = None,
#         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#         conv_stem_configs: Optional[List[ConvStemConfig]] = None,
# ):
# class FeatureExtractor(nn.Module):
#     def __init__(self):
#
#     def forward(self, x):
#         return x


class Querier(nn.Module):
    def __init__(self, tau=1.0, num_queries=100, query_dim=64, in_channels=1, num_layers=6, num_heads=8, dropout=0.0):
        super().__init__()
        self.tau = tau

        # Step 1: Encode Information about the image
        # self.encode_image = FeatureExtractor()
        # self.perceive = PerceiverResampler( # medias = torch.randn(1, 2, 256, 1024) # (batch, time, sequence length, dimension)
        #     dim = 1024,
        #     depth = 2,
        #     dim_head = 64,
        #     heads = 8,
        #     num_latents = 64,    # the number of latents to shrink your media sequence to, perceiver style
        #     num_media_embeds = 1  # say you have 4(1 in this case) images maximum in your dialogue
        # )
        #
        # self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        # self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        # # Example
        # # inputs = self.feature_extractor(image, return_tensors="pt") # Note: Do in loader
        # # with torch.no_grad():
        # #     outputs = model(**inputs)


        self.softmax = nn.Softmax(dim=-1)

        # Create embeddings for each attribute
        self.embedding = nn.Embedding(num_queries, query_dim) # max_norm=True ?

        # Transformer (group embeddings)

        encoder_layer = nn.TransformerEncoderLayer(d_model=query_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        # src = torch.rand(10, 32, 512)
        # out = self.transformer_encoder(src)

        self.query_dim = query_dim

    def update_tau(self, tau):
        self.tau = tau

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embedding.transpose(0, 1))

    def forward(self, x, mask, return_attn = False):
        device = x.device

        print(x.shape)
        # inputs = self.feature_extractor(x[:, :3], return_tensors="pt")
        # print(inputs)
        # input('s')
        with torch.no_grad():
            outputs = self.vit(**inputs)
        print(outputs.shape)
        exit()
        # x = self.encode_image(x)
        # print(x.shape)
        x = self.perceive(x)
        print(x.shape)
        exit()

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

class QueryEncoder(nn.Module):
    def __init__(self, query_dim=64, num_layers=6, num_heads=8, dropout=0.0):
        super().__init__()

        self.query_dim = query_dim
        self.mode = 'avg_pool'
        encoder_layer = nn.TransformerEncoderLayer(d_model=query_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # # Transformer usage:
        # # src = torch.rand(10, 32, 512)
        # # out = self.transformer_encoder(src)

    def forward(self, x):
        out = self.transformer_encoder(x)
        # TODO: Check implemented architectures with CLS
        #  1: Add CLS token.
        #  2: Check gradients.
        # Note: Average for now.
        if self.mode == 'avg_pool':
            out = out.mean(1)
        return out

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