import torch
import torch.utils.data
import torchvision as tv
from torch import nn
from torch.nn import functional as F
import pdb
import utils
import math

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


# class QueryEncoder(nn.Module):
#     def __init__(self, query_dim=64, num_layers=3, num_heads=5, dropout=0.0):
#         super().__init__()
#
#         self.query_dim = query_dim
#         encoder_layer = nn.TransformerEncoderLayer(d_model=query_dim, nhead=num_heads)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         # # Transformer usage:
#         # # src = torch.rand(10, 32, 512)
#         # # out = self.transformer_encoder(src)
#
#     def forward(self, x, reduce='avg_pooling'):
#         out = self.transformer_encoder(x)
#         # Note: Average for now.
#         if reduce == 'avg_pooling':
#             out = out.mean(1)
#         elif reduce == 'max_pooling':
#             out = out.max(1)[0]
#         elif reduce == 'cls':
#             out = out[:, 0]
#         else: raise NotImplementedError
#         return out

#TODO: vertical + horizontal Linears intercalated.
class QueryLinearEncoder(nn.Module):
    def __init__(self, attr_dim, n_obj, out_dim, embed_dim, reduce=None, add_answer='keep_pos'):
        super().__init__()
        self.reduce = reduce
        self.add_answer = add_answer
        self.attr_dim = attr_dim
        self.n_obj = n_obj
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        if self.reduce == 'linear_all':
            self.fc = nn.Linear(attr_dim * n_obj * embed_dim, out_dim)
        elif self.reduce == 'linear_per_filter':
            self.fc = nn.Linear(attr_dim * n_obj, 1)
        elif self.reduce == 'linear_per_attr':
            self.fc = nn.Linear(attr_dim, 1)
        if self.add_answer == 'linear_merge':
            self.ans_fc = nn.Linear(1, embed_dim, bias=False)

        self.act = nn.SiLU()

    def forward(self, x, ans=None, eps=1e-8):
        if ans is not None:
            if self.add_answer=='keep_pos': #TODO: Second condition hardcoded
                ans[ans != 1] = 0
                x = x * ans
            elif self.add_answer == 'pos_neg':
                ans_neg = torch.zeros_like(ans)
                ans_pos = torch.zeros_like(ans)
                ans_neg[ans == -1] = 1
                ans_pos[ans == 1] = 1
                if isinstance(x, tuple):
                    x_pos, x_neg = x[0] * ans_pos, x[1] * ans_neg
                else:
                    x_pos = x * ans_pos
                    x_neg = -(x * ans_neg)# ? Note: not the correct way of implementing it. They cancel out.
                    raise NotImplementedError
            elif self.add_answer == 'linear_merge':
                x = x + self.ans_fc(ans)
            else: raise NotImplementedError
        if self.reduce == 'linear_all':
            x = x.reshape(x.shape[0], -1, self.n_obj * self.attr_dim * self.embed_dim)
            out = self.act(self.fc(x)).reshape(x.shape[0], -1)
        elif self.reduce == 'linear_per_filter':
            x = x.reshape(x.shape[0], -1, self.embed_dim).permute(0, 2, 1)
            out = self.act(self.fc(x)).reshape(x.shape[0], -1)
        elif self.reduce == 'linear_per_attr':
            x = x.reshape(x.shape[0], self.attr_dim, self.n_obj, self.embed_dim).permute(0, 2, 3, 1)
            x = x.sum(1)
            out = self.act(self.fc(x)).reshape(x.shape[0], -1)
        elif self.reduce == 'max_pooling':
            x = x.reshape(x.shape[0], -1, x.shape[-1])
            out = x.max(-2)[0] # Note: could this work? Not for repeated attrs
        elif self.reduce == 'avg_pooling':
            x = x.reshape(x.shape[0], -1, x.shape[-1])
            out = x.mean(-2)
        elif self.reduce == 'pos_avg_pooling':
            x = x.reshape(x.shape[0], -1, x.shape[-1])
            ans = ans.reshape(ans.shape[0], -1, ans.shape[-1])
            out = x.sum(-2) / (ans.sum(-2) + eps)
        elif self.reduce == 'pos_sum_pooling':
            x = x.reshape(x.shape[0], -1, x.shape[-1])
            # ans = ans.reshape(ans.shape[0], -1, ans.shape[-1])
            out = x.sum(-2)
        elif self.reduce == 'pos_neg_sum_max_pooling': # used to be posneg_sum_max_pooling
            x_pos = x_pos.reshape(-1, self.attr_dim, self.n_obj, self.embed_dim).sum(2)
            x_neg = x_neg.reshape(-1, self.attr_dim, self.n_obj, self.embed_dim).max(2)[0]
            x = torch.cat([x_pos, x_neg], dim=1).sum(1)
            out = x.reshape(x.shape[0], self.embed_dim)
        elif self.reduce == 'pos_neg_sum_max_pooling': # used to be posneg_sum_max_pooling
            x_pos = x_pos.reshape(-1, self.attr_dim, self.n_obj, self.embed_dim).sum(2)
            x_neg = x_neg.reshape(-1, self.attr_dim, self.n_obj, self.embed_dim).max(2)[0]
            x = torch.cat([x_pos, x_neg], dim=1).max(1)[0]
            out = x.reshape(x.shape[0], self.embed_dim)
        return out

class QueryEncoder(nn.Module):
    def __init__(self, attr_dim, n_obj, out_dim, embed_dim, reduce=None, use_answers=False, add_answer='keep_pos'):
        super().__init__()
        self.reduce = reduce
        self.add_answer = add_answer
        self.attr_dim = attr_dim
        self.n_obj = n_obj
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.use_answers = use_answers
        if not use_answers:
            if self.reduce == 'linear_all':
                self.fc = nn.Linear(attr_dim * n_obj * embed_dim, out_dim)
            elif self.reduce == 'linear_per_filter':
                self.fc = nn.Linear(attr_dim * n_obj, 1)
            elif self.reduce == 'linear_per_attr':
                self.fc = nn.Linear(attr_dim, 1)
            elif self.reduce == 'pos_neg_sum_max_pooling':
                pass
            if self.add_answer == 'linear_merge':
                self.ans_fc = nn.Linear(1, embed_dim, bias=False)
            self.act = nn.SiLU()

        self.cond_embedding_nobj = nn.Embedding(n_obj,
                                                embed_dim)  # , padding_idx=0 , max_norm=True #TODO: review the in_channels

    def object_embedding(self, batch_size, device):
        return self.cond_embedding_nobj(torch.arange(self.n_obj).to(device))[None, None, :, :] \
            .repeat(batch_size, self.attr_dim, 1, 1).reshape(-1, self.attr_dim * self.n_obj, self.embed_dim)
    def forward(self, cond, ans=None, cond_att=None, ans_att=None):
        bs = cond[0].shape[0]
        if not self.use_answers:
            x = cond
            if self.reduce == 'flatten':
                x_pos, x_neg = x[0], x[1]
                x_all = x_pos + x_neg
                if len(x) == 3:
                    x_all = x_all + x[2]
                x_all = x_all.reshape(bs, self.attr_dim * self.n_obj * self.embed_dim)
                out = x_all.reshape(x_all.shape[0], -1)

            elif self.reduce == 'flatten_obj':
                x_pos, x_neg = x[0], x[1]
                x_all = x_pos + x_neg
                if len(x) == 3:
                    x_all = x_all + x[2]
                # torch.cat([x_pos, x_neg], dim=1) # There shouldn't be overlap
                out = x_all.reshape(bs, self.attr_dim, -1)

            elif self.reduce == 'single_queries':
                device, batch_size = x[0].device, x[0].shape[0]

                x_pos, x_neg = x[0], x[1]
                x_all = x_pos + x_neg
                if len(x) == 3:
                    x_all = x_all + x[2]
                obj_embed = self.object_embedding(batch_size, device)
                x_all_o = torch.cat([x_all, obj_embed], dim=-1) # TODO: unasked xs are also given the object embedding. not a problem at the moment because we use a mask for the conditions, but might be in the future
                out = x_all_o.reshape(bs, self.attr_dim * self.n_obj, -1)

                if cond_att is not None:
                    x_pos, x_neg = cond_att[0], cond_att[1]
                    x_all = x_pos + x_neg
                    out = torch.cat([out, x_all.reshape(bs, 1, -1)], dim=1)
                    # TODO: if elements in batch are zeroed, zero this one too.
                    #  But this only matters when training the diffuser.

        else:
            if ans_att is not None:
                raise NotImplementedError
            x = ans
            x_pos, x_neg = x[0], x[1]
            x_all = x_pos + x_neg
            if len(x) == 3:
                x_all = x_all + x[2]
            out = x_all.reshape(-1, self.attr_dim * self.n_obj * 1)
        return out


def answer_queries(q, gt_attrs, all_a=None):
    '''
    :param q: [q x num_attr x max_obj], binary
    :param gt_attrs: [b x max_obj]
    :return:
    '''
    add_nobj = True
    batch, nat, nobj = q.shape
    if add_nobj:
        gtat = gt_attrs.clone()
        gtat[gtat != 0] = 1
        nobj = gtat.sum(1)
    if all_a is None:
        all_a = torch.zeros_like(q)
        for b in range(batch):
            if gt_attrs[b].sum() > 0:
                bins = torch.bincount(gt_attrs[b], minlength=nat+1)[1:] # works with 1-d vectors
                for at_id in range(nat):
                    n_instances = bins[at_id]
                    all_a[b, at_id, :n_instances] = 1
                    all_a[b, at_id, n_instances:] = -1
            if add_nobj:
                all_a[b, -1, :nobj[b]] = 1
                all_a[b, -1, nobj[b]:] = -1

    # else: all_a = all_a.reshape(batch, nat, nobj)

    return all_a.reshape(batch, -1, 1).type(torch.cuda.FloatTensor)

def answer_single_query(q, gt_attrs_rem):
    '''
    :param q: [q x num_attr x max_obj], binary
    :param gt_attrs: [b x max_obj]
    :return:
    '''

    batch, _, _ = q.shape
    # q = q.reshape(batch, -1)
    a = torch.zeros_like(q[:, 0, 0])
    chosen_attrs = a.clone()
    for b in range(batch):
        v, vid = q[b].max(1)[0].max(0)
        if (vid+1) in gt_attrs_rem[b]:
            a[b] = 1
            id = torch.abs(gt_attrs_rem[b] - (vid+1)).min(0)[1]
            gt_attrs_rem[b, id] = 0
        else:
            a[b] = -1
        chosen_attrs[b] = vid + 1
    return a.reshape(batch, 1, 1), chosen_attrs.reshape(batch, 1), gt_attrs_rem.reshape(batch, -1)

def answer_queries_hierarchical(q, gt_attrs):
    '''
    :param q: [q x num_attr x max_obj], binary
    :param gt_attrs: [b x max_obj]
    :return:
    '''
    a = torch.zeros_like(q)
    attrs = a.clone()
    batch, nat, nobj = q.shape
    for b in range(batch):
        for at_id in range(nat):
            for obj_id in range(nobj):
                v = q[b, at_id, obj_id]
                if v == 1:
                    if ((at_id + 1) in gt_attrs[b]): # We discard object 0 which stands for nothing
                        a[b, at_id, obj_id] = 1
                        id = torch.abs(gt_attrs[b] - (at_id + 1)).min(0)[1]
                        attrs[b, at_id, obj_id] = gt_attrs[b, id]
                        gt_attrs[b, id] = 0 # We change only the value of the first element equal to that value
                    else: a[b, at_id, obj_id] = -1
    # TODO: answer si: sum object embeddings in each attribute. Then max_pool in attribute dimension.
    #       answer no: max_pool in object dimension and max_pool in attribute dimension
    #       sum all
    return attrs.reshape(batch, -1), a.reshape(batch, -1, 1).type(torch.cuda.FloatTensor), gt_attrs.reshape(batch, -1)

def answer_single_query_hierarchical(q, gt_attrs_rem):
    '''
    :param q: [q x num_attr x max_obj], binary
    :param gt_attrs: [b x max_obj]
    :return:
    '''

    batch, _, _ = q.shape
    # q = q.reshape(batch, -1)
    a = torch.zeros_like(q[:, 0, 0])
    chosen_attrs = a.clone()
    for b in range(batch):
        v, vid = q[b].max(1)[0].max(0)
        if (vid+1) in gt_attrs_rem[b]:
            a[b] = 1
            id = torch.abs(gt_attrs_rem[b] - (vid+1)).min(0)[1]
            gt_attrs_rem[b, id] = 0
        else:
            a[b] = -1
        chosen_attrs[b] = vid + 1
    return a.reshape(batch, 1, 1), chosen_attrs.reshape(batch, 1), gt_attrs_rem.reshape(batch, -1)


if __name__ == '__main__':

    # Test
    # q = torch.randint(0, 2, (1, 4, 3))
    q = torch.ones((1, 4, 3))
    gt_attrs = torch.tensor([[1,1,3]])
    print(q, '\n', gt_attrs)
    answer_queries(q, gt_attrs)