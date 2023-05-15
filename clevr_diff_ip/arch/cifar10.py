import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import pdb
import utils

class QuerierCIFAR(nn.Module):
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

class QuerierAE(nn.Module):
    def __init__(self, c_in, c_out=1, size=60, out_size=None, patch_size=5, reduction=16):
        # (self, num_classes=10, tau=1.0, q_size=(26, 26), in_channels = 1):

        super().__init__()

        if out_size is None:
            out_size = size - patch_size + 1
        self.tau = 1

        # ENCODER
        self.conv1 = nn.Conv2d(c_in, 32, 3)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bnorm4 = nn.BatchNorm2d(256)

        # Decoder
        self.deconv1 = torch.nn.Sequential(
            # nn.Upsample(size=(10, 10), mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.bnorm5 = nn.BatchNorm2d(128)
        self.deconv2 = torch.nn.Sequential(
            # nn.Upsample(size=(12, 12), mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1))
        self.bnorm6 = nn.BatchNorm2d(64)
        self.deconv3 = torch.nn.Sequential(nn.Upsample(size=out_size, mode='nearest'),
                                           nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1))
        self.bnorm7 = nn.BatchNorm2d(32)
        self.decoded_image_pixels = torch.nn.Conv2d(32, 1, 1, bias=False)
        self.unmaxpool1 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                              nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.unmaxpool2 = torch.nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                              nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))

        # activations
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(2)

        self.relu = nn.LeakyReLU(negative_slope=0.3)
        self.softmax = nn.Softmax(dim=-1)
        # self.decoded_image_pixels.weight.data.fill_(0.01)

        self.tau=1
    def encode(self, x):
        x = self.relu(self.bnorm1(self.conv1(x)))
        x = self.maxpool1(self.relu(self.bnorm2(self.conv2(x))))
        x = self.maxpool2(self.relu(self.bnorm3(self.conv3(x))))
        x = self.maxpool3(self.relu(self.bnorm4(self.conv4(x))))
        return x

    def decode(self, x):
        x = self.relu((self.unmaxpool1(x)))
        x = self.relu(self.bnorm5(self.deconv1(x)))
        x = self.relu(self.unmaxpool2(x))
        x = self.relu(self.bnorm6(self.deconv2(x)))
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

        query_mask = torch.where(mask == 1, -1e9, torch.zeros((1,)).to(x.device))  # TODO: Check why.
        # identity_mask = torch.zeros_like(mask).reshape(*x.shape)
        # identity_mask[..., x.shape[-2] // 2, x.shape[-1] // 2] = 1
        # query_mask += identity_mask.reshape(*query_mask.shape)

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
            return query_out, query, query_logits_pre

        return query_out, query