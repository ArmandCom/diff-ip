import os


os.environ["CUDA_VISIBLE_DEVICES"]='2'


import torch
from imagen_pytorch.imagen_pytorch_ import Unet, Imagen

# unet for imagen
'''
unet1 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
    layer_cross_attns = (False, True, True, True)
)

unet2 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = (2, 4, 8, 8),
    layer_attns = (False, False, False, True),
    layer_cross_attns = (False, False, False, True)
)

# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    unets = (unet1, unet2),
    image_sizes = (64, 256),
    timesteps = 1000,
    cond_drop_prob = 0.1
).cuda()

# mock images (get a lot of this) and text encodings from large T5

#text_embeds = torch.randn(4, 256, 768).cuda()
#images = torch.randn(4, 3, 256, 256).cuda()

# feed images into imagen, training each unet in the cascade

#for i in (1, 2):
#    loss = imagen(images, text_embeds = text_embeds, unet_number = i)
#    loss.backward()

#images = imagen.sample(text_embeds = text_embeds, cond_scale = 3.)
'''

from arch.modules import QuerierCelebAttr
text_embeds = torch.randn(4, 40, 256)
images = torch.randn(4, 3, 64, 64)
attr_mask_wo_query = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
         0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
         0., 0., 0., 0.]])

querier = QuerierCelebAttr(only_image=False)
query_vec, query_logits, attn = querier(image = images, mask = attr_mask_wo_query, query_features = text_embeds)

print("end")