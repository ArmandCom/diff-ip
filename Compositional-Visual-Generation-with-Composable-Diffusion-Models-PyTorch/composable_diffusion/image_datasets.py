import os
import math
import random
import csv

import pickle

import torch
import torch as th

import numpy as np

from pycocotools.coco import COCO
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils import data
from collections import namedtuple

from torchvision.datasets.utils import verify_str_arg

from functools import partial
from typing import Any, Callable, List, Optional, Union, Tuple

from composable_diffusion.download import download_data

# import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


class ClevrDataset(Dataset):
    def __init__(
        self,
        resolution,
        data_path,
        use_captions=False,
        random_crop=False,
        random_flip=False
    ):
        self.resolution = resolution
        self.use_captions = use_captions
        self.random_crop = random_crop
        self.random_flip = random_flip

        data = np.load(data_path)
        print(f'loading data from {data_path}...')
        print(f'using {"captions" if use_captions else "numeric labels"}...')
        self.ims, self.labels = data['ims'], data['labels']

        # caption mapping
        colors_to_idx = {"gray": 0, "red": 1, "blue": 2, "green": 3, "brown": 4, "purple": 5, "cyan": 6, "yellow": 7, "none": 8}
        shapes_to_idx = {"cube": 0, "sphere": 1, "cylinder": 2, "none": 3}
        materials_to_idx = {"rubber": 0, "metal": 1, "none": 2}
        sizes_to_idx = {"small": 0, "large": 1, "none": 2}
        relations_to_idx = {"left": 0, "right": 1, "front": 2, "behind": 3, "below": 4, "above": 5, "none": 6}

        self.label_description = {
            "left": "to the left of",
            "right": "to the right of",
            "behind": "behind",
            "front": "in front of",
            "above": "above",
            "below": "below"
        }

        self.colors = list(colors_to_idx.keys())
        self.shapes = list(shapes_to_idx.keys())
        self.materials = list(materials_to_idx.keys())
        self.sizes = list(sizes_to_idx.keys())
        self.relations = list(relations_to_idx.keys())

        for k,v in data.items():
            print(k)
            try:
                print(v[0].shape)
                print(v[0])
            except:
                pass
        exit()
    def __len__(self):
        return self.ims.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.ims[index]).convert('RGB')
        label = self.labels[index]

        if self.random_crop:
            arr = random_crop_arr(image, self.resolution)
        else:
            arr = center_crop_arr(image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1
        if self.use_captions:
            out_dict = {'caption': self.get_caption(label)}
        else:
            out_dict = {"y": label}
            masks = random.random() > 0.05
            out_dict.update(dict(masks=masks))

        return np.transpose(arr, [2, 0, 1]), out_dict

    def get_caption(self, label):
        text_label = []
        for i in range(2):
            shape, size, color, material, pos = label[i * 5:i * 5 + 5]
            obj = ' '.join([self.sizes[size], self.colors[color],
                            self.materials[material], self.shapes[shape]]).strip()
            text_label.append(obj)
        relation = self.relations[label[-1]]
        if 'none' in relation:
            return text_label[0]
        else:
            return f'{text_label[0]} {self.label_description[relation]} {text_label[1]}'

# import tensorflow.compat.v1 as tf
#
# COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
# IMAGE_SIZE = [240, 320]
# # The maximum number of foreground and background entities in the provided
# # dataset. This corresponds to the number of segmentation masks returned per
# # scene.
# MAX_NUM_ENTITIES = 11
# BYTE_FEATURES = ['mask', 'image', 'color', 'material', 'shape', 'size']
#
# # Create a dictionary mapping feature names to `tf.Example`-compatible
# # shape and data type descriptors.
# features = {
#     'image': tf.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
#     'mask': tf.FixedLenFeature([MAX_NUM_ENTITIES]+IMAGE_SIZE+[1], tf.string),
#     'x': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
#     'y': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
#     'z': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
#     'pixel_coords': tf.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
#     'rotation': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
#     'size': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
#     'material': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
#     'shape': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
#     'color': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
#     'visibility': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
# }

import functools
class Clevr2DPosDataset(Dataset):
    def __init__(
            self,
            resolution,
            data_path,
            use_captions=False,
            random_crop=False,
            random_flip=False,
    ):
        self.resolution = resolution
        self.use_captions = use_captions
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.generate_dataset = False
        max_num = 5
        if self.generate_dataset:
            dataset = self.dataset(data_path)
            # length_dataset = len(list(dataset))
            # print(length_dataset)
            dict = {}
            valid = 0
            for i, elem in enumerate(dataset.as_numpy_iterator()):
                if elem['visibility'].sum() > (max_num + 1):
                    continue
                else:
                    valid += 1
                for k,v in elem.items():
                    if k in dict.keys():
                        dict[k].append(v)
                    else:
                        dict[k] = [v]
                if valid % 1000 == 0:
                    print('Num processed: ', valid)
                if valid == 10000 - 1:
                    break
            out_dict = {}
            for k,v in dict.items():
                out_dict[k] = np.stack(v, axis=0)
            np.savez(data_path[:-10]+'.npz',
                     image=out_dict['image'],
                     mask=out_dict['mask'],
                     pixel_coords=out_dict['pixel_coords'],
                     x=out_dict['x'],
                     y=out_dict['y'],
                     z=out_dict['z'])

            print('Done generating dataset!')
            exit()


        # if data_path[-9:] == 'tfrecords':
        #     data_path = data_path[:-9]+'npz'

        data = np.load(data_path)
        print(f'loading data from {data_path}...')
        print(f'using {"captions" if use_captions else "numeric labels"}...')

        self.ims, self.labels = data['ims'], data['coords_labels']
        # self.ims, self.masks = data['image'], data['mask']

    def __len__(self):
        return self.ims.shape[0]

    # def __getitem__(self, index):
    #     for index in range(20):
    #         image = Image.fromarray(self.ims[index+3]).convert('RGB')
    #         image.save('sample_image{}.png'.format(index))
    #     exit()
    #     mask = Image.fromarray( # Note: element 0 is the background.
    #         255 - self.masks[index][0, ..., 0],
    #             mode='L')
    #     mask.save('sample_mask.png')
    #     if self.random_crop:
    #         mask = random_crop_arr(mask, self.resolution) # Res set to 128
    #         arr = random_crop_arr(image, self.resolution)
    #     else:
    #         mask = center_crop_arr(mask, self.resolution)
    #         arr = center_crop_arr(image, self.resolution)
    #
    #     if self.random_flip and random.random() < 0.5:
    #         arr = arr[:, ::-1]
    #         mask = mask[:, ::-1]
    #
    #     image = Image.fromarray(arr).convert('RGB')
    #     image.save('sample_image_cropped.png')
    #     exit()
    #     arr = arr.astype(np.float32) / 127.5 - 1
    #     mask = mask.astype(np.float32) / 127.5 - 1
    #
    #     return np.transpose(arr, [2, 0, 1]), np.transpose(mask, [2, 0, 1])

    def _decode(self, example_proto):
        # Parse the input `tf.Example` proto using the feature description dict above.
        single_example = tf.io.parse_single_example(example_proto, features)
        for k in BYTE_FEATURES:
            single_example[k] = tf.squeeze(tf.io.decode_raw(single_example[k], tf.uint8),
                                           axis=-1)
        return single_example

    def dataset(self, tfrecords_path, read_buffer_size=None, map_parallel_calls=None):
        """Read, decompress, and parse the TFRecords file.

        Args:
          tfrecords_path: str. Path to the dataset file.
          read_buffer_size: int. Number of bytes in the read buffer. See documentation
            for `tf.data.TFRecordDataset.__init__`.
          map_parallel_calls: int. Number of elements decoded asynchronously in
            parallel. See documentation for `tf.data.Dataset.map`.

        Returns:
          An unbatched `tf.data.TFRecordDataset`.
        """
        print(tfrecords_path)
        # read_buffer_size = 64
        raw_dataset = tf.data.TFRecordDataset(
            tfrecords_path, compression_type=COMPRESSION_TYPE,
            buffer_size=read_buffer_size)
        # for elem in raw_dataset.take(10):
        #     print(repr(elem))
        return raw_dataset.map(self._decode, num_parallel_calls=map_parallel_calls)

    def __getitem__(self, index):
        image = Image.fromarray(self.ims[index]).convert('RGB')
        label = self.labels[index]

        if self.random_crop:
            arr = random_crop_arr(image, self.resolution)
        else:
            arr = center_crop_arr(image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1
        if self.use_captions:
            out_dict = {'caption': f"[{label[0]:.2f}, {label[1]:.2f}]"}
        else:
            out_dict = {"y": label}
            masks = random.random() > 0.1
            out_dict.update(dict(masks=masks))

        return np.transpose(arr, [2, 0, 1]), out_dict

class CocoDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, resolution, random_crop=False, random_flip=False):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.random_crop:
            arr = random_crop_arr(image, self.resolution)
        else:
            arr = center_crop_arr(image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1
        out_dict = {'caption': caption}

        return arr, out_dict

    def __len__(self):
        return len(self.ids)


def load_data(
    *,
    root,
    split,
    dataset_type,
    batch_size,
    image_size,
    use_captions=False,
    deterministic=False,
    random_crop=False,
    random_flip=False
):
    if not root:
        raise ValueError("unspecified data directory")
    if dataset_type == 'clevr_pos_multiple':
        data_path = download_data(dataset='clevr_pos', cache_dir=root)
    else: data_path = download_data(dataset=dataset_type, cache_dir=root)
    # data_path = './dataset/clevr_with_masks_train.tfrecords'
    if dataset_type == 'clevr_rel':
        dataset = ClevrDataset(
            resolution=image_size,
            data_path=data_path,
            use_captions=use_captions,
            random_crop=random_crop,
            random_flip=random_flip
        )
    elif dataset_type == 'clevr_pos':
        dataset = Clevr2DPosDataset(
            resolution=image_size,
            data_path=data_path,
            use_captions=use_captions,
            random_crop=random_crop,
            random_flip=random_flip
        )
    elif dataset_type == 'clevr_pos_multiple':
        dataset = Clevr2DPosMultipleDataset(
            resolution=image_size,
            data_path=data_path,
            use_captions=use_captions,
            random_crop=random_crop,
            random_flip=random_flip,
            max_objects=5
        )
    elif dataset_type == 'coco':
        # specify the root path and json path
        root = None
        json = None
        dataset = CocoDataset(
            root,
            json,
            resolution=image_size,
            random_crop=random_crop,
            random_flip=random_flip
        )
    else:
        raise NotImplementedError(f'dataset: {dataset_type} is not implemented.')

    num_workers = 0
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
    while True:
        yield from loader
