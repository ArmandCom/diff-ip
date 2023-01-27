import os
import torch
import random
import numpy as np
from PIL import Image

import json
class Clevr_with_masks(torch.utils.data.Dataset):
	def __init__(self, img_dir, split, transform=None, max_num=None, perc_train=0.8):

		self.img_dir = img_dir
		self.transform = transform
		self.image_paths = os.listdir(self.img_dir)
		self.image_paths = [path for path in self.image_paths if not path.endswith('mask.png')]
		if max_num is not None:
			self.image_paths = self.image_paths[:max_num]

		random.shuffle(self.image_paths)
		num_train = int(perc_train * len(self.image_paths))
		if split == 'train':
			self.train = True
			self.image_paths = self.image_paths[:num_train]
		elif split == 'test':
			self.train = False
			self.image_paths = self.image_paths[num_train:]

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):

		# good = False
		# while not good:
		# 	img_path = os.path.join(self.img_dir, self.image_paths[idx])
		# 	try:
		# 		image = (np.array(Image.open(img_path)) / 255)
		# 	except:
		# 		idx = (idx + 1) % len(self.image_paths)
		# 		continue
		# 	good = True

		img_path = os.path.join(self.img_dir, self.image_paths[idx])
		image = (np.array(Image.open(img_path)) / 255)
		mask_path = os.path.join(self.img_dir, self.image_paths[idx].rstrip('.png') + '_mask.png')
		mask = (np.array(Image.open(mask_path)) / 255)
		mask[mask!=1] = 0
		if self.transform:
			image = self.transform(image).float()
			mask = self.transform(mask).float()

		# scene_path = os.path.join(self.img_dir.rstrip('images/'), 'scenes', self.image_paths[idx].rstrip('.png') + '.json')

		# s = open(scene_path)
		# scene = json.load(s)

		return image[:3], torch.cat([torch.clamp(mask[:1], 0, 1), image[:3]])

class Clevr_with_attr(torch.utils.data.Dataset):
	def __init__(self, img_dir, split, attribute='color', max_attributes=5, transform=None, max_num=None, perc_train=0.8):

		self.attribute = attribute
		self.max_attributes = max_attributes

		self.img_dir = img_dir
		self.transform = transform
		self.image_paths = os.listdir(self.img_dir)
		self.image_paths = [path for path in self.image_paths if not path.endswith('mask.png')]
		if max_num is not None:
			self.image_paths = self.image_paths[:max_num]

		if attribute == 'color':
			self.attribute_list = ['gray', 'blue', 'brown', 'yellow', 'red', 'green', 'purple', 'cyan']
		elif attribute == 'shape':
			self.attribute_list = ['cube', 'sphere', 'cylinder']
		else: raise NotImplementedError
		# self.materials = []

		random.shuffle(self.image_paths)
		num_train = int(perc_train * len(self.image_paths))
		if split == 'train':
			self.train = True
			self.image_paths = self.image_paths[:num_train]
		elif split == 'test':
			self.train = False
			self.image_paths = self.image_paths[num_train:]

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):

		# good = False
		# while not good:
		# 	img_path = os.path.join(self.img_dir, self.image_paths[idx])
		# 	try:
		# 		image = (np.array(Image.open(img_path)) / 255)
		# 	except:
		# 		idx = (idx + 1) % len(self.image_paths)
		# 		continue
		# 	good = True

		img_path = os.path.join(self.img_dir, self.image_paths[idx])
		image = (np.array(Image.open(img_path)) / 255)
		# mask_path = os.path.join(self.img_dir, self.image_paths[idx].rstrip('.png') + '_mask.png')
		# mask = (np.array(Image.open(mask_path)) / 255)
		# mask[mask!=1] = 0
		if self.transform:
			image = self.transform(image).float()
			# mask = self.transform(mask).float()

		scene_path = os.path.join(self.img_dir.rstrip('images/'), 'scenes', self.image_paths[idx].rstrip('.png') + '.json')

		s = open(scene_path)
		scene = json.load(s)

		atts = torch.zeros((self.max_attributes,), dtype=torch.int32)
		for i, object in enumerate(scene['objects']):
			atts[i] = self.attribute_list.index(object[self.attribute]) + 1
		return image[:3], atts
