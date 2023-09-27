import os
import torch
import random
import numpy as np
from PIL import Image

import json



# !/usr/bin/env python
# coding: utf-8
from torchvision import transforms
from datasets import load_dataset
import requests
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from io import BytesIO

# tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="tokenizer") # "runwayml/stable-diffusion-inpainting"

def download_image(url):
	try:
		response = requests.get(url, timeout=1)
		response.raise_for_status()
		image_data = response.content
		image = Image.open(BytesIO(image_data)).convert('RGB')
		return image

	except:
		# print('example fora')
		return None

def process_url_dataset(example):
	image = download_image(example['URL'])
	example['image'] = image
	return example

# def prepare_mask_and_masked_image(image, mask):
# 	image = np.array(image.convert("RGB"))
# 	image = image[None].transpose(0, 3, 1, 2)
# 	image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
#
# 	mask = np.array(mask.convert("L"))
# 	mask = mask.astype(np.float32) / 255.0
# 	mask = mask[None, None]
# 	mask[mask < 0.5] = 0
# 	mask[mask >= 0.5] = 1
# 	mask = torch.from_numpy(mask)
#
# 	masked_image = image * (mask < 0.5)
#
# 	return mask, masked_image

# def generate_mask(image_shape, max_queries, seed=None):  # only resol//n patches version
# 	if seed is not None:
# 		random.seed(seed)
# 	resolution = image_shape[0]
# 	patch_size = resolution // 8
# 	mask = np.ones((resolution, resolution), dtype=np.uint8) * 255
# 	num_patches = np.random.randint(0, max_queries + 1)
# 	for i in range(num_patches):
# 		row = np.random.randint(0, resolution // patch_size)
# 		col = np.random.randint(0, resolution // patch_size)
# 		top = row * patch_size
# 		left = col * patch_size
# 		mask[top:top + patch_size, left:left + patch_size] = 0
# 	return Image.fromarray(mask)


# def collate_fn(examples):
# 	# input_ids = [example["instance_prompt_ids"] for example in examples]
# 	# image_label = [example["image_label"] for example in examples]
# 	pixel_values = [example["instance_images"] for example in examples]
#
#
# 	masks = []
# 	masked_images = []
# 	# for example in examples:
# 	# 	pil_image = example["PIL_images"]
# 	# 	# generate a random mask
# 	# 	# mask = random_mask(pil_image.size, mask_size=args.patch_size)
# 	# 	mask = generate_mask(pil_image.size, max_queries=args.max_queries)
# 	# 	# prepare mask and masked image
# 	# 	mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)
# 	masks.append(mask)
# 	masked_images.append(masked_image)
# 	pixel_values = torch.stack(pixel_values)
# 	pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
# 	input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
# 	masks = torch.stack(masks)
# 	masked_images = torch.stack(masked_images)
# 	batch = {"input_ids": input_ids, "pixel_values": pixel_values, "masks": masks,
# 			 "masked_images": masked_images, "image_label": image_label, "PIL_image": pil_image}
# 	return batch

class CustomHFDataset(Dataset):
	"""
	A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
	It pre-processes the images and the tokenizes prompts.
	"""



	def __init__(
			self,
			data_dir,
			split,
			transform,
			size=512,
			seed = 0,
	):
		# (args.data_dir, split='train', transform=transform)
		self.size = size

		dataset_name = ''
		if split is 'train':
			self.dataset = load_dataset(
				'ChristophSchuhmann/improved_aesthetics_6.5plus',
				split='train',
				cache_dir=data_dir
			).shuffle(seed=42).train_test_split(test_size=0.3, seed=42)['train']

		elif split is 'test':
			self.dataset = load_dataset(
				'ChristophSchuhmann/improved_aesthetics_6.5plus',
				split='train',
				cache_dir=data_dir
			).shuffle(seed=42).train_test_split(test_size=0.3, seed=42)['test']  # ARMAND: has de saber el tamany del dataset
		# ABANS per triar split
		else:
			raise ValueError(
				"Introduce a valid split_name for the dataset [train, test]"
			)

		self.dataset = self.dataset.filter(lambda example: example['URL'].endswith('.jpg')
														   and example['HEIGHT'] is not None
														   and example['WIDTH'] is not None
														   and example['HEIGHT'] == example['WIDTH']
														   and example['HEIGHT'] >= 128
														   and example['HEIGHT'] <= 500
										   )
		##ATENCIÓ AMIC##:
		# El dataset es enorme, filtra pel tamany de les imatges originals per fer un subset. Si vols fer
		# una execució de proba comença amb max height i min height a 410 que hi ha unes 20 imatges només.

		# Downloading each image from the URL described in the dataset
		print('Downloading images...')
		self.dataset = self.dataset.map(
			process_url_dataset
		)

		# Processing the text (tokenize and text encoding)
		'''
		print('Encoding image descriptions...')
		self.dataset = self.dataset.map(
			lambda example: encode_text_description(example, tokenizer=self.tokenizer, text_encoder=self.text_encoder),
			remove_columns=['URL', 'WIDTH', 'HEIGHT', 'similarity', 'punsafe',
							'pwatermark', 'AESTHETIC_SCORE', 'hash', '__index_level_0__']
		)
		'''
		# ATENTIÓ AMIC:
		# Això son els encodings de la descripció. Pots utilitzar-la si fas bs=1, sinó has de encodejar
		# en cada iteració com hem parlat

		print('Dataset has a total of ' + str(len(self.dataset)) + ' samples')

		self.dataset = self.dataset.filter(lambda example: example["image"] is not None)

		print('Dataset has ' + str(len(self.dataset)) + ' valid samples')

		self.num_instance_images = len(self.dataset)
		self._length = self.num_instance_images

		# FF: Cropping has been removed from the compose!!
		self.image_transforms_resize_and_crop = transforms.Compose(
			[
				transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
			]
		)
		self.warn = True
		self.image_transforms = transform
		# transforms.Compose(
		# 	[
		# 		transforms.ToTensor(),
		# 		transforms.Normalize([0.5], [0.5]),
		# 	]
		# )


	def __len__(self):
		return self._length


	def __getitem__(self, index):
		example = {}


		instance_image = self.dataset[index]['image']

		if not instance_image.mode == "RGB":
			instance_image = instance_image.convert("RGB")
		instance_image = self.image_transforms_resize_and_crop(instance_image)

		# example["PIL_images"] = instance_image
		image = self.image_transforms(instance_image)

		# example["instance_prompt_ids"] = self.tokenizer(
		# 	self.dataset[index]['TEXT'],  # FF
		# 	padding="do_not_pad",
		# 	truncation=True,
		# 	max_length=self.tokenizer.model_max_length,
		# ).input_ids

		# example["image_label"] = self.dataset[index]['TEXT']  ##dataset##

		# example['encoder_hidden_states'] = self.dataset[index]['encoder_hidden_states'] #encoded when loading the dataset
		cond_image = image.clone()
		image[0, 0, 0] = self.dataset[index]['hash']
		image[0, 0, 1] = self.dataset[index]['__index_level_0__']
		if self.warn:
			print('Hash and Index passed as pixel value !!! \n\n')
			self.warn=False
		return image, cond_image

# if args.evaluation:
# 	print('Loading test dataset...')
# 	test_dataset = CustomHFDataset(
# 		dataset_name=args.dataset_name,
# 		split_name="Test",
# 		instance_prompt=args.instance_prompt,
# 		tokenizer=tokenizer,
# 		text_encoder=text_encoder,
# 		size=args.resolution,
# 		data_dir=args.data_dir,
# 		seed=args.seed
# 	)
#
# print('Loading train dataset...')
# train_dataset = CustomHFDataset(
# 	dataset_name=args.dataset_name,
# 	split_name="Train",
# 	instance_prompt=args.instance_prompt,
# 	tokenizer=tokenizer,
# 	text_encoder=text_encoder,
# 	size=args.resolution,
# 	data_dir=args.data_dir,
# 	seed=args.seed
# )
#
# train_dataloader = torch.utils.data.DataLoader(
# 	train_dataset, batch_size=args.train_batch_size, shuffle=True
# )
#
# if args.evaluation:
# 	test_dataloader = torch.utils.data.DataLoader(
# 		test_dataset, batch_size=args.test_batch_size, shuffle=False
# 	)
