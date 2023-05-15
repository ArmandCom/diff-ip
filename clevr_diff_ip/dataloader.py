import os
import torch
import random
import numpy as np
from PIL import Image

import json

# Python program to store list to file using pickle module
import pickle

# write list to binary file

class Clevr(torch.utils.data.Dataset):
	def __init__(self, img_dir, split, transform=None, max_num=None, perc_train=0.8):

		self.img_dir = os.path.join(img_dir, 'clevr-dataset-gen/output/images/')
		self.transform = transform
		self.image_paths = os.listdir(self.img_dir)
		self.image_paths = [path for path in self.image_paths if not path.endswith('mask.png')]
		if max_num is not None:
			self.image_paths = self.image_paths[:max_num]

		random.shuffle(self.image_paths)
		# random.Random(4).shuffle(self.image_paths)
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

		img_path = os.path.join(self.img_dir, self.image_paths[idx])
		image = (np.array(Image.open(img_path)) / 255)
		# mask_path = os.path.join(self.img_dir, self.image_paths[idx].rstrip('.png') + '_mask.png')
		# mask = (np.array(Image.open(mask_path)) / 255)
		# mask[mask!=1] = 0

		if self.transform:
			image = self.transform(image).float()[:3]
		# mask = self.transform(mask).float()

		return image, image


class Clevr_with_masks(torch.utils.data.Dataset):
	def \
			__init__(self, img_dir, split, transform=None, max_num=None, perc_train=0.8):

		self.img_dir = os.path.join(img_dir, 'clevr-dataset-gen/output/images/')
		self.transform = transform
		self.image_paths = os.listdir(self.img_dir)
		self.image_paths = [path for path in self.image_paths if not path.endswith('mask.png')]
		if max_num is not None:
			self.image_paths = self.image_paths[:max_num]

		random.shuffle(self.image_paths)
		# random.Random(4).shuffle(self.image_paths)
		num_train = int(perc_train * len(self.image_paths))
		if split == 'train':
			with open(os.path.join(self.img_dir, "shuffled_data_ids.json"), 'w') as f:
				json.dump(self.image_paths, f, indent=2)
			self.train = True
			self.image_paths = self.image_paths[:num_train]
		elif split == 'test':
			with open(os.path.join(self.img_dir, "shuffled_data_ids.json"), 'r') as f:
				self.image_paths = json.load(f)
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
		good = False
		while not good:
			img_path = os.path.join(self.img_dir, self.image_paths[idx])
			try:
				img_path = os.path.join(self.img_dir, self.image_paths[idx])
				image = (np.array(Image.open(img_path)) / 255)
				mask_path = os.path.join(self.img_dir, self.image_paths[idx].rstrip('.png') + '_mask.png')
				mask = (np.array(Image.open(mask_path)) / 255)
				mask[mask != 1] = 0
			except:
				idx = (idx + 1) % len(self.image_paths)
				continue
			good = True

		if self.transform:
			image = self.transform(image).float()
			mask = self.transform(mask).float()

		# scene_path = os.path.join(self.img_dir.rstrip('images/'), 'scenes', self.image_paths[idx].rstrip('.png') + '.json')

		# s = open(scene_path)
		# scene = json.load(s)

		return image[:3], torch.cat([torch.clamp(mask[:1], 0, 1), image[:3]])


class CelebADataset(torch.utils.data.Dataset):
	def __init__(self, img_dir, split, transform=None):
		max_num = 50000
		self.img_dir = img_dir
		self.transform = transform
		self.image_paths = os.listdir(self.img_dir)[:max_num]

		random.shuffle(self.image_paths)
		# random.Random(4).shuffle(self.image_paths)
		num_train = int(0.7 * len(self.image_paths))

		if split == 'train':
			with open(os.path.join(self.img_dir, "shuffled_data_ids.json"), 'w') as f:
				json.dump(self.image_paths, f, indent=2)
			self.train = True
			self.image_paths = self.image_paths[:num_train]
		elif split == 'test':
			with open(os.path.join(self.img_dir, "shuffled_data_ids.json"), 'r') as f:
				self.image_paths = json.load(f)
			self.train = False
			self.image_paths = self.image_paths[num_train:]

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		good = False
		while not good:
			img_path = os.path.join(self.img_dir, self.image_paths[idx])
			try:
				image = (np.array(Image.open(img_path)) / 255)
			except:
				idx = (idx + 1) % len(self.image_paths)
				continue
			good = True

		if self.transform:
			image = self.transform(image).float()
		return image, image


class Clevr_with_attr(torch.utils.data.Dataset):
	def __init__(self, img_dir, split, attribute='color', max_attributes=5, transform=None, max_num=None,
				 perc_train=0.8):

		self.attribute = attribute
		self.max_attributes = max_attributes

		self.img_dir = os.path.join(img_dir, 'clevr-dataset-gen/output/images/')
		self.transform = transform
		self.image_paths = os.listdir(self.img_dir)
		self.image_paths = [path for path in self.image_paths if not path.endswith('mask.png')]
		if max_num is not None:
			self.image_paths = self.image_paths[:max_num]

		if attribute == 'color':
			self.attribute_list = ['gray', 'blue', 'brown', 'yellow', 'red', 'green', 'purple', 'cyan']
		elif attribute == 'shape':
			self.attribute_list = ['cube', 'sphere', 'cylinder']
		else:
			raise NotImplementedError
		# self.materials = []

		random.Random(4).shuffle(self.image_paths)
		num_train = int(perc_train * len(self.image_paths))
		if split == 'train':
			with open(os.path.join(self.img_dir, "shuffled_data_ids.json"), 'w') as f:
				json.dump(self.image_paths, f, indent=2)
			self.train = True
			self.image_paths = self.image_paths[:num_train]
		elif split == 'test':
			with open(os.path.join(self.img_dir, "shuffled_data_ids.json"), 'r') as f:
				self.image_paths = json.load(f)
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

		scene_path = os.path.join(self.img_dir.rstrip('images/'), 'scenes',
								  self.image_paths[idx].rstrip('.png') + '.json')

		s = open(scene_path)
		scene = json.load(s)

		atts = torch.zeros((self.max_attributes,), dtype=torch.int32)
		for i, object in enumerate(scene['objects']):
			atts[i] = self.attribute_list.index(object[self.attribute]) + 1
		cond_image = image[:3]
		return image[:3], atts, cond_image


# class CUB(torch.utils.data.Dataset):
# 	def __init__(self, img_dir, split, transform=None, max_num=None, perc_train=0.8):
#
# 		self.img_dir = img_dir
# 		self.transform = transform
# 		self.image_paths = os.listdir(self.img_dir)
# 		self.image_paths = [path for path in self.image_paths if not path.endswith('mask.png')]
# 		if max_num is not None:
# 			self.image_paths = self.image_paths[:max_num]
#
# 		if attribute == 'color':
# 			self.attribute_list = ['gray', 'blue', 'brown', 'yellow', 'red', 'green', 'purple', 'cyan']
# 		elif attribute == 'shape':
# 			self.attribute_list = ['cube', 'sphere', 'cylinder']
# 		else: raise NotImplementedError
# 		# self.materials = []
#
# 		random.shuffle(self.image_paths)
# 		num_train = int(perc_train * len(self.image_paths))
# 		if split == 'train':
# 			self.train = True
# 			self.image_paths = self.image_paths[:num_train]
# 		elif split == 'test':
# 			self.train = False
# 			self.image_paths = self.image_paths[num_train:]
#
# 	def __len__(self):
# 		return len(self.image_paths)
#
# 	def __getitem__(self, idx):
#
# 		# good = False
# 		# while not good:
# 		# 	img_path = os.path.join(self.img_dir, self.image_paths[idx])
# 		# 	try:
# 		# 		image = (np.array(Image.open(img_path)) / 255)
# 		# 	except:
# 		# 		idx = (idx + 1) % len(self.image_paths)
# 		# 		continue
# 		# 	good = True
#
# 		img_path = os.path.join(self.img_dir, self.image_paths[idx])
# 		image = (np.array(Image.open(img_path)) / 255)
# 		# mask_path = os.path.join(self.img_dir, self.image_paths[idx].rstrip('.png') + '_mask.png')
# 		# mask = (np.array(Image.open(mask_path)) / 255)
# 		# mask[mask!=1] = 0
# 		if self.transform:
# 			image = self.transform(image).float()
# 			# mask = self.transform(mask).float()
#
# 		scene_path = os.path.join(self.img_dir.rstrip('images/'), 'scenes', self.image_paths[idx].rstrip('.png') + '.json')
#
# 		s = open(scene_path)
# 		scene = json.load(s)
#
# 		atts = torch.zeros((self.max_attributes,), dtype=torch.int32)
# 		for i, object in enumerate(scene['objects']):
# 			atts[i] = self.attribute_list.index(object[self.attribute]) + 1
# 		cond_image = image[:3]
# 		return image[:3], atts, cond_image

import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
import pickle


class Cub(torch.utils.data.Dataset):
	base_folder = 'CUB_200_2011/images'
	url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
	filename = 'CUB_200_2011.tgz'
	tgz_md5 = '97eceeb196236b17998738112f37df78'

	def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
		self.root = os.path.expanduser(root)
		self.transform = transform
		self.loader = default_loader
		self.train = train
		self._load_metadata()
		self.img_ids = self.data.img_id.unique()

	# if download:
	# 	self._download()
	# if not self._check_integrity():
	# 	raise RuntimeError('Dataset not found or corrupted.' +
	# 					   ' You can use download=True to download it')

	def _load_metadata(self):
		images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
							 names=['img_id', 'filepath'])
		image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
										 sep=' ', names=['img_id', 'target'])
		df = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'attributes', 'image_attribute_labels.txt'), sep=' ',
						 names=['img_id', 'attribute_id', 'is_present', 'certainty_id', 'time', 'void1', 'void2'])
		image_attributes = df[df.columns[df.columns.isin(['img_id', 'attribute_id', 'is_present'])]]
		train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
									   sep=' ', names=['img_id', 'is_training_img'])

		images_class = images.merge(image_class_labels, on='img_id')
		data = images_class.merge(image_attributes, on='img_id')
		self.data = data.merge(train_test_split, on='img_id')

		if self.train:
			self.data = self.data[self.data.is_training_img == 1]
		else:
			self.data = self.data[self.data.is_training_img == 0]

	def _check_integrity(self):
		try:
			self._load_metadata()
		except Exception:
			return False

		for index, row in self.data.iterrows():
			filepath = os.path.join(self.root, self.base_folder, row.filepath)
			if not os.path.isfile(filepath):
				print(filepath)
				return False
		return True

	# <certainty_id> <certainty_name>
	# <image_id> <attribute_id> <is_present> <certainty_id> <time>
	def _download(self):
		import tarfile

		if self._check_integrity():
			print('Files already downloaded and verified')
			return

		download_url(self.url, self.root, self.filename, self.tgz_md5)

		with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
			tar.extractall(path=self.root)

	def __len__(self):
		return len(self.img_ids)

	def __getitem__(self, idx):
		sample = self.data[self.data.img_id == self.img_ids[idx]]  # self.data.iloc[idx]

		atts = torch.tensor(sample.is_present.to_list())[:, None]
		atts[atts == 0] = -1

		path = os.path.join(self.root, self.base_folder, sample.filepath.iloc[0])
		# target = sample.target.iloc[0] - 1  # Targets start at 1 by default, so shift to 0
		img = self.loader(path)

		if self.transform is not None:
			img = self.transform(img)

		return img, atts, img


class CubFiltered(torch.utils.data.Dataset):
	base_folder = 'CUB_200_2011/images'
	url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
	filename = 'CUB_200_2011.tgz'
	tgz_md5 = '97eceeb196236b17998738112f37df78'

	def __init__(self, root, split='train', transform=None, loader=default_loader, download=True):
		self.root = os.path.expanduser(root)
		self.transform = transform
		self.loader = default_loader
		self.image_dir = os.path.join(self.root, 'CUB_200_2011')
		# self.train = train
		# self._load_metadata()
		# self.img_ids = self.data.img_id.unique()

		## Image
		pkl_file_path = os.path.join(self.root, 'CUB', f'{split}class_level_all_features.pkl')
		self.data = []
		with open(pkl_file_path, "rb") as f:
			self.data.extend(pickle.load(f))
		for item in self.data:
			item['img_path'] = os.path.join(self.root, 'CUB_200_2011/images',
											'/'.join(item['img_path'].split('/')[-2:]))

	## Classes
	# self.classes = pd.read_csv(os.path.join(self.image_dir, 'classes.txt'))['idx species'].values

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		_dict = self.data[idx]

		# image
		img_path = _dict['img_path']
		_idx = img_path.split("/").index("CUB_200_2011")
		img_path = os.path.join(self.root, 'CUB_200_2011', *img_path.split("/")[_idx + 1:])
		img = Image.open(img_path).convert("RGB")
		if self.transform:
			img = self.transform(img)

		# attribute
		attr = np.array(_dict['attribute_label'])
		attr = np.float32(attr)

		# class label
		class_label = _dict["class_label"]
		return img, attr, img

# def __getitem__(self, idx):
# 	sample = self.data[self.data.img_id == self.img_ids[idx]] # self.data.iloc[idx]
#
# 	atts = torch.tensor(sample.is_present.to_list())[:, None]
# 	atts[atts == 0] = -1
#
# 	path = os.path.join(self.root, self.base_folder, sample.filepath.iloc[0])
# 	# target = sample.target.iloc[0] - 1  # Targets start at 1 by default, so shift to 0
# 	img = self.loader(path)
#
# 	if self.transform is not None:
# 		img = self.transform(img)
#
# 	return img, atts, img


# class CUB200(torch.utils.data.Dataset):
# 	"""
# 	Returns a compatible Torch Dataset object customized for the CUB dataset
# 	"""
#
# 	def __init__(
# 			self,
# 			root,
# 			image_dir='CUB_200_2011',
# 			seed=0,
# 			split='train',
# 			transform=None,
# 			raw=False
# 	):
# 		self.root = root
# 		self.image_dir = os.path.join(self.root, 'CUB', image_dir)
# 		self.transform = transform
#
# 		## Image
# 		pkl_file_path = os.path.join(self.root, 'CUB', f'{split}class_level_all_features.pkl')
# 		self.data = []
# 		with open(pkl_file_path, "rb") as f:
# 			self.data.extend(pickle.load(f))
#
# 		## Classes
# 		self.classes = pd.read_csv(os.path.join(self.image_dir, 'classes.txt'))['idx species'].values
#
# 	def __len__(self):
# 		return len(self.data)
#
# 	def __getitem__(self, idx):
# 		_dict = self.data[idx]
#
# 		# image
# 		img_path = _dict['img_path']
# 		_idx = img_path.split("/").index("CUB_200_2011")
# 		img_path = os.path.join(self.root, 'CUB/CUB_200_2011', *img_path.split("/")[_idx + 1:])
# 		img = Image.open(img_path).convert("RGB")
# 		if self.transform:
# 			img = self.transform(img)
#
# 		# attribute
# 		attr = np.array(_dict['attribute_label'])
# 		attr = np.float32(attr)
#
# 		# class label
# 		class_label = _dict["class_label"]
# 		return img, class_label, attr
#
# 	def __getitem__(self, idx):
# 		sample = self.data[self.data.img_id == self.img_ids[idx]] # self.data.iloc[idx]
#
# 		atts = torch.tensor(sample.is_present.to_list())[:, None]
# 		atts[atts == 0] = -1
#
# 		path = os.path.join(self.root, self.base_folder, sample.filepath.iloc[0])
# 		# target = sample.target.iloc[0] - 1  # Targets start at 1 by default, so shift to 0
# 		img = self.loader(path)
#
# 		if self.transform is not None:
# 			img = self.transform(img)
#
# 		return img, atts, img

import codecs
import os
import os.path
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import URLError

import numpy as np
import torch
from PIL import Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class MNIST(VisionDataset):
	"""`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``MNIST/raw/train-images-idx3-ubyte``
            and  ``MNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

	mirrors = [
		"http://yann.lecun.com/exdb/mnist/",
		"https://ossci-datasets.s3.amazonaws.com/mnist/",
	]

	resources = [
		("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
		("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
		("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
		("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
	]

	training_file = "training.pt"
	test_file = "test.pt"
	classes = [
		"0 - zero",
		"1 - one",
		"2 - two",
		"3 - three",
		"4 - four",
		"5 - five",
		"6 - six",
		"7 - seven",
		"8 - eight",
		"9 - nine",
	]

	@property
	def train_labels(self):
		warnings.warn("train_labels has been renamed targets")
		return self.targets

	@property
	def test_labels(self):
		warnings.warn("test_labels has been renamed targets")
		return self.targets

	@property
	def train_data(self):
		warnings.warn("train_data has been renamed data")
		return self.data

	@property
	def test_data(self):
		warnings.warn("test_data has been renamed data")
		return self.data

	def __init__(
			self,
			root: str,
			train: bool = True,
			transform: Optional[Callable] = None,
			target_transform: Optional[Callable] = None,
			download: bool = False,
	) -> None:
		super().__init__(root, transform=transform, target_transform=target_transform)
		self.train = train  # training set or test set

		if self._check_legacy_exist():
			self.data, self.targets = self._load_legacy_data()
			return

		if download:
			self.download()

		if not self._check_exists():
			raise RuntimeError("Dataset not found. You can use download=True to download it")

		self.data, self.targets = self._load_data()

	def _check_legacy_exist(self):
		processed_folder_exists = os.path.exists(self.processed_folder)
		if not processed_folder_exists:
			return False

		return all(
			check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
		)

	def _load_legacy_data(self):
		# This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
		# directly.
		data_file = self.training_file if self.train else self.test_file
		return torch.load(os.path.join(self.processed_folder, data_file))

	def _load_data(self):
		image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
		data = read_image_file(os.path.join(self.raw_folder, image_file))

		label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
		targets = read_label_file(os.path.join(self.raw_folder, label_file))

		return data, targets

	def __getitem__(self, index: int) -> Tuple[Any, Any]:
		"""
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
		img, target = self.data[index], int(self.targets[index])

		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img.numpy(), mode="L")

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, img

	def __len__(self) -> int:
		return len(self.data)

	@property
	def raw_folder(self) -> str:
		return os.path.join(self.root, self.__class__.__name__, "raw")

	@property
	def processed_folder(self) -> str:
		return os.path.join(self.root, self.__class__.__name__, "processed")

	@property
	def class_to_idx(self) -> Dict[str, int]:
		return {_class: i for i, _class in enumerate(self.classes)}

	def _check_exists(self) -> bool:
		return all(
			check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]))
			for url, _ in self.resources
		)

	def download(self) -> None:
		"""Download the MNIST data if it doesn't exist already."""

		if self._check_exists():
			return

		os.makedirs(self.raw_folder, exist_ok=True)

		# download files
		for filename, md5 in self.resources:
			for mirror in self.mirrors:
				url = f"{mirror}{filename}"
				try:
					print(f"Downloading {url}")
					download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)
				except URLError as error:
					print(f"Failed to download (trying next):\n{error}")
					continue
				finally:
					print()
				break
			else:
				raise RuntimeError(f"Error downloading {filename}")

	def extra_repr(self) -> str:
		split = "Train" if self.train is True else "Test"
		return f"Split: {split}"


def get_int(b: bytes) -> int:
	return int(codecs.encode(b, "hex"), 16)


SN3_PASCALVINCENT_TYPEMAP = {
	8: torch.uint8,
	9: torch.int8,
	11: torch.int16,
	12: torch.int32,
	13: torch.float32,
	14: torch.float64,
}


def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
	"""Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
	# read
	with open(path, "rb") as f:
		data = f.read()
	# parse
	magic = get_int(data[0:4])
	nd = magic % 256
	ty = magic // 256
	assert 1 <= nd <= 3
	assert 8 <= ty <= 14
	torch_type = SN3_PASCALVINCENT_TYPEMAP[ty]
	s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]

	parsed = torch.frombuffer(bytearray(data), dtype=torch_type, offset=(4 * (nd + 1)))

	# The MNIST format uses the big endian byte order, while `torch.frombuffer` uses whatever the system uses. In case
	# that is little endian and the dtype has more than one byte, we need to flip them.
	# if sys.byteorder == "little" and parsed.element_size() > 1:
	#     parsed = _flip_byte_order(parsed)

	assert parsed.shape[0] == np.prod(s) or not strict
	return parsed.view(*s)


def read_label_file(path: str) -> torch.Tensor:
	x = read_sn3_pascalvincent_tensor(path, strict=False)
	if x.dtype != torch.uint8:
		raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
	if x.ndimension() != 1:
		raise ValueError(f"x should have 1 dimension instead of {x.ndimension()}")
	return x.long()


def read_image_file(path: str) -> torch.Tensor:
	x = read_sn3_pascalvincent_tensor(path, strict=False)
	if x.dtype != torch.uint8:
		raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
	if x.ndimension() != 3:
		raise ValueError(f"x should have 3 dimension instead of {x.ndimension()}")
	return x
