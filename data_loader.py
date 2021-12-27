

import torch
import torchvision
import torchvision.transforms.functional as F
import folders


class DataLoader(object):
	"""
	Dataset class for IQA databases
	"""

	def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True):

		self.batch_size = batch_size
		self.istrain = istrain

		if (dataset == 'live') | (dataset == 'csiq') | (dataset == 'tid2013') | (dataset == 'clive')| (dataset == 'kadid10k'):
			if istrain:
				transforms = torchvision.transforms.Compose([
					torchvision.transforms.RandomHorizontalFlip(),
					torchvision.transforms.RandomVerticalFlip(),
					torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
													 std=(0.229, 0.224, 0.225))
				])
			else:
				transforms = torchvision.transforms.Compose([
					torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
													 std=(0.229, 0.224, 0.225))
				])
		elif dataset == 'koniq':
			if istrain:
				transforms = torchvision.transforms.Compose([
					torchvision.transforms.RandomHorizontalFlip(),
					torchvision.transforms.RandomVerticalFlip(),
					torchvision.transforms.Resize((512, 384)),
					torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
													 std=(0.229, 0.224, 0.225))])
			else:
				transforms = torchvision.transforms.Compose([
					torchvision.transforms.Resize((512, 384)),
					torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
													 std=(0.229, 0.224, 0.225))])
		elif dataset == 'fblive':
			if istrain:
				transforms = torchvision.transforms.Compose([
					torchvision.transforms.Resize((512, 512)),
					torchvision.transforms.RandomHorizontalFlip(),
					torchvision.transforms.RandomVerticalFlip(),
					torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
													 std=(0.229, 0.224, 0.225))])
			else:
				transforms = torchvision.transforms.Compose([
					torchvision.transforms.Resize((512, 512)),
					torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
													 std=(0.229, 0.224, 0.225))])

		if dataset == 'live':
			self.data = folders.LIVEFolder(
				root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		elif dataset == 'clive':
			self.data = folders.LIVEChallengeFolder(
				root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		elif dataset == 'csiq':
			self.data = folders.CSIQFolder(
				root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		elif dataset == 'koniq':
			self.data = folders.Koniq_10kFolder(
				root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		elif dataset == 'fblive':
			self.data = folders.FBLIVEFolder(
				root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		elif dataset == 'tid2013':
			self.data = folders.TID2013Folder(
				root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		elif dataset == 'kadid10k':
			self.data = folders.Kadid10k(
				root=path, index=img_indx, transform=transforms, patch_num=patch_num)			

	def get_data(self):
		if self.istrain:
			dataloader = torch.utils.data.DataLoader(
				self.data, batch_size=self.batch_size, shuffle=True)
		else:
			dataloader = torch.utils.data.DataLoader(
				self.data, batch_size=1, shuffle=False)
		return dataloader