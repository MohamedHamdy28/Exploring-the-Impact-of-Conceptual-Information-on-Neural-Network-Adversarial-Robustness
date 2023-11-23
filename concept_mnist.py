from mimetypes import init
from pyexpat import model
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Lambda
from torchvision import transforms
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

import matplotlib.pyplot as plt
import random
import pdb
import argparse


class MNISTDatasetWithConcepts(Dataset):
	def __init__(self,split,num_classes,transform):
		isTrain = False
		if split == "train":
			isTrain=True
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.data = MNIST(root = "./synthetic_datasets",train=isTrain, download=True)
		self.num_classes = num_classes
		self.transform = transform
		# print(len(set([self.data[i][1] for i in range(len(self.data))])))

	def __len__(self):
		return len(self.data)

	def __getitem__(self,idx):
		img,label = self.data[idx][0],self.data[idx][1]
		onehot = torch.zeros((self.num_classes,))
		onehot[label] =1
		concept = self.make_concepts_mnist(label)
		label = onehot.to(self.device)
		return [self.transform(img).to(self.device),label,concept]

	def make_concepts_mnist(self,label):
		if label == 0:
			hard_label = torch.tensor([1,0,0,0,0,0,0,0,0,0]+[1,0])
		elif label == 1:
			hard_label = torch.tensor([0,1,0,0,0,0,0,0,0,0]+[0,1])
		elif label == 2:
			hard_label = torch.tensor([0,0,1,0,0,0,0,0,0,0]+[1,1])
		elif label == 3:
			hard_label = torch.tensor([0,0,0,1,0,0,0,0,0,0]+[1,0])
		elif label == 4:
			hard_label = torch.tensor([0,0,0,0,1,0,0,0,0,0]+[0,1])
		elif label == 5:
			hard_label = torch.tensor([0,0,0,0,0,1,0,0,0,0]+[1,1])
		elif label == 6:
			hard_label = torch.tensor([0,0,0,0,0,0,1,0,0,0]+[1,0])
		elif label == 7:
			hard_label = torch.tensor([0,0,0,0,0,0,0,1,0,0]+[0,1])
		elif label == 8:
			hard_label = torch.tensor([0,0,0,0,0,0,0,0,1,0]+[1,0])
		elif label == 9:
			hard_label = torch.tensor([0,0,0,0,0,0,0,0,0,1]+[1,0])
		
		# pdb.set_trace()

		# hard_label = torch.zeros((self.num_classes,))
		# hard_label[label] = 1
		hard_label = hard_label.float().to(self.device)
		return hard_label
	

def load_mnist_dataloader(split,bsz):
	dataset = MNISTDatasetWithConcepts(split = split, num_classes = 10, transform=ToTensor())
	dataloader = DataLoader(dataset, batch_size=bsz, shuffle=True)
	return dataloader


def main():
	train_loader = load_mnist_dataloader(split = "train",bsz=64)
	test_loader = load_mnist_dataloader(split = "test",bsz=64)

main()
