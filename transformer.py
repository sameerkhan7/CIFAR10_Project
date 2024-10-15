import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np

print("Downloading Dataset...")
#training set
data_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
#test set
data_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
print("Dataset Downloaded!")

