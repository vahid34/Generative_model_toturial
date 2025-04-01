import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



tensor_transform = transforms.ToTensor()
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=tensor_transform)
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)