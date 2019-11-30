import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import models
from load_data import MalariaData
from train import train_model

IMG_SIZE = 50

TEST_SIZE = 0.2

BATCH_SIZE = 65

EPOCHS = 20

MODEL_NAME = "raw_cnn"

NUM_CLASSES = 2

FEATURE_EXTRACT = True

USE_PRETRAINED = True

Net, IMG_SIZE = models.initialize_model(MODEL_NAME, NUM_CLASSES, FEATURE_EXTRACT, USE_PRETRAINED)

# Device config

if torch.cuda.is_available():
    print("Running on GPU")
else:
    print("Running on CPU")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Initializing Network")
net = Net.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

print("Compiling Data")
malaria = MalariaData()
dataloaders = malaria.compile_dataloaders(img_size=IMG_SIZE, batch_size=BATCH_SIZE)

print("Starting Training")

train_model(net, dataloaders, loss_function, optimizer, device=device, num_epochs=EPOCHS)
