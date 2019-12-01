import random
import string
import json

import torch
import torch.nn as nn
import torch.optim as optim

import models
from download_and_split_dataset import download_and_split_data
from dataloaders import MalariaData
from train import train_model
from test import test_model

IMG_SIZE = 50

TEST_SIZE = 0.2

BATCH_SIZE = 40

EPOCHS = 30

# Options: raw_cnn, resnet, alexnet, vgg, squeezenet, densenet, inception
MODEL_NAME = "raw_cnn"

NUM_CLASSES = 2

FEATURE_EXTRACT = True

USE_PRETRAINED = True

USE_GRAYSCALE = MODEL_NAME == "raw_cnn"

DOWNLOAD_DATA = False

Net, IMG_SIZE = models.initialize_model(MODEL_NAME, NUM_CLASSES, FEATURE_EXTRACT, USE_PRETRAINED)

if DOWNLOAD_DATA:
    download_and_split_data()

if torch.cuda.is_available():
    print("Running on GPU")
else:
    print("Running on CPU")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Initializing Network")
net = Net.to(device)

print("Params to learn:")
params_to_update = net.parameters()
if FEATURE_EXTRACT:
    params_to_update = []
    for name, param in net.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in net.named_parameters():
        if param.requires_grad:
            print("\t", name)

optimizer = optim.Adam(params_to_update, lr=0.001)
loss_function = nn.CrossEntropyLoss()

print("Compiling Data")
malaria = MalariaData()
dataloaders = malaria.compile_dataloaders(img_size=IMG_SIZE, batch_size=BATCH_SIZE, use_grayscale=USE_GRAYSCALE)

print("Starting Training")

trained_model, val_acc_history = train_model(net, dataloaders, loss_function, optimizer, device=device, num_epochs=EPOCHS)

loss, acc = test_model(trained_model, dataloaders["test"], optimizer, loss_function, device)

torch.save(trained_model.state_dict(), "trained_networks/{}.pt".format(''.join(random.choices(string.ascii_uppercase + string.digits, k=10))))

with open("trained_networks/best.json", "w+", encoding='utf-8', errors='ignore') as f:
    try:
        f_json = json.load(f, strict=False)
    except json.JSONDecodeError:
        f_json = {"acc": 0}
    if acc > f_json["acc"]:
        json.dump({"acc": acc.item(), "model_name": MODEL_NAME}, f)
        torch.save(trained_model.state_dict(), "trained_networks/best.pt")
