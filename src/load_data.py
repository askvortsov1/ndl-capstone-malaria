import os
from torchvision import datasets, transforms
import torch


class MalariaData:
    def compile_dataloaders(self, data_dir="data/malaria", img_size=50, batch_size=30):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]),
            'val': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]),
        }

        malaria_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val"]}

        malaria_dataloaders = {x: torch.utils.data.DataLoader(malaria_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

        return malaria_dataloaders
