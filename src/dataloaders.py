import os
from torchvision import datasets, transforms
import torch


class MalariaData:
    def compile_dataloaders(self, data_dir="data/malaria", img_size=50, batch_size=30, use_grayscale=True):
        normalize = [0.5] if use_grayscale else [0.5, 0.5, 0.5]
        data_transforms = {
            'train': [
                transforms.RandomRotation(20),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomGrayscale(int(use_grayscale)),
                transforms.ToTensor(),
                transforms.Normalize(normalize, normalize)
            ],
            'val': [
                transforms.Resize((img_size, img_size)),
                transforms.RandomGrayscale(int(use_grayscale)),
                transforms.ToTensor(),
                transforms.Normalize(normalize, normalize)
            ],
            'test': [
                transforms.Resize((img_size, img_size)),
                transforms.RandomGrayscale(int(use_grayscale)),
                transforms.ToTensor(),
                transforms.Normalize(normalize, normalize)
            ]}
        if use_grayscale:
            for transform in data_transforms:
                data_transforms[transform].insert(0, transforms.Grayscale())
        data_transforms = {k: transforms.Compose(v) for k, v in data_transforms.items()}

        malaria_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val", "test"]}

        malaria_dataloaders = {x: torch.utils.data.DataLoader(
            malaria_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', "test"]}

        return malaria_dataloaders
