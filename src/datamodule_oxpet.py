import torch
from torch import nn
import os
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

# For reproducibility
pl.seed_everything(42)

# Create a LightningDataModule for OxPet
class OxPetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 64, target_type: str = "segmentation"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_type = target_type

        # Transform the image by resizing and converting to tensors 
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.target_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.PILToTensor(), # for segmentation mask
        ])

    # This is called on one process only.
    # Use it for downloading the dataset.
    def prepare_data(self):
        datasets.OxfordIIITPet(self.data_dir, download=True)

    # This is called on every process (GPU).
    # Use it to split data, create datasets, etc.
    def setup(self, stage: str):
        root = os.path.join(self.data_dir, "oxford-iiit-pet")

        # Read the official splits
        trainval_file = os.path.join(root, "annotations", "trainval.txt")
        test_file = os.path.join(root, "annotations", "test.txt")

        with open(trainval_file, "r") as file:
            trainval_names = [line.strip().split(" ")[0] for line in file.readlines()]
        
        with open(test_file, "r") as file:
            test_names = [line.strip().split(" ")[0] for line in file.readlines()]


        ds_trainval = datasets.OxfordIIITPet(
            self.data_dir,
            target_types=self.target_type,
            transform=self.transform,
            target_transform=self.target_transform,
            split="trainval"
        )

        ds_test = datasets.OxfordIIITPet(
            self.data_dir,
            target_types=self.target_type,
            transform=self.transform,
            target_transform=self.target_transform,
            split="test"
        )

        # Create indices for trainval images from the full dataset
        trainval_name_to_idx = {
            os.path.splitext(os.path.basename(path))[0]: i
            for i, path in enumerate(ds_trainval._images)
        }
        trainval_indices = [trainval_name_to_idx[n] for n in trainval_names if n in trainval_name_to_idx]
        trainval_dataset = Subset(ds_trainval, trainval_indices)

        # Create indices for test images from the full dataset
        test_name_to_idx = {
            os.path.splitext(os.path.basename(path))[0]: i
            for i, path in enumerate(ds_test._images)
        }
        test_indices = [test_name_to_idx[n] for n in test_names if n in test_name_to_idx]
        test_dataset = Subset(ds_test, test_indices)

        # Stage setup
        if stage == "fit" or stage is None:
            # 80/20 split for train and val datasets
            n_train = int(0.8 * len(trainval_dataset))
            n_val = len(trainval_dataset) - n_train
            self.oxpet_train, self.oxpet_val = random_split(
                trainval_dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42)
            )

        if stage == "test" or stage is None:
            # No splits for test
            self.oxpet_test = test_dataset

    # Return the training dataloader
    def train_dataloader(self):
        return DataLoader(self.oxpet_train, batch_size=self.batch_size, shuffle=True,)

    # Return the validation dataloader
    def val_dataloader(self):
        return DataLoader(self.oxpet_val, batch_size=self.batch_size)

    # Return the test dataloader
    def test_dataloader(self):
        return DataLoader(self.oxpet_test, batch_size=self.batch_size)
    
if __name__ == "__main__":
    # For debugging
    dm = OxPetDataModule(target_type="segmentation")
    dm.prepare_data()
    dm.setup("fit")
    imgs, masks = next(iter(dm.train_dataloader()))
    print("Train batch:", imgs.shape, masks.shape)

    imgs, masks = next(iter(dm.val_dataloader()))
    print("Val batch:", imgs.shape, masks.shape)

    dm.setup("test")
    imgs, masks = next(iter(dm.test_dataloader()))
    print("Test batch:", imgs.shape, masks.shape)