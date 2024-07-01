import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from utils import *
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

class MoleculeDataset(Dataset):
    def __init__(self, filepath, transform=None):
        self.data = pd.read_csv(filepath)
        self.transform = transform

        self.smiles = self.data[SMILES_COL_NAME]
        self.labels = np.zeros((len(self.smiles),1))

        # Building the vocabulary and converting to one-hot vectors
        self.vocab, self.inv_dict = build_vocab(self.data)
        self.vocab_size = len(self.vocab)
        self.data = make_one_hot(self.data[SMILES_COL_NAME], self.vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'smiles': torch.FloatTensor(self.data[idx])}
        if self.transform:
            sample = self.transform(sample)
        return sample




class MoleculeDataModule(LightningDataModule):
    def __init__(self, train_file, val_file, batch_size=32):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = MoleculeDataset(self.train_file)
        self.val_dataset = MoleculeDataset(self.val_file)
        self.vocab_size = self.train_dataset.vocab_size
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=100000)


def main():
    # Define file paths
    train_file = '../data/chembl_50_small.csv'  # replace with your train file path
    val_file = '../data/chembl_50_small.csv'  # replace with your validation file path

    # Create a MoleculeDataModule
    data_module = MoleculeDataModule(train_file, val_file)

    # Setup the data module
    data_module.setup()
    print(data_module.vocab_size)
    # Get the train and validation data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Print the first batch of the train and validation data
    for i, batch in enumerate(train_loader):
        print(batch['smiles'].shape)
        if i == 0:
            break

    # for i, batch in enumerate(val_loader):
    #     print(f"Validation Batch {i+1}: {batch}")
    #     if i == 0:
    #         break

if __name__ == "__main__":
    main()