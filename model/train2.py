import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from dataset import MoleculeDataModule
from model import MolecularVAE
from pytorch_lightning.callbacks import ProgressBar


def main():
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="checkpoints_{epoch:02d}-{val_loss:.2f}",
        monitor='val_loss',
        save_top_k=3,
        mode='min'
    )
    # bar = CustomProgressBar()

    early_stopping = pl.callbacks.early_stopping.EarlyStopping("val_loss", patience=15)
    # Initialize data module and model
    train_file = '../data/chembl_50_small.csv'  # replace with your train file path
    val_file = '../data/chembl_50_small.csv'  # replace with your validation file path

    # Create a MoleculeDataModule
    data_module = MoleculeDataModule(train_file, val_file)

    # Setup the data module
    data_module.setup()

    model = MolecularVAE(vocab_size=data_module.vocab_size)

    # Initialize a trainer
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping],
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        accumulate_grad_batches=1,
        log_every_n_steps=10,
        min_epochs=5,
        # gpus=True,
        max_epochs=30,
    )

    # Train the model
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()