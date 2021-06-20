import os
import pdb

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from dataloader_module import DataLoaderModule
from unet3d_model import loss, unet3d

import argparse
from multiprocessing import cpu_count


class BasicUnet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = unet3d.UnetModel(1, 1)
        self.loss_function = loss.DiceLoss()

    def forward(self, x):

        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):

        x, y = batch[0], batch[1]
        prediction = self.forward(x)
        loss_calculated = self.loss_function.forward(y, prediction)
        self.log(
            "train_loss",
            loss_calculated,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss_calculated

    def validation_step(self, batch, batch_idx):

        x, y = batch[0], batch[1]
        prediction = self.forward(x)
        loss_calculated = self.loss_function.forward(y, prediction)
        self.log(
            "val_loss",
            loss_calculated,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return prediction

    def test_step(self, batch, batch_idx):

        x, y = batch[0], batch[1]
        prediction = self.forward(x)
        loss_calculated = self.loss_function.forward(y, prediction)
        self.log(
            "test_loss",
            loss_calculated,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return prediction

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path", default=None, help="path to the extracted folder of the NFBS dataset"
    )
    ap.add_argument(
        "--max_epochs", default=50, help="Number of epochs to run your trainig for"
    )
    
    ap.add_argument(
        "--gpus",
        default=None,
        help="Number of gpus on run on training. Assumption all gpus on the same machine. Also we assume atleast one gpu is present",
    )

    arguments = ap.parse_args()

    if arguments.path is None:
        data_module = DataLoaderModule(
            "/extern/home/harshagarwal/brain_imaging/NFBS_Dataset"
        )
    else:
        data_module = DataLoaderModule(arguments.path)

    logger = TensorBoardLogger(save_dir="./logs/", version=1, name="lightning_logs")
    basicModel = BasicUnet()

    trainer = pl.Trainer(
        gpus=arguments.gpus,
        accelerator="ddp",
        check_val_every_n_epoch=5,
        flush_logs_every_n_steps=100,
        logger=logger,
        max_epochs=arguments.max_epochs,
        num_processes=cpu_count(),
    )
    trainer.fit(basicModel, data_module)
