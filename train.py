import os
import pdb

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from dataloader_module import DataLoaderModule
from unet3d_model import loss, unet3d


class BasicUnet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = unet3d.UnetModel(1, 1)

    def forward(self, x):

        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):

        x, y = batch[0], batch[1]
        prediction = self.forward(x)
        loss_calculated = loss.DiceLoss(y, prediction)
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
        loss_calculated = loss.DiceLoss(y, prediction)
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
        loss_calculated = loss.DiceLoss(y, prediction)
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


data_module = DataLoaderModule("/extern/home/harshagarwal/brain_imaging/NFBS_Dataset")
logger = TensorBoardLogger(save_dir="./logs/", version=1, name="lightning_logs")
basicModel = BasicUnet()

trainer = pl.Trainer(
    gpus=[0],
    accelerator="ddp",
    check_val_every_n_epoch=2,
    flush_logs_every_n_steps=100,
    logger=logger,
    max_epochs=2,
    num_processes=4,
)
trainer.fit(basicModel, data_module)
