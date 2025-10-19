from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from datamodule_oxpet import *
from model_unetpp import *

# print(torch.cuda.is_available())      # Should print True
# print(torch.cuda.get_device_name(0))  # Should print "NVIDIA GeForce RTX 5070 Ti"

# Instantiate data module and split into train and val
dm = OxPetDataModule(batch_size=8)
dm.prepare_data()
dm.setup("fit")


# Create a model checkpoint for callback
checkpoint_callback = ModelCheckpoint(
    monitor="val/dice",
    mode="max",
    save_top_k=1,
    filename="best-unetpp-{epoch:02d}-{val_dice:.4f}",
    verbose=True,
)

# Instantiate model
model = UNetPPLightning()

# --- Instantiate the Trainer ---
# The Trainer automates the training, validation, and testing loops.
# You can specify the number of epochs, accelerator (cpu, gpu, tpu), devices, etc.
trainer = pl.Trainer(
    max_epochs=5,
    accelerator="gpu",
    devices=1,
    precision=16,
    callbacks=[checkpoint_callback],
)

# --- Start Training ---
# The .fit() method takes the model and the datamodule.
# It will automatically call the setup, dataloaders, and training/validation steps.
trainer.fit(model, dm)