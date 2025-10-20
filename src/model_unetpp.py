import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss


class UNetPPLightning(pl.LightningModule):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        classes=3,
        lr=1e-3,
        weight_decay=1e-4,
        encoder_name="resnet34",
        use_dice_loss=False,
        max_epochs=50,
    ):
        super().__init__()
        self.save_hyperparameters()

        # UNet++ model
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            out_channels=out_channels,
            classes=classes,
            activation=None,
        )

        # Loss functions
        self.use_dice_loss = use_dice_loss
        self.ce_loss = nn.CrossEntropyLoss()
        if use_dice_loss:
            self.dice_loss = DiceLoss(mode="multiclass")

        # Metrics
        self.train_iou = torchmetrics.JaccardIndex(
            num_classes=classes, task="multiclass"
        )
        self.val_iou = torchmetrics.JaccardIndex(
            num_classes=classes, task="multiclass"
        )
        self.val_dice = torchmetrics.F1Score(
            num_classes=classes, task="multiclass", average="macro"
        )

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        imgs, masks = batch
        logits = self(imgs)

        # Compute loss
        if self.use_dice_loss:
            if self.training:
                # DiceLoss handles one-hot encoding internally for multiclass mode
                loss = 0.5 * self.dice_loss(logits, masks) + 0.5 * self.ce_loss(
                    logits, masks
                )
            else:
                # During validation/test, only use CE loss
                loss = self.ce_loss(logits, masks)
        else:
            loss = self.ce_loss(logits, masks)

        # Predictions
        preds = torch.argmax(logits, dim=1)
        return loss, preds, masks

    def training_step(self, batch, batch_idx):
        loss, preds, masks = self.step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Update metrics safely
        self.train_iou.update(preds, masks)
        self.log(
            "train_mIoU",
            self.train_iou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, masks = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)

        self.val_iou.update(preds, masks)
        self.val_dice.update(preds, masks)

        self.log(
            "val_mIoU", self.val_iou, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val_Dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        loss, preds, masks = self.step(batch)
        self.log("test_loss", loss)
        self.log("test_mIoU", self.val_iou(preds, masks))
        self.log("test_Dice", self.val_dice(preds, masks))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}