from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# --------------------
# Basic building blocks
# --------------------
class ConvBlock(nn.Module):
    """Two conv layers with BatchNorm and ReLU (same padding)."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpSample(nn.Module):
    """Upsample by factor 2 using bilinear interpolation followed by a conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


# --------------------
# UNet++ architecture
# --------------------
class UNetPP(nn.Module):
    """
    UNet++ (nested U-Net) implementation.
    Args:
        in_channels: input channels (e.g., 3)
        out_channels: number of output channels (e.g., number of classes or 1 for binary)
        filters: list of feature counts per encoder stage, e.g. [64, 128, 256, 512]
        deep_supervision: if True, provide output from intermediate decoder stages (as in the paper)
    """
    def __init__(self, in_channels=3, out_channels=1, filters: Optional[List[int]] = None, deep_supervision: bool = False):
        super().__init__()
        if filters is None:
            filters = [64, 128, 256, 512]  # default configuration (4 levels)

        self.deep_supervision = deep_supervision
        n = len(filters)

        # Encoder blocks and pooling
        self.conv0_0 = ConvBlock(in_channels, filters[0])
        self.pool0 = nn.MaxPool2d(2)

        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.pool1 = nn.MaxPool2d(2)

        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.pool2 = nn.MaxPool2d(2)

        self.conv3_0 = ConvBlock(filters[2], filters[3])
        # if you want deeper, add more blocks similarly

        # Decoder nested blocks (x_i_j where i = depth, j = level of nested aggregation)
        # Following names: conv{depth}_{stage}
        # conv0_1 depends on conv0_0 and upsample(conv1_0)
        self.up1_0 = UpSample(filters[1], filters[0])
        self.conv0_1 = ConvBlock(filters[0] + filters[0], filters[0])

        self.up2_1 = UpSample(filters[2], filters[1])
        self.conv1_1 = ConvBlock(filters[1] + filters[1], filters[1])

        self.up3_2 = UpSample(filters[3], filters[2])
        self.conv2_1 = ConvBlock(filters[2] + filters[2], filters[2])

        # second nested level
        # conv0_2 uses conv0_0, conv0_1, upsample(conv1_1)
        self.up1_1 = UpSample(filters[1], filters[0])
        self.conv0_2 = ConvBlock(filters[0] * 2 + filters[0], filters[0])

        self.up2_2 = UpSample(filters[2], filters[1])
        self.conv1_2 = ConvBlock(filters[1] * 2 + filters[1], filters[1])

        # third nested level (if depth allows)
        self.up1_2 = UpSample(filters[1], filters[0])
        self.conv0_3 = ConvBlock(filters[0] * 3 + filters[0], filters[0])

        # Final 1x1 convs for outputs (deep supervision)
        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # encoder
        x0_0 = self.conv0_0(x)                 # level 0, stage 0
        x1_0 = self.conv1_0(self.pool0(x0_0))  # level 1, stage 0
        x2_0 = self.conv2_0(self.pool1(x1_0))  # level 2, stage 0
        x3_0 = self.conv3_0(self.pool2(x2_0))  # level 3, stage 0

        # first nested layer
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_1(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_2(x3_0)], dim=1))

        # second nested layer
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_2(x2_1)], dim=1))

        # third nested layer (top-most aggregation)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], dim=1))

        # outputs
        if self.deep_supervision:
            # Return tuple of outputs from different depths (paper uses these for deep supervision)
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            # Upsample to input resolution if necessary (should already match)
            return (out1, out2, out3)
        else:
            return self.final(x0_3)


# --------------------
# Loss and metrics
# --------------------
def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """
    pred: probabilities or logits after sigmoid (shape Bx1xHxW or BxCxHxW)
    target: same shape, binary {0,1}
    """
    if pred.ndim > 2:
        pred = pred.flatten(2)
        target = target.flatten(2)
    intersection = (pred * target).sum(-1)
    sums = pred.sum(-1) + target.sum(-1)
    dice = 2.0 * intersection / (sums + eps)
    return dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, weight_bce=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce

    def forward(self, logits, target):
        # logits: raw model output (no sigmoid)
        bce_loss = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dice = dice_coefficient(probs, target)
        dice_loss = 1.0 - dice
        return self.weight_bce * bce_loss + (1.0 - self.weight_bce) * dice_loss


# --------------------
# Lightning module
# --------------------
class UNetPPLightning(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        filters: Optional[List[int]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['filters'])
        self.model = UNetPP(in_channels, out_channels, filters=filters, deep_supervision=deep_supervision)
        self.loss_fn = DiceBCELoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.deep_supervision = deep_supervision

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch  # mask should be float tensor with values 0/1 (Bx1xHxW)
        logits = self(img)

        if self.deep_supervision:
            # logits is a tuple of outputs; compute loss on last (or average)
            # We'll average loss across deep supervision outputs.
            losses = []
            for out in logits:
                # resize out to mask size if necessary
                if out.shape != mask.shape:
                    out = F.interpolate(out, size=mask.shape[2:], mode='bilinear', align_corners=True)
                losses.append(self.loss_fn(out, mask))
            loss = torch.stack(losses).mean()
            # Use last output for metrics
            pred = torch.sigmoid(logits[-1])
        else:
            if logits.shape != mask.shape:
                logits = F.interpolate(logits, size=mask.shape[2:], mode='bilinear', align_corners=True)
            loss = self.loss_fn(logits, mask)
            pred = torch.sigmoid(logits)

        dice = dice_coefficient(pred, mask)
        # log
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/dice", dice, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        logits = self(img)
        if self.deep_supervision:
            out = logits[-1]
        else:
            out = logits

        if out.shape != mask.shape:
            out = F.interpolate(out, size=mask.shape[2:], mode='bilinear', align_corners=True)

        loss = self.loss_fn(out, mask)
        prob = torch.sigmoid(out)
        dice = dice_coefficient(prob, mask)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/dice", dice, on_epoch=True, prog_bar=True)

        return {"val_loss": loss, "val_dice": dice}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # simple ReduceLROnPlateau as example
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3, factor=0.5),
            'monitor': 'val/dice',
            'interval': 'epoch',
            'frequency': 1,
        }
        return {"optimizer": opt, "lr_scheduler": scheduler}
