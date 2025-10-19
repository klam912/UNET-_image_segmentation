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
    """
    def __init__(self, in_channels=3, out_channels=3, filters: Optional[List[int]] = None, deep_supervision: bool = False):
        super().__init__()
        if filters is None:
            filters = [64, 128, 256, 512]

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

        # Decoder nested blocks
        self.up1_0 = UpSample(filters[1], filters[0])
        self.conv0_1 = ConvBlock(filters[0] + filters[0], filters[0])

        self.up2_1 = UpSample(filters[2], filters[1])
        self.conv1_1 = ConvBlock(filters[1] + filters[1], filters[1])

        self.up3_2 = UpSample(filters[3], filters[2])
        self.conv2_1 = ConvBlock(filters[2] + filters[2], filters[2])

        # Second nested level
        self.up1_1 = UpSample(filters[1], filters[0])
        self.conv0_2 = ConvBlock(filters[0] * 2 + filters[0], filters[0])

        self.up2_2 = UpSample(filters[2], filters[1])
        self.conv1_2 = ConvBlock(filters[1] * 2 + filters[1], filters[1])

        # Third nested level
        self.up1_2 = UpSample(filters[1], filters[0])
        self.conv0_3 = ConvBlock(filters[0] * 3 + filters[0], filters[0])

        # Final outputs
        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool0(x0_0))
        x2_0 = self.conv2_0(self.pool1(x1_0))
        x3_0 = self.conv3_0(self.pool2(x2_0))

        # first nested layer
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_1(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_2(x3_0)], dim=1))

        # second nested layer
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_2(x2_1)], dim=1))

        # third nested layer
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], dim=1))

        # outputs
        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            return (out1, out2, out3)
        else:
            return self.final(x0_3)


# --------------------
# Loss and metrics
# --------------------
def dice_coefficient_multiclass(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3, eps: float = 1e-6):
    """
    Multi-class Dice coefficient.
    pred: logits (BxCxHxW) or probabilities after softmax
    target: class indices (BxHxW) with values in [0, num_classes-1]
    """
    # Convert logits to probabilities if needed
    if pred.shape[1] == num_classes:
        pred = F.softmax(pred, dim=1)
    
    # Convert target to one-hot encoding
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes)  # BxHxWxC
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # BxCxHxW
    
    # Compute Dice per class
    dice_scores = []
    for c in range(num_classes):
        pred_c = pred[:, c]
        target_c = target_one_hot[:, c]
        
        pred_c = pred_c.flatten()
        target_c = target_c.flatten()
        
        intersection = (pred_c * target_c).sum()
        dice = (2.0 * intersection + eps) / (pred_c.sum() + target_c.sum() + eps)
        dice_scores.append(dice)
    
    return torch.stack(dice_scores).mean()


def iou_coefficient_multiclass(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3, eps: float = 1e-6):
    """
    Multi-class IoU coefficient.
    pred: logits (BxCxHxW) or probabilities after softmax
    target: class indices (BxHxW) with values in [0, num_classes-1]
    """
    # Convert logits to class predictions
    pred_classes = torch.argmax(pred, dim=1)  # BxHxW
    
    iou_scores = []
    for c in range(num_classes):
        pred_c = (pred_classes == c)
        target_c = (target == c)
        
        intersection = (pred_c & target_c).float().sum()
        union = (pred_c | target_c).float().sum()
        
        iou = (intersection + eps) / (union + eps)
        iou_scores.append(iou)
    
    return torch.stack(iou_scores).mean()


# --------------------
# Lightning module
# --------------------
class UNetPPLightning(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        filters: Optional[List[int]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['filters'])
        self.model = UNetPP(in_channels, out_channels, filters=filters, deep_supervision=deep_supervision)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.deep_supervision = deep_supervision
        self.num_classes = out_channels

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch  # mask: (B, H, W) with values [0, 1, 2]
        logits = self(img)

        if self.deep_supervision:
            losses = []
            for out in logits:
                if out.shape[2:] != mask.shape[1:]:
                    out = F.interpolate(out, size=mask.shape[1:], mode='bilinear', align_corners=True)
                losses.append(self.loss_fn(out, mask))
            loss = torch.stack(losses).mean()
            pred_logits = logits[-1]
        else:
            if logits.shape[2:] != mask.shape[1:]:
                logits = F.interpolate(logits, size=mask.shape[1:], mode='bilinear', align_corners=True)
            loss = self.loss_fn(logits, mask)
            pred_logits = logits

        dice = dice_coefficient_multiclass(pred_logits, mask, num_classes=self.num_classes)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/dice", dice, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch  # mask: (B, H, W)
        logits = self(img)
        
        if self.deep_supervision:
            out = logits[-1]
        else:
            out = logits

        if out.shape[2:] != mask.shape[1:]:
            out = F.interpolate(out, size=mask.shape[1:], mode='bilinear', align_corners=True)

        loss = self.loss_fn(out, mask)
        dice = dice_coefficient_multiclass(out, mask, num_classes=self.num_classes)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/dice", dice, on_epoch=True, prog_bar=True)

        return {"val_loss": loss, "val_dice": dice}

    def test_step(self, batch, batch_idx):
        img, mask = batch  # mask: (B, H, W)
        logits = self(img)
        
        if self.deep_supervision:
            out = logits[-1]
        else:
            out = logits
            
        if out.shape[2:] != mask.shape[1:]:
            out = F.interpolate(out, size=mask.shape[1:], mode='bilinear', align_corners=True)

        loss = self.loss_fn(out, mask)
        
        # Compute metrics
        dice_score = dice_coefficient_multiclass(out, mask, num_classes=self.num_classes)
        iou_score = iou_coefficient_multiclass(out, mask, num_classes=self.num_classes)

        self.log("test/loss", loss, on_epoch=True, prog_bar=True)
        self.log("test/dice", dice_score, on_epoch=True, prog_bar=True)
        self.log("test/iou", iou_score, on_epoch=True, prog_bar=True)

        return {"test_loss": loss, "test_dice": dice_score, "test_iou": iou_score}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3, factor=0.5),
            'monitor': 'val/dice',
            'interval': 'epoch',
            'frequency': 1,
        }
        return {"optimizer": opt, "lr_scheduler": scheduler}