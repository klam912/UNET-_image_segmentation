import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss as SMPDiceLoss
import wandb


# UNET++ is consisted of a ConvBlock, UpSampling, DownSampling 
class ConvBlock(nn.Module):
    """Double convolution block: Conv2d -> BatchNorm -> ReLU"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # First block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Second block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x 
    
class UpSampling(nn.Module):
    """Upsample with a convolution"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Scale factor by 2 (64 -> 128)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)

        return x

class DensityBlock(nn.Module):
    """Nested dense skip connection block in UNET++"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    """
        UNET++ Architecture:
        - Use dense skip connections (allow layers to skip to other layers in the same encoder/decoder level)

        Architecture breakdown:
        - Encoder: 4 levels with max pooling (downsampling to reduce # of channels)
        - Decoder: 4 levels with upsampling and dense connections (increase # of channels to revert back to the image with dense connections create a more robust feature map)
    """

    def __init__(self, in_channels: int = 3, classes: int = 1, filters: tuple[int, ...] = [64, 128, 256, 512, 1024]):
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.filters = filters

        # Encoder + downsampling
        ## Level 0: 64 filters
        self.encoder0 = ConvBlock(in_channels, filters[0])
        self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2)

        ## Level 1: 128 filters
        self.encoder1 = ConvBlock(filters[0], filters[1])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        ## Level 2: 256 filters
        self.encoder2 = ConvBlock(filters[1], filters[2])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        ## Level 3: 512 filters
        self.encoder3 = ConvBlock(filters[2], filters[3])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        ## Level 4: 1024 filters
        self.encoder4 = ConvBlock(filters[3], filters[4])

        # Decoder with dense connections + upsampling
        # Level 3 (1024 -> 512)
        self.up3_0 = UpSampling(filters[4], filters[3])
        self.dense3_0 = DensityBlock(filters[3] + filters[3], filters[3]) # concat with encoder 3

        # Level 2 (512 -> 256)
        self.up2_0 = UpSampling(filters[3], filters[2])
        self.dense2_0 = DensityBlock(filters[2] + filters[2], filters[2]) # concat with encoder 2

        self.up2_1 = UpSampling(filters[3], filters[2])
        self.dense2_1 = DensityBlock(filters[2] + filters[2] + filters[2], filters[2])

        # Level 1 (256 -> 128)
        self.up1_0 = UpSampling(filters[2], filters[1])
        self.dense1_0 = DensityBlock(filters[1] + filters[1], filters[1]) # concat with encoder 1

        self.up1_1 = UpSampling(filters[2], filters[1])
        self.dense1_1 = DensityBlock(filters[1] + filters[1] + filters[1], filters[1])

        self.up1_2 = UpSampling(filters[2], filters[1])
        self.dense1_2 = DensityBlock(filters[1] + filters[1] + filters[1] + filters[1], filters[1])

        # Level 0 (128 -> 64)
        self.up0_0 = UpSampling(filters[1], filters[0])
        self.dense0_0 = DensityBlock(filters[0] + filters[0], filters[0]) # concat with encoder 0

        self.up0_1 = UpSampling(filters[1], filters[0])
        self.dense0_1 = DensityBlock(filters[0] + filters[0] + filters[0], filters[0])

        self.up0_2 = UpSampling(filters[1], filters[0])
        self.dense0_2 = DensityBlock(filters[0] + filters[0] + filters[0] + filters[0], filters[0])

        self.up0_3 = UpSampling(filters[1], filters[0])
        self.dense0_3 = DensityBlock(filters[0] + filters[0] + filters[0] + filters[0] + filters[0], filters[0])

        # Final output layer
        self.final = nn.Conv2d(filters[0], classes, kernel_size=1)


    def forward(self, x):
        # Encoder forward
        e0 = self.encoder0(x) # (B, 64, H, W)
        x = self.maxpool0(e0)

        e1 = self.encoder1(x) # (B, 128, H/2, W/2)
        x = self.maxpool1(e1)

        e2 = self.encoder2(x) # (B, 256, H/4, W/4)
        x = self.maxpool2(e2)

        e3 = self.encoder3(x) # (B, 512, H/8, W/8)
        x = self.maxpool3(e3)

        e4 = self.encoder4(x) # (B, 1024, H/16, W/16) -- Bottle Neck

        # Decoder forward
        # Level 3 decoder
        d3_0 = self.up3_0(e4) # (B, 512, H/8, W/8)
        d3_0 = torch.cat([d3_0, e3], dim=1) 
        d3_0 = self.dense3_0(d3_0)

        # Level 2 decoder
        d2_0 = self.up2_0(d3_0) # (B, 256, H/4, W/4)
        d2_0 = torch.cat([d2_0, e2], dim=1)
        d2_0 = self.dense2_0(d2_0)

        d2_1 = self.up2_1(d3_0) # (B, 256, H/4, W/4)
        d2_1 = torch.cat([d2_1, e2, d2_0], dim=1)
        d2_1 = self.dense2_1(d2_1)

        # Level 1 decoder
        d1_0 = self.up1_0(d2_0) # (B, 128, H/2, W/2)
        d1_0 = torch.cat([d1_0, e1], dim=1)
        d1_0 = self.dense1_0(d1_0)

        d1_1 = self.up1_1(d2_1) # (B, 128, H/2, W/2)
        d1_1 = torch.cat([d1_1, e1, d1_0], dim=1)
        d1_1 = self.dense1_1(d1_1)

        d1_2 = self.up1_2(d2_1) # (B, 128, H/2, W/2)
        d1_2 = torch.cat([d1_2, e1, d1_0, d1_1], dim=1)
        d1_2 = self.dense1_2(d1_2)

        # Level 0 decoder
        d0_0 = self.up0_0(d1_0)  # (B, 64, H, W)
        d0_0 = torch.cat([d0_0, e0], dim=1)
        d0_0 = self.dense0_0(d0_0)
        
        d0_1 = self.up0_1(d1_1)  # (B, 64, H, W)
        d0_1 = torch.cat([d0_1, e0, d0_0], dim=1)
        d0_1 = self.dense0_1(d0_1)
        
        d0_2 = self.up0_2(d1_2)  # (B, 64, H, W)
        d0_2 = torch.cat([d0_2, e0, d0_0, d0_1], dim=1)
        d0_2 = self.dense0_2(d0_2)
        
        d0_3 = self.up0_3(d1_2)  # (B, 64, H, W)
        d0_3 = torch.cat([d0_3, e0, d0_0, d0_1, d0_2], dim=1)
        d0_3 = self.dense0_3(d0_3)

        # Final output
        output = self.final(d0_3)
        return output

# DiceLoss function
class DiceLoss(nn.Module):
    """Dice Loss"""
    def __init__(self, smooth: float = 1.0, epsilon: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.epsilon = epsilon
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply softmax to get probabilities
        predictions = F.softmax(predictions, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets.long(), num_classes=predictions.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Flatten spatial dimensions
        predictions = predictions.reshape(predictions.shape[0], predictions.shape[1], -1)
        targets_one_hot = targets_one_hot.reshape(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
        
        # Calculate Dice Loss for each class
        intersection = 2.0 * (predictions * targets_one_hot).sum(dim=2)
        union = predictions.sum(dim=2) + targets_one_hot.sum(dim=2)
        dice = (intersection + self.smooth) / (union + self.smooth + self.epsilon)
        
        return 1.0 - dice.mean()

# PyTorch Lightning Module
class UNetPPLightning(pl.LightningModule):
    def __init__(
        self, 
        in_channels: int = 3, 
        out_channels: int = 3, 
        use_dice_loss: bool = False, 
        classes: int = 3, 
        lr: float = 1e-3, 
        max_epochs: int = 50, 
        weight_decay: float = 1e-6
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = UNetPlusPlus(in_channels=in_channels, classes=classes)
        self.lr = lr
        self.classes = classes
        self.max_epochs = max_epochs
    
        # Loss functions
        self.use_dice_loss = use_dice_loss
        self.ce_loss = nn.CrossEntropyLoss()
        if use_dice_loss:
            self.dice_loss = SMPDiceLoss(mode="multiclass")

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
                dice_loss_value = self.dice_loss(logits, masks)
                ce_loss_value = self.ce_loss(logits, masks)
                loss = 0.5 * dice_loss_value + 0.5 * ce_loss_value
                
                # Log individual loss components for W&B
                self.log("train_dice_loss", dice_loss_value, on_step=False, on_epoch=True, prog_bar=False)
                self.log("train_ce_loss", ce_loss_value, on_step=False, on_epoch=True, prog_bar=False)
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
        
        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Update and log metrics
        self.train_iou.update(preds, masks)
        self.log(
            "train_mIoU",
            self.train_iou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        
        # Log learning rate to W&B
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", current_lr, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, masks = self.step(batch)
        
        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Update metrics
        self.val_iou.update(preds, masks)
        self.val_dice.update(preds, masks)

        # Log metrics
        self.log(
            "val_mIoU", self.val_iou, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val_Dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        loss, preds, masks = self.step(batch)
        
        # Log test metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_mIoU", self.val_iou(preds, masks), on_step=False, on_epoch=True)
        self.log("test_Dice", self.val_dice(preds, masks), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
        )

        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
    
    def on_train_epoch_end(self):
        """Log additional metrics at the end of each epoch"""
        # This is useful for custom logging to W&B
        pass
    
    def on_validation_epoch_end(self):
        """Log additional metrics at the end of validation"""
        # You can add custom visualizations here
        pass