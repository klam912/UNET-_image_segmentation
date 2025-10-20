"""
viz.py - Visualize UNet++ model predictions vs ground truth
"""
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

from datamodule_oxpet import OxPetDataModule
from model_unetpp import UNetPPLightning


# Color palette for trimap classes
TRIMAP_COLORS = {
    0: (220, 20, 60),    # Pet (Crimson Red)
    1: (119, 135, 150),  # Background (Gray)
    2: (34, 139, 34)     # Border (Forest Green)
}

BINARY_COLORS = {
    0: (119, 135, 150),  # Background (Gray)
    1: (220, 20, 60)     # Pet (Crimson Red)
}

CLASS_NAMES = {
    'trimap': ['Pet', 'Background', 'Border'],
    'binary': ['Background', 'Pet']
}


def colorize_mask(mask, mode='trimap'):
    """
    Convert a segmentation mask to RGB visualization.
    
    Args:
        mask: numpy array or torch tensor of shape (H, W) with class indices
        mode: 'trimap' (3 classes) or 'binary' (2 classes)
    
    Returns:
        PIL Image in RGB format
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    mask = mask.astype(np.int64)
    H, W = mask.shape
    
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    colors = TRIMAP_COLORS if mode == 'trimap' else BINARY_COLORS
    
    for class_idx, color in colors.items():
        rgb[mask == class_idx] = color
    
    return Image.fromarray(rgb, mode='RGB')


def overlay_mask(image, mask_rgb, alpha=0.45):
    """
    Overlay a colored mask on top of an image.
    
    Args:
        image: PIL Image or torch tensor (C, H, W) in range [0, 1]
        mask_rgb: PIL Image (colored mask)
        alpha: transparency of mask overlay (0=invisible, 1=opaque)
    
    Returns:
        PIL Image with overlaid mask
    """
    if isinstance(image, torch.Tensor):
        image = TF.to_pil_image(image.cpu())
    
    image = image.convert('RGB')
    
    if image.size != mask_rgb.size:
        mask_rgb = mask_rgb.resize(image.size, resample=Image.NEAREST)
    
    return Image.blend(image, mask_rgb, alpha)


def denormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Denormalize an image tensor back to [0, 1] range.
    
    Args:
        tensor: torch tensor (C, H, W) normalized with ImageNet stats
        mean: normalization mean used during training
        std: normalization std used during training
    
    Returns:
        torch tensor (C, H, W) in range [0, 1]
    """
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    return tensor * std + mean


def create_comparison_grid(images, gt_masks, pred_masks, mode='trimap', 
                           denorm=True, save_path=None):
    """
    Create a grid showing: Image | GT Mask | Pred Mask | GT Overlay | Pred Overlay.
    
    Args:
        images: list of torch tensors (C, H, W) or PIL Images
        gt_masks: list of ground truth masks (H, W)
        pred_masks: list of predicted masks (H, W)
        mode: 'trimap' or 'binary'
        denorm: whether to denormalize images
        save_path: optional path to save the grid
    
    Returns:
        PIL Image containing the comparison grid
    """
    n_samples = len(images)
    rows = []
    
    for i in range(n_samples):
        img = images[i]
        
        # Denormalize if needed
        if isinstance(img, torch.Tensor) and denorm:
            img = denormalize_image(img)
        
        # Convert to PIL
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img.cpu())
        
        # Colorize masks
        gt_colored = colorize_mask(gt_masks[i], mode=mode)
        pred_colored = colorize_mask(pred_masks[i], mode=mode)
        
        # Create overlays
        gt_overlay = overlay_mask(img.copy(), gt_colored, alpha=0.45)
        pred_overlay = overlay_mask(img.copy(), pred_colored, alpha=0.45)
        
        # Create row: [Image, GT, Pred, GT Overlay, Pred Overlay]
        row_images = [img, gt_colored, pred_colored, gt_overlay, pred_overlay]
        
        # Ensure all same size
        size = img.size
        row_images = [im.resize(size, Image.NEAREST) if im.size != size else im 
                     for im in row_images]
        
        # Add text labels above each column
        labeled_images = []
        labels = ["Image", "GT Mask", "Pred Mask", "GT Overlay", "Pred Overlay"]
        
        for im, label in zip(row_images, labels):
            im_with_label = Image.new('RGB', (im.width, im.height + 25), color=(255, 255, 255))
            im_with_label.paste(im, (0, 25))
            
            draw = ImageDraw.Draw(im_with_label)
            draw.text((5, 5), label, fill=(0, 0, 0))
            labeled_images.append(im_with_label)
        
        # Concatenate horizontally
        row_width = sum(im.width for im in labeled_images)
        row_height = labeled_images[0].height
        row = Image.new('RGB', (row_width, row_height), color=(255, 255, 255))
        
        x_offset = 0
        for im in labeled_images:
            row.paste(im, (x_offset, 0))
            x_offset += im.width
        
        rows.append(row)
    
    # Concatenate all rows vertically
    if rows:
        width = rows[0].size[0]
        height = sum(row.size[1] for row in rows)
        grid = Image.new('RGB', (width, height), color=(255, 255, 255))
        
        y_offset = 0
        for row in rows:
            grid.paste(row, (0, y_offset))
            y_offset += row.size[1]
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            grid.save(save_path)
            print(f"✓ Saved visualization to {save_path}")
        
        return grid
    
    return None


def save_single_prediction(image, gt_mask, pred_mask, save_path, mode='trimap', denorm=True):
    """
    Save a single prediction with image, GT, prediction, and overlays.
    
    Args:
        image: torch tensor (C, H, W) or PIL Image
        gt_mask: ground truth mask (H, W)
        pred_mask: predicted mask (H, W)
        save_path: where to save the result
        mode: 'trimap' or 'binary'
        denorm: whether to denormalize image
    """
    grid = create_comparison_grid(
        [image], [gt_mask], [pred_mask],
        mode=mode, denorm=denorm, save_path=save_path
    )
    return grid


def predict_single_image(image_path, mask_path, checkpoint_path, mode='trimap', output_path=None):
    """
    Make a prediction on a single image and visualize results.
    
    Args:
        image_path: path to input image
        mask_path: path to ground truth mask
        checkpoint_path: path to model checkpoint
        mode: 'trimap' or 'binary'
        output_path: optional path to save visualization
    
    Returns:
        PIL Image with comparison grid
    """
    print(f"\n{'='*60}")
    print("SINGLE IMAGE PREDICTION")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = UNetPPLightning.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}\n")
    
    # Load image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    # Load mask
    print(f"Loading mask: {mask_path}")
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)
    
    # Preprocess image (same as training)
    from torchvision.transforms import Compose, Normalize, ToTensor, Resize
    transforms = Compose([
        Resize((384, 384)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transforms(image).unsqueeze(0).to(device)
    
    # Resize mask to match preprocessed image size
    mask_pil = Image.fromarray(mask)
    mask_pil = mask_pil.resize((384, 384), Image.NEAREST)
    mask = np.array(mask_pil)
    
    print(f"Image shape: {image_tensor.shape}")
    print(f"Mask shape: {mask.shape}\n")
    
    # Get prediction
    with torch.no_grad():
        logits = model(image_tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0)
    
    # Visualize
    grid = create_comparison_grid(
        [image_tensor[0]],
        [torch.tensor(mask)],
        [pred],
        mode=mode,
        denorm=True,
        save_path=output_path
    )
    
    print(f"{'='*60}")
    print("PREDICTION COMPLETE")
    print(f"{'='*60}\n")
    
    return grid


def visualize_batch(checkpoint_path, data_split="val", num_batches=2, 
                    samples_per_batch=4, mode='trimap', output_dir='./visualizations'):
    """
    Visualize predictions on batches from dataloader.
    
    Args:
        checkpoint_path: path to model checkpoint
        data_split: 'train', 'val', or 'test'
        num_batches: number of batches to visualize
        samples_per_batch: samples to show per batch
        mode: 'trimap' (3 classes) or 'binary' (2 classes)
        output_dir: directory to save visualizations
    """
    print(f"\n{'='*60}")
    print("BATCH VISUALIZATION")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = UNetPPLightning.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data module...")
    dm = OxPetDataModule(batch_size=4, num_workers=4)
    dm.prepare_data()
    dm.setup("fit")
    
    # Get appropriate dataloader
    if data_split == "val":
        dataloader = dm.val_dataloader()
    elif data_split == "test":
        dm.setup("test")
        dataloader = dm.test_dataloader()
    else:
        dataloader = dm.train_dataloader()
    
    print(f"Visualizing {num_batches} batches from {data_split} set (mode: {mode})...\n")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Visualize batches
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            images, masks = batch
            images = images.to(device)
            
            # Get predictions
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            # Limit samples per batch
            n_samples = min(samples_per_batch, images.shape[0])
            images_subset = images[:n_samples].cpu()
            masks_subset = masks[:n_samples]
            preds_subset = preds[:n_samples]
            
            # Create visualization
            save_path = Path(output_dir) / f"predictions_{data_split}_batch{batch_idx}.png"
            print(f"Batch {batch_idx + 1}/{num_batches}:")
            
            create_comparison_grid(
                list(images_subset),
                list(masks_subset),
                list(preds_subset),
                mode=mode,
                denorm=True,
                save_path=str(save_path)
            )
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize UNet++ model predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single image for prediction",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to ground truth mask (required if --image is used)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help="Data split to visualize (ignored if --image is used)",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=2,
        help="Number of batches to visualize",
    )
    parser.add_argument(
        "--samples_per_batch",
        type=int,
        default=4,
        help="Number of samples to show per batch",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["trimap", "binary"],
        default="trimap",
        help="Visualization mode (3 or 2 classes)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations",
        help="Directory to save visualizations",
    )
    
    args = parser.parse_args()
    
    # Handle wildcard checkpoint paths
    checkpoint_path = args.checkpoint
    if '*' in checkpoint_path:
        parent_dir = Path(checkpoint_path).parent
        pattern = Path(checkpoint_path).name
        matches = list(parent_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No checkpoints matching: {checkpoint_path}")
        checkpoint_path = str(matches[0])
        print(f"Found checkpoint: {checkpoint_path}\n")
    
    # Check checkpoint exists
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Single image mode
    if args.image:
        if not args.mask:
            raise ValueError("--mask is required when using --image")
        if not Path(args.image).exists():
            raise FileNotFoundError(f"Image not found: {args.image}")
        if not Path(args.mask).exists():
            raise FileNotFoundError(f"Mask not found: {args.mask}")
        
        output_path = Path(args.output_dir) / "single_prediction.png"
        predict_single_image(
            args.image,
            args.mask,
            checkpoint_path,
            mode=args.mode,
            output_path=str(output_path)
        )
    else:
        # Batch mode
        visualize_batch(
            checkpoint_path,
            data_split=args.split,
            num_batches=args.num_batches,
            samples_per_batch=args.samples_per_batch,
            mode=args.mode,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()