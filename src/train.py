"""
train.py - CLI to train and evaluate UNet++ model with PyTorch Lightning
"""
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from datamodule_oxpet import OxPetDataModule
from model_unetpp import UNetPPLightning


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate UNet++ for image segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "test", "both"], 
        default="both",
        help="Mode: train only, test only, or both"
    )
    
    # Data
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    # Model
    parser.add_argument("--in_channels", type=int, default=3, help="Input channels")
    parser.add_argument("--out_channels", type=int, default=3, help="Output channels (3 for pet/background/border)")
    
    # Training
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--precision", type=str, default="16-mixed", 
                       choices=["32", "16-mixed", "bf16-mixed"], help="Training precision")
    
    # Callbacks
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--no_early_stop", action="store_true", help="Disable early stopping")
    
    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for testing/resuming")
    parser.add_argument("--save_dir", type=str, default="./logs", help="Directory to save logs and checkpoints")
    
    # Hardware
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--accelerator", type=str, default="gpu", 
                       choices=["cpu", "gpu", "tpu"], help="Accelerator type")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fast_dev_run", action="store_true", help="Quick run for debugging")
    
    return parser.parse_args()


def train(args, dm, model, logger):
    """Train the model."""
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60 + "\n")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_Dice",
        mode="max",
        save_top_k=1,
        filename="best-unetpp-{epoch:02d}-{val/dice:.4f}",
        verbose=True,
    )
    
    callbacks = [checkpoint_callback]
    
    # Early stopping
    if not args.no_early_stop:
        early_stop = EarlyStopping(
            monitor="val_Dice",
            patience=args.patience,
            mode="max",
            verbose=True,
        )
        callbacks.append(early_stop)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.gpus,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
    )
    
    # Train
    trainer.fit(model, dm, ckpt_path=args.checkpoint)
    
    print(f"\n✓ Training complete!")
    print(f"✓ Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"✓ Best val/dice: {checkpoint_callback.best_model_score:.4f}\n")
    
    return trainer, checkpoint_callback.best_model_path


def evaluate(args, dm, checkpoint_path=None):
    """Evaluate the model on validation and test sets."""
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60 + "\n")
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = args.checkpoint
    
    if checkpoint_path is None:
        raise ValueError("No checkpoint provided for evaluation. Use --checkpoint or train first.")
    
    print(f"Loading model from: {checkpoint_path}\n")
    model = UNetPPLightning.load_from_checkpoint(checkpoint_path)
    
    # Trainer for evaluation
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.gpus,
        logger=False,
    )
    
    # Setup test data
    dm.setup("test")
    
    # Validate
    print("Validating on validation set...")
    val_results = trainer.validate(model, datamodule=dm)
    print(f"\nValidation Results:")
    for key, value in val_results[0].items():
        print(f"  {key}: {value:.6f}")
    
    # Test
    print("\nTesting on test set...")
    test_results = trainer.test(model, datamodule=dm)
    print(f"\nTest Results:")
    for key, value in test_results[0].items():
        print(f"  {key}: {value:.6f}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60 + "\n")
    
    return val_results, test_results


def main():
    """Main function."""
    args = parse_args()
    
    # Set seed
    pl.seed_everything(args.seed)
    
    # Print configuration
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("="*60 + "\n")
    
    # Check GPU
    if args.accelerator == "gpu":
        if not torch.cuda.is_available():
            print("⚠ WARNING: GPU requested but not available. Falling back to CPU.")
            args.accelerator = "cpu"
        else:
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA Version: {torch.version.cuda}\n")
    
    # Data module
    print("Loading data...")
    dm = OxPetDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    dm.prepare_data()
    dm.setup("fit")
    print(f"✓ Data loaded: {len(dm.train_dataloader())} train batches, "
          f"{len(dm.val_dataloader())} val batches\n")
    
    # Logger
    logger = CSVLogger(save_dir=args.save_dir, name="UNetPP")
    
    # Model
    model = UNetPPLightning(
       in_channels=args.in_channels,
       out_channels=args.out_channels,
       lr=args.lr,
        classes=args.out_channels, 
       weight_decay=args.weight_decay,
       max_epochs=args.max_epochs,
   )
    
    # Execute based on mode
    checkpoint_path = None
    
    if args.mode in ["train", "both"]:
        trainer, checkpoint_path = train(args, dm, model, logger)
    
    if args.mode in ["test", "both"]:
        evaluate(args, dm, checkpoint_path)


if __name__ == "__main__":
    main()