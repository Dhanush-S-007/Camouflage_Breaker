import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Add project root to path so we can import utils and models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resunet import ResUNet
from utils.dataset import COD10KDataset, get_dataloader
from utils.metrics import BCEDiceLoss, iou_score, dice_score


# ==================== CONFIG ====================
class Config:
    """Training hyperparameters and paths."""
    DATASET_ROOT = "dataset"
    SPLIT = "Train"
    IMAGE_SIZE = 352
    BATCH_SIZE = 4          # Keep small for CPU; increase to 8-16 if you have GPU
    NUM_WORKERS = 0         # 0 for Windows to avoid multiprocessing issues
    EPOCHS = 20             # Start with 20; increase if validation improves
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    VAL_RATIO = 0.1         # 10% validation split
    SEED = 42
    
    CHECKPOINT_DIR = "saved_models"
    CHECKPOINT_NAME = "resunet_best.pth"
    SAVE_INTERVAL = 5       # Save every N epochs (plus best model)
    
    # Loss weights
    BCE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5
    
    # Early stopping
    PATIENCE = 5            # Stop if val IoU doesn't improve for 5 epochs


def set_seed(seed=42):
    """For reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloaders(config):
    """
    Create train and validation dataloaders from COD10K Train split.
    Uses 90/10 random split.
    """
    # Load full dataset
    full_dataset = COD10KDataset(
        root_dir=config.DATASET_ROOT,
        split=config.SPLIT,
        image_size=config.IMAGE_SIZE,
        return_label=False
    )
    
    total = len(full_dataset)
    val_size = int(total * config.VAL_RATIO)
    train_size = total - val_size
    
    print(f"[INFO] Total training samples: {total}")
    print(f"[INFO] Train split: {train_size} | Val split: {val_size}")
    
    train_set, val_set = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_batches = len(loader)
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            batch_iou = iou_score(probs, masks)
            batch_dice = dice_score(probs, masks)
        
        total_loss += loss.item()
        total_iou += batch_iou
        total_dice += batch_dice
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'iou': f"{batch_iou:.4f}",
            'dice': f"{batch_dice:.4f}"
        })
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches
    
    return avg_loss, avg_iou, avg_dice


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_batches = len(loader)
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        logits = model(images)
        loss = criterion(logits, masks)
        probs = torch.sigmoid(logits)
        
        batch_iou = iou_score(probs, masks)
        batch_dice = dice_score(probs, masks)
        
        total_loss += loss.item()
        total_iou += batch_iou
        total_dice += batch_dice
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'iou': f"{batch_iou:.4f}",
            'dice': f"{batch_dice:.4f}"
        })
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches
    
    return avg_loss, avg_iou, avg_dice


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(checkpoint, path)
    print(f"[OK] Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, scheduler, path):
    """Load model checkpoint if exists."""
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"[OK] Resumed from checkpoint: {path} (epoch {start_epoch-1})")
        return start_epoch
    return 0


def main():
    print("=" * 60)
    print("Camouflage Breaker - Segmentation Training")
    print("=" * 60)
    
    config = Config()
    set_seed(config.SEED)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    
    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)
    
    # DataLoaders
    print("\n[INFO] Loading dataset...")
    train_loader, val_loader = get_dataloaders(config)
    
    # Model
    print("\n[INFO] Building ResUNet...")
    model = ResUNet(encoder_name="resnet50", encoder_weights="imagenet")
    model = model.to(device)
    
    # Loss
    criterion = BCEDiceLoss(
        bce_weight=config.BCE_WEIGHT,
        dice_weight=config.DICE_WEIGHT
    )
    
    # Optimizer & Scheduler
    optimizer = Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3
    )
    
    # Resume if checkpoint exists
    start_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
    
    # Training history
    history = {
        'train_loss': [], 'train_iou': [], 'train_dice': [],
        'val_loss': [], 'val_iou': [], 'val_dice': []
    }
    
    best_val_iou = 0.0
    epochs_no_improve = 0
    
    print(f"\n[INFO] Starting training for {config.EPOCHS} epochs...")
    print("-" * 60)
    
    total_start = time.time()
    
    for epoch in range(start_epoch, config.EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss, train_iou, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_iou, val_dice = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Scheduler step
        scheduler.step(val_iou)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(
            f"Epoch {epoch:02d}/{config.EPOCHS} | "
            f"Time: {epoch_time:.1f}s | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}\n"
            f"  Train -> Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | Dice: {train_dice:.4f}\n"
            f"  Val   -> Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}"
        )
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)
            print(f"  [★] New best val IoU: {best_val_iou:.4f} -> Saved!")
        else:
            epochs_no_improve += 1
            print(f"  [ ] No improvement ({epochs_no_improve}/{config.PATIENCE})")
        
        # Periodic save
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            periodic_path = os.path.join(
                config.CHECKPOINT_DIR, 
                f"resunet_epoch_{epoch+1}.pth"
            )
            save_checkpoint(model, optimizer, scheduler, epoch, periodic_path)
        
        # Early stopping
        if epochs_no_improve >= config.PATIENCE:
            print(f"\n[STOP] Early stopping triggered after {epoch+1} epochs.")
            break
        
        print("-" * 60)
    
    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"{'=' * 60}")
    
    return history


if __name__ == "__main__":
    main()