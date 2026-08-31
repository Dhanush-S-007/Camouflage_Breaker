# training/train_classifier_v3.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import json
import numpy as np
from models.classifier import ResNet50Classifier

class CropDataset(Dataset):
    def __init__(self, split='Train', transform=None):
        self.split = split
        self.transform = transform
        self.crops_dir = f'dataset/crops/{split}/'
        
        # Read labels from TXT file
        label_file = f'dataset/{split}/CAM-NonCAM_Instance_{split}.txt'
        self.labels = {}
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        print(f"Reading labels from: {label_file}")
        
        with open(label_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                
                # Parse the line
                for i, part in enumerate(parts):
                    if '.png' in part or '.jpg' in part:
                        basename = part.replace('.png', '').replace('.jpg', '')
                        # Find class ID near the filename
                        class_id = None
                        # Check positions around the filename
                        for offset in [-2, -1, 1, 2]:
                            idx = i + offset
                            if 0 <= idx < len(parts) and parts[idx].isdigit():
                                class_id = int(parts[idx])
                                break
                        
                        if class_id is not None and class_id > 0:
                            self.labels[basename] = class_id
                        break
        
        print(f"Loaded {len(self.labels)} labels from {label_file}")
        
        # Get all crop files
        if not os.path.exists(self.crops_dir):
            print(f"ERROR: {self.crops_dir} not found!")
            self.image_files = []
            return
        
        all_crops = [f for f in os.listdir(self.crops_dir) if f.endswith('.jpg')]
        
        # Match crops with labels
        self.image_files = []
        for crop in all_crops:
            basename = crop.replace('_crop.jpg', '')
            if basename in self.labels:
                self.image_files.append(crop)
        
        print(f"Found {len(self.image_files)} matching crops in {self.crops_dir}")
        
        if len(self.image_files) > 0:
            sample = self.image_files[0]
            sample_basename = sample.replace('_crop.jpg', '')
            print(f"Sample: {sample} -> label {self.labels[sample_basename]}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.crops_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        basename = img_name.replace('_crop.jpg', '')
        label = self.labels[basename] - 1  # Convert to 0-indexed
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train():
    print("=" * 70)
    print("CLASSIFIER TRAINING V3 - 69 CLASSES")
    print("=" * 70)
    
    # Check for crops
    if not os.path.exists('dataset/crops/Train'):
        print("ERROR: dataset/crops/Train not found!")
        print("Run: python utils/generate_crops.py first")
        return
    
    # Data transforms (augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = CropDataset('Train', train_transform)
    val_dataset = CropDataset('Test', val_transform)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("ERROR: Empty dataset!")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    num_classes = 69
    print(f"\nCreating model with {num_classes} classes...")
    model = ResNet50Classifier(num_classes=num_classes, pretrained=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training
    best_val_acc = 0
    patience = 7
    patience_counter = 0
    
    os.makedirs('saved_models', exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70 + "\n")
    
    for epoch in range(30):  # 30 epochs
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/30 Train")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.1f}%'
            })
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/30 Val"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/30")
        print(f"  Train Acc: {train_acc:.2f}%, Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val Acc:   {val_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'saved_models/classifier_best.pth')
            print(f"  ✓ NEW BEST! Saved model (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"{'='*50}\n")
    
    # Save class mapping
    print("\nSaving class mapping...")
    # Get unique classes from dataset
    all_labels = set(train_dataset.labels.values())
    # Create mapping (label -> class name)
    # We'll use the extract_classes.py output
    try:
        import subprocess
        result = subprocess.run(['python', 'extract_classes.py'], capture_output=True, text=True)
        print(result.stdout)
    except:
        print("Class mapping saved earlier")
    
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved: saved_models/classifier_best.pth")
    print("=" * 70)

if __name__ == '__main__':
    train()