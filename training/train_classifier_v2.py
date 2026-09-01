# training/train_classifier_v2.py
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
from models.classifier import ResNet50Classifier

class CropDataset(Dataset):
    def __init__(self, split='Train', transform=None):
        self.split = split
        self.transform = transform
        self.crops_dir = f'dataset/crops/{split}/'
        
        label_file = f'dataset/{split}/CAM-NonCAM_Instance_{split}.txt'
        self.labels = {}
        
        print(f"Reading labels from: {label_file}")
        
        with open(label_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                
                # Find the filename and extract class ID
                for i, part in enumerate(parts):
                    if '.png' in part or '.jpg' in part:
                        basename = part.replace('.png', '').replace('.jpg', '')
                        
                        # CORRECT: Extract class ID from filename
                        # Format: COD10K-CAM-1-Aquatic-1-BatFish-2.jpg
                        # Class ID is at position 4 (0-indexed)
                        filename_parts = basename.split('-')
                        if len(filename_parts) >= 6:
                            try:
                                class_id = int(filename_parts[4])  # This is the class ID!
                                self.labels[basename] = class_id
                            except ValueError:
                                pass
                        break
        
        print(f"Loaded {len(self.labels)} labels")
        
        all_crops = [f for f in os.listdir(self.crops_dir) if f.endswith('.jpg')]
        self.image_files = []
        for crop in all_crops:
            basename = crop.replace('_crop.jpg', '')
            if basename in self.labels:
                self.image_files.append(crop)
        
        print(f"{split}: {len(self.image_files)} matching crops loaded")
        
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
    print("CLASSIFIER TRAINING - CORRECTED LABELS")
    print("=" * 70)
    
    # Transforms
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
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Model
    num_classes = 69
    model = ResNet50Classifier(num_classes=num_classes, pretrained=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training
    best_val_acc = 0
    patience_counter = 0
    os.makedirs('saved_models', exist_ok=True)
    
    print("\nStarting training...\n")
    
    for epoch in range(30):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/30 Train"):
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
        
        print(f"\nEpoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'saved_models/classifier_best.pth')
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE! Best validation accuracy: {best_val_acc:.2f}%")
    print("=" * 70)

if __name__ == '__main__':
    train()