# training/train_classifier_final.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import json
import numpy as np

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=69):
        super(ResNet50Classifier, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class CropDataset(Dataset):
    def __init__(self, split='Train', transform=None):
        self.transform = transform
        self.crops_dir = f'dataset/crops/{split}/'
        self.image_files = []
        self.labels = []
        
        # Get all crop files
        all_files = [f for f in os.listdir(self.crops_dir) if f.endswith('.jpg')]
        
        for filename in all_files:
            # Extract class ID from filename
            # Format: COD10K-CAM-1-Aquatic-1-BatFish-2_crop.jpg
            parts = filename.split('-')
            if len(parts) >= 6:
                try:
                    class_id = int(parts[4])  # Class ID is at position 4
                    if 1 <= class_id <= 69:
                        self.image_files.append(filename)
                        self.labels.append(class_id - 1)  # Convert to 0-indexed
                except ValueError:
                    continue
        
        print(f"{split}: Loaded {len(self.image_files)} images with labels")
        
        # Print some samples
        if len(self.image_files) > 0:
            print(f"Sample: {self.image_files[0]} -> label {self.labels[0] + 1}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.crops_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train():
    print("=" * 70)
    print("CLASSIFIER TRAINING - FINAL VERSION")
    print("=" * 70)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = CropDataset('Train', train_transform)
    val_dataset = CropDataset('Test', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # Model
    model = ResNet50Classifier(num_classes=69)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # Training
    best_acc = 0
    os.makedirs('saved_models', exist_ok=True)
    
    for epoch in range(20):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/20")
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
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"\nEpoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'saved_models/classifier_best.pth')
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE! Best validation accuracy: {best_acc:.2f}%")
    print("=" * 70)
    
    # Verify
    checkpoint = torch.load('saved_models/classifier_best.pth', map_location='cpu')
    weights = checkpoint['backbone.fc.4.weight']
    print(f"Model Std: {weights.std().item()}")

if __name__ == '__main__':
    train()