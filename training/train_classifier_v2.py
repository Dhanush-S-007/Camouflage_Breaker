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
        
        # Read labels from the TXT file
        label_file = f'dataset/{split}/CAM-NonCAM_Instance_{split}.txt'
        self.labels = {}
        
        print(f"Reading labels from: {label_file}")
        
        with open(label_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                
                # Find the filename (contains .png or .jpg)
                for i, part in enumerate(parts):
                    if '.png' in part or '.jpg' in part:
                        basename = part.replace('.png', '').replace('.jpg', '')
                        # Look for a number near the filename
                        # Try different positions
                        category = 0
                        for offset in [-2, -1, 1, 2]:
                            idx = i + offset
                            if 0 <= idx < len(parts) and parts[idx].isdigit():
                                category = int(parts[idx])
                                break
                        self.labels[basename] = category
                        break
        
        print(f"Loaded {len(self.labels)} labels")
        
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
        
        print(f"Found {len(self.image_files)} matching crops")
        
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
        label = self.labels[basename]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train():
    print("=" * 60)
    print("CLASSIFIER TRAINING STARTING")
    print("=" * 60)
    
    # 1. Check for crops
    if not os.path.exists('dataset/crops/Train'):
        print("ERROR: dataset/crops/Train not found!")
        print("Run: python utils/generate_crops.py first")
        return
    
    # 2. Data transforms
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
    
    # 3. Create datasets
    print("\nLoading datasets...")
    train_dataset = CropDataset('Train', train_transform)
    val_dataset = CropDataset('Test', val_transform)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("ERROR: Empty dataset!")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # 4. Model
    num_classes = 69
    print(f"\nCreating model with {num_classes} classes...")
    model = ResNet50Classifier(num_classes=num_classes, pretrained=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # 5. Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # 6. Training loop
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    os.makedirs('saved_models', exist_ok=True)
    
    print("\nStarting training...")
    print("-" * 60)
    
    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")
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
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Acc: {train_acc:.2f}%, Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val Acc: {val_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}")
        
        scheduler.step(val_loss)
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'saved_models/classifier_best.pth')
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # 7. Save class mapping
    all_labels = set(train_dataset.labels.values())
    class_map = {i: f"class_{i}" for i in sorted(all_labels)}
    with open('saved_models/class_mapping.json', 'w') as f:
        json.dump(class_map, f)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved: saved_models/classifier_best.pth")
    print(f"Class mapping saved: saved_models/class_mapping.json")
    print("=" * 60)

if __name__ == '__main__':
    train()