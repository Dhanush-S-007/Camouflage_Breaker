import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Classifier(nn.Module):
    """
    ResNet50 classifier for camouflaged animal recognition.
    Input: cropped animal image (B, 3, 224, 224)
    Output: class probabilities (B, num_classes)
    """
    
    def __init__(self, num_classes=69, pretrained=True):
        super(ResNet50Classifier, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Replace final FC layer
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
    
    def predict(self, x):
        """Inference with softmax."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            return probs


# ==================== QUICK TEST ====================
if __name__ == "__main__":
    print("=" * 50)
    print("ResNet50 Classifier Test")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = ResNet50Classifier(num_classes=69, pretrained=True)
    model = model.to(device)
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable: {trainable:,}")
    
    dummy = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy)
    print(f"Input: {dummy.shape}")
    print(f"Output: {output.shape}")
    print(f"Output sum: {output.sum(dim=1)}")  # Should be ~1.0 per sample (softmax)
    
    probs = model.predict(dummy)
    pred_class = torch.argmax(probs, dim=1)
    print(f"Predicted classes: {pred_class}")
    print(f"Confidence: {probs.max(dim=1).values}")
    
    print("\n[OK] Classifier test passed!")