import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import random

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """
    CutMix data augmentation.
    
    Args:
        x: Input batch
        y: Target batch
        alpha: CutMix parameter
        device: Device to run on
    
    Returns:
        mixed_x: Mixed input batch
        y_a, y_b: Original targets for loss calculation
        lam: Mixing ratio
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if device == 'cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    # Generate random bounding box
    W = x.size(2)
    H = x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """
    CutMix loss calculation.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross entropy loss.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdvancedAugmentation:
    """
    Advanced augmentation techniques for better generalization.
    """
    def __init__(self, image_size=224):
        self.image_size = image_size
        
    def get_train_transforms(self):
        """
        Get comprehensive training transforms with advanced augmentation.
        """
        return transforms.Compose([
            # Resize with padding
            transforms.Resize((self.image_size + 32, self.image_size + 32)),
            
            # Random crop with padding
            transforms.RandomCrop((self.image_size, self.image_size), padding=4),
            
            # Geometric augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.1, 0.1), 
                scale=(0.9, 1.1),
                shear=5
            ),
            
            # Color augmentations
            transforms.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.05
            ),
            
            # Advanced augmentations
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            
            # TrivialAugmentWide for additional randomness
            transforms.TrivialAugmentWide(num_magnitude_bins=31),
            
            # Convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_val_transforms(self):
        """
        Get validation transforms (minimal augmentation).
        """
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class TestTimeAugmentation:
    """
    Test Time Augmentation for better inference.
    """
    def __init__(self, model, device, num_augmentations=5):
        self.model = model
        self.device = device
        self.num_augmentations = num_augmentations
        
        # Define TTA transforms
        self.tta_transforms = [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ]
    
    def predict(self, image):
        """
        Predict with TTA.
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for transform in self.tta_transforms[:self.num_augmentations]:
                # Apply transform
                if hasattr(image, 'convert'):
                    # PIL Image
                    transformed = transform(image)
                else:
                    # Already tensor
                    transformed = transform(image)
                
                # Add batch dimension
                transformed = transformed.unsqueeze(0).to(self.device)
                
                # Get prediction
                output = self.model(transformed)
                predictions.append(F.softmax(output, dim=1))
        
        # Average predictions
        avg_prediction = torch.mean(torch.stack(predictions), dim=0)
        return avg_prediction

def calculate_class_weights(train_targets, num_classes, method='balanced'):
    """
    Calculate class weights for handling class imbalance.
    
    Args:
        train_targets: List of training targets
        num_classes: Number of classes
        method: 'balanced', 'inverse', or 'sqrt'
    
    Returns:
        class_weights: Tensor of class weights
    """
    class_counts = np.bincount(train_targets, minlength=num_classes)
    
    if method == 'balanced':
        # sklearn's balanced method
        total_samples = len(train_targets)
        class_weights = total_samples / (num_classes * class_counts)
    elif method == 'inverse':
        # Simple inverse frequency
        class_weights = 1.0 / class_counts
    elif method == 'sqrt':
        # Square root of inverse frequency
        class_weights = 1.0 / np.sqrt(class_counts)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    return torch.tensor(class_weights, dtype=torch.float)

def get_advanced_scheduler(optimizer, method='cosine_warmup', total_epochs=50):
    """
    Get advanced learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        method: Scheduler method
        total_epochs: Total number of epochs
    
    Returns:
        scheduler: Learning rate scheduler
    """
    if method == 'cosine_warmup':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    
    elif method == 'onecycle':
        from torch.optim.lr_scheduler import OneCycleLR
        return OneCycleLR(
            optimizer, 
            max_lr=optimizer.param_groups[0]['lr'],
            total_steps=total_epochs,
            pct_start=0.3,
            anneal_strategy='cos'
        )
    
    elif method == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-7,
            verbose=True
        )
    
    else:
        raise ValueError(f"Unknown scheduler method: {method}")

def apply_mixup_cutmix_probability():
    """
    Randomly choose between Mixup and CutMix based on probability.
    """
    return random.choice(['mixup', 'cutmix', 'none'])
