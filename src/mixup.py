import torch
import torch.nn as nn
import numpy as np

def mixup_data(x, y, alpha=1.0, device='cuda'):
    """
    Mixup data augmentation.
    
    Args:
        x: Input batch
        y: Target batch
        alpha: Mixup parameter (higher = more mixing)
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

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss calculation.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a, y_b: Original targets
        lam: Mixing ratio
    
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class MixupTrainer:
    """
    Mixup training wrapper.
    """
    def __init__(self, model, optimizer, criterion, device, alpha=0.2):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.alpha = alpha
    
    def train_step(self, dataloader):
        """
        Single training step with mixup.
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply mixup
            data, target_a, target_b, lam = mixup_data(data, target, self.alpha, self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            # For accuracy calculation, use original targets
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (lam * predicted.eq(target_a.data).cpu().sum().float() + 
                       (1 - lam) * predicted.eq(target_b.data).cpu().sum().float())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy.item()
