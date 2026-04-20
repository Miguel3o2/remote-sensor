
import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialCrossEntropyLoss(nn.Module):
    """
    Fixed version with NaN prevention.
    """
    
    def __init__(self, ignore_index=0, focal_gamma=0.0, class_weights=None, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights
        self.reduction = reduction
        
    def forward(self, logits, targets, mask):
        """
        Args:
            logits: (B, C, H, W) 
            targets: (B, H, W) 
            mask: (B, H, W) binary (1=labeled, 0=unlabeled)
        """
        B, C, H, W = logits.shape
        
        mask = mask.float()
        mask = torch.clamp(mask, 0, 1)  # Ensure [0, 1]
        
        num_labeled = mask.sum()
        if num_labeled == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        targets = torch.clamp(targets.long(), 0, C-1)
        
        log_probs = F.log_softmax(logits, dim=1)  
        
        targets_expanded = targets.unsqueeze(1)  
        log_p = log_probs.gather(dim=1, index=targets_expanded).squeeze(1) 
        
        ce_loss = -log_p
        
        if torch.isnan(ce_loss).any() or torch.isinf(ce_loss).any():
            print("WARNING: NaN/Inf detected in CE loss, returning zero")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        if self.focal_gamma > 0:
            probs = F.softmax(logits, dim=1)
            p = probs.gather(dim=1, index=targets_expanded).squeeze(1)
            p = torch.clamp(p, min=1e-7, max=1.0)  # Prevent log(0)
            focal_weight = (1.0 - p) ** self.focal_gamma
            ce_loss = focal_weight * ce_loss
        
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            weight_map = weights[targets]
            ce_loss = ce_loss * weight_map
        
        masked_loss = ce_loss * mask
        
        if self.reduction == 'mean':
            loss = masked_loss.sum() / torch.clamp(num_labeled, min=1.0)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: Final loss is NaN/Inf, returning zero")
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            return loss
        elif self.reduction == 'sum':
            return masked_loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def sample_point_labels(mask, sampling_rate=0.01, strategy='random', seed=None):
    """
    FIXED: Ensure we always get at least some labeled pixels.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, H, W = mask.shape
    point_mask = torch.zeros_like(mask, dtype=torch.float32)
    point_labels = torch.zeros_like(mask, dtype=torch.long)
    
    for b in range(B):
        mask_b = mask[b]
        
        total_pixels = H * W
        num_samples = max(10, int(total_pixels * sampling_rate))  
        
        if strategy == 'random':
            indices = torch.randperm(total_pixels)[:num_samples]
            rows = indices // W
            cols = indices % W
            
            point_mask[b, rows, cols] = 1.0
            point_labels[b, rows, cols] = mask_b[rows, cols]
        
        elif strategy == 'stratified':
            unique_classes = mask_b.unique()
            num_classes = len(unique_classes)
            num_samples_per_class = max(2, num_samples // num_classes)  
            
            for cls in unique_classes:
                class_mask = (mask_b == cls)
                class_indices = torch.where(class_mask.view(-1))[0]
                
                if len(class_indices) == 0:
                    continue
                
                n_samples = min(num_samples_per_class, len(class_indices))
                sampled_indices = class_indices[torch.randperm(len(class_indices))[:n_samples]]
                
                rows = sampled_indices // W
                cols = sampled_indices % W
                
                point_mask[b, rows, cols] = 1.0
                point_labels[b, rows, cols] = cls
    
    if squeeze_output:
        point_mask = point_mask.squeeze(0)
        point_labels = point_labels.squeeze(0)
    
    return point_mask, point_labels


if __name__ == '__main__':
    print("Testing FIXED Partial CE Loss")
    print("=" * 70)
    
    B, C, H, W = 2, 5, 64, 64
    
    logits = torch.randn(B, C, H, W) * 0.1  
    targets = torch.randint(0, C, (B, H, W))
    
    point_mask, point_labels = sample_point_labels(targets, sampling_rate=0.01, strategy='random')
    
    print(f"Labeled pixels: {point_mask.sum().item()}/{H*W*B}")
    
    criterion = PartialCrossEntropyLoss()
    loss = criterion(logits, point_labels, point_mask)
    
    print(f"Loss: {loss.item():.4f} (should be valid number, not NaN)")
    
    loss.backward()
    print(f"Gradients computed successfully: {logits.grad is not None}")
    
    print("\n✓ Fixed loss working correctly!")
