
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from partial_ce_loss import PartialCrossEntropyLoss
from dataset import get_dataloaders

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()

        self.enc1 = self._block(in_channels, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        self.enc4 = self._block(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self._block(256, 512)

        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._block(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._block(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._block(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._block(64, 32)

        self.out = nn.Conv2d(32, num_classes, 1)
    
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)


def compute_miou(preds, targets, num_classes=5):
    """Compute mean IoU."""
    ious = []
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return np.mean(ious), ious


def train_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch WITHOUT mixed precision."""
    model.train()
    running_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(loader, desc='Train', leave=False)
    for images, masks_full, masks_points, labels_points in pbar:
        images = images.to(device, non_blocking=True)
        labels_points = labels_points.to(device, non_blocking=True)
        masks_points = masks_points.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels_points, masks_points)

        if torch.isnan(loss) or torch.isinf(loss):
          print("WARNING: Invalid loss detected, skipping batch")
          continue

        loss_value = loss.item()   

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss_value
        n_batches += 1
        pbar.set_postfix(loss=f'{loss.item():.4f}')
    
    if n_batches == 0:
        return 0.0
    
    return running_loss / n_batches


@torch.no_grad()
def validate_epoch(model, loader, device, num_classes=5):
    """Validate one epoch."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for images, masks_full, _, _ in tqdm(loader, desc='Val', leave=False):
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        
        all_preds.append(preds.cpu())
        all_targets.append(masks_full.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    miou, class_ious = compute_miou(all_preds, all_targets, num_classes)
    
    return miou, class_ious


def train(args):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Mixed precision: DISABLED (for stability)")
    
    # Data
    train_dl, val_dl, test_dl = get_dataloaders(
    batch_size=args.batch_size,
    sampling_rate=args.sampling_rate,
    sampling_strategy=args.strategy,
    num_workers=2,
    use_synthetic=True
)
    
    model = UNet(in_channels=3, num_classes=5).to(device)
    
    criterion = PartialCrossEntropyLoss(focal_gamma=args.focal_gamma)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    best_miou = 0.0
    history = {'train_loss': [], 'val_miou': [], 'val_class_iou': []}
    
    print(f"\nTraining: {args.epochs} epochs")
    print(f"Sampling: {args.sampling_rate*100:.1f}% ({args.strategy})")
    print(f"Learning rate: {args.lr}")
    print("-" * 60)
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_dl, criterion, optimizer, device)
        val_miou, val_class_ious = validate_epoch(model, val_dl, device)
        
        scheduler.step(val_miou)
        
        history['train_loss'].append(train_loss)
        history['val_miou'].append(val_miou)
        history['val_class_iou'].append(val_class_ious)
        
        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"loss {train_loss:.4f}  "
              f"mIoU {val_miou:.4f}", end='')
        
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'miou': val_miou,
                'class_ious': val_class_ious,
                'args': vars(args)
            }, f'best_{args.exp_name}.pt')
            print("  ← best", end='')
        print()
    
    print("\nFinal test evaluation:")
    test_miou, test_class_ious = validate_epoch(model, test_dl, device)
    print(f"Test mIoU: {test_miou:.4f}")
    print(f"\nPer-class IoU:")
    class_names = ['background', 'building', 'woodland', 'water', 'road']
    for name, iou in zip(class_names, test_class_ious):
        print(f"  {name}: {iou:.4f}")
    
    results = {
        'sampling_rate': args.sampling_rate,
        'strategy': args.strategy,
        'best_val_miou': float(best_miou),
        'test_miou': float(test_miou),
        'test_class_ious': [float(x) for x in test_class_ious],
        'final_train_loss': float(train_loss),
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_miou': [float(x) for x in history['val_miou']]
        }
    }
    
    with open(f'results_{args.exp_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"  Best validation mIoU: {best_miou:.4f}")
    print(f"  Test mIoU: {test_miou:.4f}")
    print(f"  Results saved to: results_{args.exp_name}.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=5e-4, type=float, help='Lower LR for stability')
    parser.add_argument('--sampling_rate', default=0.01, type=float)
    parser.add_argument('--strategy', default='random', choices=['random', 'stratified'])
    parser.add_argument('--focal_gamma', default=0.0, type=float)
    parser.add_argument('--exp_name', default='exp', type=str)
    args = parser.parse_args()
    
    train(args)
