import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import urllib.request
from pathlib import Path


class RemoteSensingDataset(Dataset):
    """
    Remote sensing segmentation dataset with point label simulation.
    
    Args:
        root_dir: path to dataset
        split: 'train', 'val', or 'test'
        sampling_rate: fraction of pixels to label (e.g., 0.01 = 1%)
        sampling_strategy: 'random' or 'stratified'
        transform: optional image transforms
    """
    
    def __init__(self, root_dir='./data', split='train', sampling_rate=0.01, 
                 sampling_strategy='random', transform=None, use_synthetic=False):
        self.root_dir = Path(root_dir)
        self.split = split
        self.sampling_rate = sampling_rate
        self.sampling_strategy = sampling_strategy
        self.transform = transform
        self.use_synthetic = use_synthetic
        
        self.classes = ['background', 'building', 'woodland', 'water', 'road']
        self.num_classes = len(self.classes)
        
        if use_synthetic:
            self._setup_synthetic()
        else:
            self._setup_real()
    
    def _setup_synthetic(self):
        """Generate synthetic data for testing."""
        print("Using synthetic data (for testing)")
        
        if self.split == 'train':
            self.n_samples = 80
        elif self.split == 'val':
            self.n_samples = 10
        else:
            self.n_samples = 10
    
    def _setup_real(self):
        """Setup real remote sensing dataset."""
        
        print(f"Setting up remote sensing dataset: {self.split}")
        
        data_path = self.root_dir / 'landcover'
        if not data_path.exists():
            print("Dataset not found, using synthetic data instead")
            self.use_synthetic = True
            self._setup_synthetic()
            return
        
        self.image_files = sorted((data_path / self.split / 'images').glob('*.png'))
        self.mask_files = sorted((data_path / self.split / 'masks').glob('*.png'))
        
        if len(self.image_files) == 0:
            print("No images found, using synthetic data")
            self.use_synthetic = True
            self._setup_synthetic()
            return
        
        self.n_samples = len(self.image_files)
        print(f"Loaded {self.n_samples} samples")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Returns:
            image: (3, H, W) RGB tensor
            mask_full: (H, W) complete ground truth
            mask_points: (H, W) binary (1=labeled, 0=unlabeled)
            labels_points: (H, W) labels at sampled points
        """
        if self.use_synthetic:
            H, W = 512, 512
            
            image = np.random.rand(3, H, W).astype(np.float32) * 0.5
            
            x = np.linspace(-1, 1, W)
            y = np.linspace(-1, 1, H)
            xx, yy = np.meshgrid(x, y)
            
            buildings = ((np.abs(xx) < 0.3) & (np.abs(yy) < 0.3)).astype(float)
            
            roads = (np.abs(yy) < 0.05).astype(float)
            
            water = ((xx**2 + yy**2) < 0.2).astype(float)
            
            np.random.seed(idx)
            woodland = (np.random.rand(H, W) > 0.7).astype(float)
            
            mask_full = np.zeros((H, W), dtype=np.int64)
            mask_full[woodland > 0] = 2  
            mask_full[roads > 0] = 4     
            mask_full[buildings > 0] = 1  
            mask_full[water > 0] = 3    
            
            for c in range(3):
                image[c] += buildings * 0.3
                image[c] += roads * 0.2
                image[c, :, :] *= (1 + water * 0.5)
            
            image = np.clip(image, 0, 1).astype(np.float32)
        
        else:
            image = np.array(Image.open(self.image_files[idx]).convert('RGB'))
            mask_full = np.array(Image.open(self.mask_files[idx]))
            
            image = image.astype(np.float32) / 255.0
            
            image = image.transpose(2, 0, 1)
            
            H, W = mask_full.shape
        
        mask_points, labels_points = self._sample_points(mask_full)
        
        image = torch.from_numpy(image).float()
        mask_full = torch.from_numpy(mask_full).long()
        mask_points = torch.from_numpy(mask_points).float()
        labels_points = torch.from_numpy(labels_points).long()
        
        return image, mask_full, mask_points, labels_points
    
    def _sample_points(self, mask):
        """Sample point labels from full mask."""
        from partial_ce_loss import sample_point_labels
        
        mask_tensor = torch.from_numpy(mask)
        mask_points, labels_points = sample_point_labels(
            mask_tensor,
            sampling_rate=self.sampling_rate,
            strategy=self.sampling_strategy,
            seed=None  
        )
        
        return mask_points.numpy(), labels_points.numpy()


def get_dataloaders(batch_size=8, sampling_rate=0.01, sampling_strategy='random', 
                    num_workers=4, use_synthetic=True):
    """
    This creates train/val/test dataloaders.
    
    Args:
        batch_size: batch size
        sampling_rate: fraction of pixels to label
        sampling_strategy: 'random' or 'stratified'--depeding on our usage
        num_workers: dataloader workers, i purposely increased so I can get better results a bit faster.
        use_synthetic: use synthetic data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = RemoteSensingDataset(
        split='train',
        sampling_rate=sampling_rate,
        sampling_strategy=sampling_strategy,
        use_synthetic=use_synthetic
    )
    
    val_dataset = RemoteSensingDataset(
        split='val',
        sampling_rate=sampling_rate,
        sampling_strategy=sampling_strategy,
        use_synthetic=use_synthetic
    )
    
    test_dataset = RemoteSensingDataset(
        split='test',
        sampling_rate=1.0,  
        sampling_strategy='random',
        use_synthetic=use_synthetic
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    print("Testing Remote Sensing Dataset")
    print("=" * 70)
    
    # Create dataset
    dataset = RemoteSensingDataset(
        split='train',
        sampling_rate=0.01,
        sampling_strategy='random',
        use_synthetic=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.classes}")
    
    image, mask_full, mask_points, labels_points = dataset[0]
    
    print(f"\nSample 0:")
    print(f"  Image: {image.shape} (C, H, W)")
    print(f"  Full mask: {mask_full.shape}")
    print(f"  Point mask: {mask_points.shape}")
    print(f"  Labeled pixels: {mask_points.sum().item()}/{mask_full.numel()} "
          f"({mask_points.mean()*100:.2f}%)")
    print(f"  Classes in full mask: {mask_full.unique().tolist()}")
    
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=4, use_synthetic=True)
    
    images, masks_full, masks_points, labels_points = next(iter(train_dl))
    print(f"\nBatch:")
    print(f"  Images: {images.shape}")
    print(f"  Full masks: {masks_full.shape}")
    print(f"  Point masks: {masks_points.shape}")
    
    print("\n✓ Dataset working correctly!")
