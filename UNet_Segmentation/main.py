"""
UNet for Liver CT Segmentation
============================================================
Original Source: https://github.com/yudijiao/UNet
Modified for AIoT Project with optimizations for higher accuracy

OPTIMIZATIONS ADDED:
- Dice Loss + BCE Loss combined (handles class imbalance better)
- Data augmentation (horizontal flip, rotation)
- Learning rate scheduler
- Checkpoint saving and resume capability
- Dice coefficient metric tracking
- Progress bar with tqdm
"""

import torch
from torchvision.transforms import transforms as T
import torchvision.transforms.functional as TF
import argparse
import unet
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import LiverDataset
from torch.utils.data import DataLoader
import os
import random
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# Custom Transforms with synchronized augmentation for image and mask
# =============================================================================
class DualTransform:
    """Apply same random transforms to both image and mask"""
    def __init__(self, train=True):
        self.train = train
    
    def __call__(self, image, mask):
        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        if self.train:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # Random rotation (0, 90, 180, 270 degrees)
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
        
        # Normalize image only (not mask)
        image = TF.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        return image, mask


class AugmentedLiverDataset(LiverDataset):
    """Extended dataset with dual transform for augmentation"""
    def __init__(self, root, train=True):
        super().__init__(root, transform=None, target_transform=None)
        self.dual_transform = DualTransform(train=train)
    
    def __getitem__(self, index):
        import PIL.Image as Image
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        img_x, img_y = self.dual_transform(img_x, img_y)
        return img_x, img_y


# =============================================================================
# Dice Loss for better segmentation (handles class imbalance)
# =============================================================================
class DiceLoss(torch.nn.Module):
    """Dice Loss for binary segmentation"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(torch.nn.Module):
    """Combined BCE + Dice Loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = torch.nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)


def dice_coefficient(pred, target, threshold=0.5):
    """Calculate Dice coefficient metric"""
    pred = (pred > threshold).float()
    smooth = 1.0
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


# =============================================================================
# Training Function (OPTIMIZED)
# =============================================================================
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=50, save_dir='weights'):
    """
    Optimized training function with:
    - Validation after each epoch
    - Dice coefficient tracking
    - Best model saving
    - Checkpoint for resume
    """
    os.makedirs(save_dir, exist_ok=True)
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}/{num_epochs - 1}')
        print('-' * 40)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        pbar = tqdm(train_loader, desc='Training')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice_coefficient(outputs, y).item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                
                val_loss += loss.item()
                val_dice += dice_coefficient(outputs, y).item()
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}')
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'*** New best model saved! Dice: {best_dice:.4f} ***')
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_dice': best_dice
        }, os.path.join(save_dir, 'checkpoint.pth'))
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'weights_{epoch}.pth'))
    
    print(f'\n{"="*60}')
    print(f'Training finished! Best Dice: {best_dice:.4f}')
    print(f'{"="*60}')
    
    return model


def train(args):
    """Main training function"""
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Model
    model = unet.UNet(3, 1).to(device)
    
    # OPTIMIZED: Combined loss function
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    
    # OPTIMIZED: Adam with weight decay
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # OPTIMIZED: Learning rate scheduler (verbose removed for PyTorch compatibility)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # OPTIMIZED: Augmented datasets
    train_dataset = AugmentedLiverDataset("data/train", train=True)
    val_dataset = AugmentedLiverDataset("data/val", train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Resume training if checkpoint exists
    start_epoch = 0
    if args.resume and os.path.exists('weights/checkpoint.pth'):
        print("Loading checkpoint...")
        checkpoint = torch.load('weights/checkpoint.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Train
    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, 
                num_epochs=args.epochs, save_dir='weights')


def test(args):
    """Test function with visualization"""
    model = unet.UNet(3, 1)
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))
    model.eval()
    
    val_dataset = AugmentedLiverDataset("data/val", train=False)
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    total_dice = 0.0
    
    import matplotlib.pyplot as plt
    plt.ion()
    
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            pred = model(x)
            dice = dice_coefficient(pred, y)
            total_dice += dice.item()
            
            # Visualize first 10 samples
            if i < 10:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(x.squeeze().permute(1, 2, 0).numpy() * 0.5 + 0.5)
                axes[0].set_title('Input')
                axes[1].imshow(y.squeeze().numpy(), cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[2].imshow(pred.squeeze().numpy(), cmap='gray')
                axes[2].set_title(f'Prediction (Dice: {dice:.4f})')
                plt.tight_layout()
                plt.pause(1)
                plt.close()
    
    avg_dice = total_dice / len(val_loader)
    print(f'\nAverage Dice Coefficient: {avg_dice:.4f}')


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UNet for Liver CT Segmentation')
    parser.add_argument('action', type=str, help='train or test')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)  # OPTIMIZED: More epochs
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight', type=str, default='weights/best_model.pth',
                        help='the path of the model weight file')
    parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
    args = parser.parse_args()
    
    print("=" * 60)
    print("UNet Liver CT Segmentation (OPTIMIZED)")
    print("Original: https://github.com/yudijiao/UNet")
    print("=" * 60)
    
    if args.action == 'train':
        train(args)
    elif args.action == 'test':
        test(args)
    else:
        print("Invalid action. Use 'train' or 'test'")
