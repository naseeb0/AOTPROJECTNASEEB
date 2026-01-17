#!/bin/bash
# =============================================================================
# Azure VM Training Script for AIoT Project
# Run this script on Azure VM with GPU to train both models overnight
# =============================================================================

echo "=============================================="
echo "AIoT Project - Azure Training Script"
echo "=============================================="

# Check GPU
echo "Checking GPU..."
nvidia-smi || echo "WARNING: No GPU detected!"

# Navigate to project directory
cd ~/DANGI_NASEEB_AOT_PROJECT || { echo "Project directory not found!"; exit 1; }

# Install dependencies if not already installed
echo "Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib pillow tqdm tensorboard

# =============================================================================
# Train ResNet18 on MNIST (Classification)
# Expected time: ~30-60 minutes for 50 epochs with GPU
# Expected accuracy: >99%
# =============================================================================
echo ""
echo "=============================================="
echo "Training ResNet18 on MNIST..."
echo "=============================================="

cd ResNet18_Classification

# Train for 50 epochs (optimal for MNIST)
python main.py train --epochs 50 --batch_size 256

# Test the model
python main.py test --weight weights/best_model.pth

cd ..

# =============================================================================
# Train UNet on Liver CT (Segmentation)
# Expected time: ~2-4 hours for 100 epochs with GPU
# Expected Dice score: >0.90
# =============================================================================
echo ""
echo "=============================================="
echo "Training UNet on Liver CT..."
echo "=============================================="

cd UNet_Segmentation

# Train for 100 epochs (segmentation needs more epochs)
python main.py train --epochs 100 --batch_size 8

# Test the model
python main.py test --weight weights/best_model.pth

cd ..

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo ""
echo "Results saved in:"
echo "  - ResNet18_Classification/weights/best_model.pth"
echo "  - UNet_Segmentation/weights/best_model.pth"
echo ""
echo "Next steps:"
echo "  1. git add ."
echo "  2. git commit -m 'Add trained weights'"
echo "  3. git push origin main"
echo "=============================================="
