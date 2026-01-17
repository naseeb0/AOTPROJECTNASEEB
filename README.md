# AIoT Project - Dangi Naseeb (2120256035)

## Project Overview
This project reproduces two deep learning models for image processing tasks:

1. **ResNet18** - Image Classification on MNIST dataset
2. **UNet** - Image Segmentation on Liver CT dataset

## Project Structure
```
DANGI_NASEEB_AOT_PROJECT/
├── ResNet18_Classification/     # Classification model
│   ├── main.py                  # Training and testing (OPTIMIZED)
│   ├── requirements.txt         # Dependencies
│   ├── Code_Explanation.md      # Code documentation
│   └── weights/                 # Trained model weights
│
├── UNet_Segmentation/           # Segmentation model
│   ├── main.py                  # Training and testing (OPTIMIZED)
│   ├── unet.py                  # UNet architecture (original)
│   ├── dataset.py               # Liver CT data loading (original)
│   ├── requirements.txt         # Dependencies
│   ├── Code_Explanation.md      # Code documentation
│   ├── data/                    # Dataset (400 train + 40 val)
│   └── weights/                 # Trained model weights
│
├── Experiment_Report.md         # Comprehensive experiment report
├── README.md                    # This file
├── environment.yml              # Conda environment
├── requirements.txt             # Pip requirements
└── train_on_azure.sh           # Azure VM training script
```

## Quick Start

### Local Setup (CPU)
```bash
# Install dependencies
pip install -r requirements.txt

# Train ResNet18 (quick test)
cd ResNet18_Classification
python main.py train --epochs 5 --batch_size 64

# Train UNet (quick test)
cd ../UNet_Segmentation
python main.py train --epochs 5 --batch_size 2
```

### Azure VM Setup (GPU) - Recommended
```bash
# 1. Clone the repo on Azure VM
git clone <your-repo-url>
cd DANGI_NASEEB_AOT_PROJECT

# 2. Run the training script
chmod +x train_on_azure.sh
./train_on_azure.sh
```

## Model Optimizations

### ResNet18 (Classification)
- **CosineAnnealingLR** scheduler for better convergence
- **Enhanced data augmentation** (rotation, translation, crop)
- **Higher initial LR** (0.1) with scheduler
- **Checkpoint saving** for resume capability
- **Expected accuracy**: >99% on MNIST

### UNet (Segmentation)
- **Combined Dice + BCE Loss** (handles class imbalance)
- **Synchronized augmentation** (same transform for image & mask)
- **ReduceLROnPlateau** scheduler
- **Dice coefficient** tracking
- **Expected Dice score**: >0.90 on Liver CT

## Training Commands

```bash
# ResNet18 - Full training (50 epochs recommended)
python main.py train --epochs 50 --batch_size 128

# UNet - Full training (100 epochs recommended)
python main.py train --epochs 100 --batch_size 4

# Resume interrupted training
python main.py train --resume
```

## Testing Commands

```bash
# Test ResNet18
cd ResNet18_Classification
python main.py test --weight weights/best_model.pth

# Test UNet
cd UNet_Segmentation
python main.py test --weight weights/best_model.pth
```

## Author
- **Name:** Dangi Naseeb
- **Student ID:** 2120256035
- **Course:** AIoT (Artificial Intelligence of Things)
- **Deadline:** January 20, 2026
