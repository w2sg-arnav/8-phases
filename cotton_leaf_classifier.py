# Imports (common to many phases - consolidate at the top of your file)
import os
import sys  # For exiting on errors
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Install required packages (if not already installed)
try:
    from skimage.segmentation import slic
    from skimage.color import label2rgb
except ImportError:
    print("Installing scikit-image...")
    try:
        # Try conda first (preferred if using conda)
        import conda.cli
        conda.cli.main('conda', 'install', '-y', '-c', 'conda-forge', 'scikit-image')
    except ImportError:
        # Fallback to pip
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image"])
    from skimage.segmentation import slic  # Try importing again
    from skimage.color import label2rgb


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

try:
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights, vit_b_16, ViT_B_16_Weights
except ImportError:
    print("Installing torchvision (likely needed for newer model weights)...")
    try:
        import conda.cli
        conda.cli.main('conda', 'install', '-y', '-c', 'pytorch', 'torchvision')
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision"])
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights, vit_b_16, ViT_B_16_Weights


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# For experiment tracking (choose one - example with Weights & Biases)
try:
    import wandb
except ImportError:
    print("Installing wandb...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb

# For hyperparameter optimization (example with Optuna)
try:
    import optuna
except ImportError:
    print("Installing optuna...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna

# For image augmentations (albumentations is very powerful)
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("Installing albumentations...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "albumentations"])
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

# For Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("Installing grad-cam...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "grad-cam"])
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image