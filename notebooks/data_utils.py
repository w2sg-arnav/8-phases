# --- data_utils.py ---
import os
import glob
import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import slic
from skimage.color import label2rgb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(image_size=(224, 224), train=True):
    """
    Define image transformations using Albumentations.

    Args:
        image_size (tuple): The desired image size (height, width).
        train (bool): Whether to apply training-specific augmentations.

    Returns:
        A.Compose: An Albumentations Compose object containing the transformations.
    """
    if train:
        transforms_list = [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Affine(scale=(0.8, 1.2), rotate=(-45, 45), shear=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Use tuple for mean and std
            ToTensorV2(),
        ]
    else:
        transforms_list = [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Use tuple for mean and std
            ToTensorV2(),
        ]
    return A.Compose(transforms_list)


def segment_leaf(image):
    """Segments the leaf using SLIC superpixels."""
    # Convert PIL Image to NumPy array
    image_np = np.array(image)

    # Perform SLIC segmentation
    segments = slic(image_np, n_segments=100, compactness=10)  # Adjust n_segments as needed
    segmented_image_np = label2rgb(segments, image_np, kind='avg')

    # Create a leaf mask (example: based on segment labels)
    leaf_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)  # Initialize mask

    # Example: Mark segments in the lower half of the image as "leaf"
    for i in range(segments.max() + 1):
        if np.any((segments == i) & (np.arange(image_np.shape[0])[:, None] > image_np.shape[0] // 2)):
            leaf_mask[segments == i] = 1

    # Debugging prints
    print(f"Shape of segmented_image_np: {segmented_image_np.shape}")
    print(f"Data type of segmented_image_np: {segmented_image_np.dtype}")
    print(f"Shape of leaf_mask: {leaf_mask.shape}")
    print(f"Data type of leaf_mask: {leaf_mask.dtype}")

    # Convert to uint8 and create PIL Images
    segmented_image_np = (segmented_image_np * 255).astype(np.uint8)
    leaf_mask_uint8 = (leaf_mask * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8

    # Check the shape before creating PIL images
    if segmented_image_np.shape[0] > 0 and segmented_image_np.shape[1] > 0:  # Add this check
        segmented_image_img = Image.fromarray(segmented_image_np)
    else:
        print("Warning: segmented_image_np has an invalid shape. Returning a default image.")
        segmented_image_img = Image.new("RGB", (224, 224), color="black")  # Create a blank image

    if leaf_mask_uint8.shape[0] > 0 and leaf_mask_uint8.shape[1] > 0: # Add this check
        leaf_mask_img = Image.fromarray(leaf_mask_uint8)
    else:
        print("Warning: leaf_mask_uint8 has an invalid shape. Returning a default image.")
        leaf_mask_img = Image.new("L", (224, 224), color="black")  # Create a blank grayscale image


    return segmented_image_img, leaf_mask_img



class CottonDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)  # Convert PIL Image to NumPy array
        label = self.labels[idx]

        if self.transform:
            transformed = self.transform(image=image)  # Pass image to transform
            image = transformed['image']

        return image, label


def create_data_loaders(data_dir, train_transform, val_transform, batch_size, num_workers, classes):
    """Creates training, validation, and test data loaders."""
    image_paths = []
    labels = []
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        class_dir = os.path.join(data_dir, cls)
        for image_file in glob.glob(os.path.join(class_dir, '*.jpg')):
            image_paths.append(image_file)
            labels.append(class_to_idx[cls])

    # Split data into training, validation, and test sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.25, random_state=42  # 0.25 x 0.8 = 0.2
    )

    print("--- Loaded File Paths (First 5 of each) ---")
    print("Train:", train_paths[:5])
    print("Validation:", val_paths[:5])
    print("Test:", test_paths[:5])
    print("----------------------------------------")

    train_dataset = CottonDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = CottonDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = CottonDataset(test_paths, test_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader