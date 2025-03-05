# models.py
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Make sure DEVICE is defined here, or pass it as an argument
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_vit_model(model_name="vit_b_16", pretrained=True, num_classes=7):  # Corrected num_classes
    if model_name == "vit_b_16":
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=weights)
        model.heads[-1] = nn.Linear(model.heads[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported ViT model name: {model_name}")
    return model.to(DEVICE)

# You can add other model definitions here (e.g., get_effnet_model)