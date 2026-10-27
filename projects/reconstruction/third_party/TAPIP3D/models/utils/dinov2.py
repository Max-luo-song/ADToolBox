import torch
from torchvision import transforms

def load_dinov2(model_name: str):
    raise NotImplementedError()
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return model, normalize