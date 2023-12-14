from typing import Tuple

import torch
from torchvision import transforms
import numpy as np
from PIL import Image


def generate_preprocess_transformer(size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])


def preprocess(path: str, filename: str, transformer: transforms.Compose) -> torch.Tensor:
    image = Image.open(f"{path}/{filename}").convert('L')
    return transformer(image)


def undo_preprocessing(tensor: torch.Tensor) -> Image.Image:
    tensor_np = ((0.5 * tensor.detach().numpy() + 0.5) * 255).astype(np.uint8).squeeze()
    return Image.fromarray(tensor_np).convert('RGB')