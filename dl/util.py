import matplotlib.pyplot as plt
import torch
import numpy as np


def undo_preprocessing(image_tensor: torch.Tensor) -> np.ndarray:
    image_numpy = image_tensor.numpy()
    image_tensor_transpose = image_numpy.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    image = std * image_tensor_transpose + mean

    return np.clip(image, 0, 1)


def show_clean_dirty_images(tensor_img_clean: torch.Tensor, tensor_img_dirty: torch.Tensor) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    img_1 = undo_preprocessing(tensor_img_clean)
    img_2 = undo_preprocessing(tensor_img_dirty)

    ax1.imshow(img_1)
    ax1.set_title('Clean Image')
    ax2.imshow(img_2)
    ax2.set_title('Dirty Image')

    plt.show()