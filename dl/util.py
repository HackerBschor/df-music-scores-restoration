import matplotlib.pyplot as plt
import torch
import numpy as np


def undo_preprocessing(image_tensor: torch.Tensor) -> np.ndarray:
    image_numpy = image_tensor.numpy()

    # Undo preprocessing
    mean = np.array(0.5)
    std = np.array(0.5)
    image = std * image_numpy + mean

    return image


def show_clean_dirty_images(tensor_img_dirty: torch.Tensor, tensor_img_clean: torch.Tensor) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    img_1 = undo_preprocessing(tensor_img_dirty)
    img_2 = undo_preprocessing(tensor_img_clean)

    ax1.imshow(img_1.squeeze(), cmap='gray')
    ax1.set_title('Dirty Image')
    ax2.imshow(img_2.squeeze(), cmap='gray')
    ax2.set_title('Clean Image')

    plt.show()


def show_clean_dirty_prediction_images(
        tensor_img_dirty: torch.Tensor, tensor_img_clean: torch.Tensor, tensor_img_prediction: torch.Tensor) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))
    img_1 = undo_preprocessing(tensor_img_dirty)
    img_2 = undo_preprocessing(tensor_img_clean)
    img_3 = undo_preprocessing(tensor_img_prediction)

    ax1.imshow(img_1.squeeze(), cmap='gray')
    ax1.set_title('Dirty Image')
    ax2.imshow(img_2.squeeze(), cmap='gray')
    ax2.set_title('Clean Image')
    ax3.imshow(img_3.squeeze(), cmap='gray')
    ax3.set_title('Prediction Image')

    plt.show()
