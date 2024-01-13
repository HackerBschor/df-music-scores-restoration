import math
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import PIL.ImageOps


transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])


def preprocess_file(path: str, filename: str, sub_image_shape=(310, 440)) -> tuple[list[list[PIL.Image]], tuple[int, int]]:
    image = Image.open(f"{path}/{filename}").convert('L')
    return preprocess_image(image, sub_image_shape)


def preprocess_image(image: PIL.Image, sub_image_shape=(310, 440)):
    image = PIL.ImageOps.invert(image)
    original_width, original_height = image.size

    # Calculate the number of sub-images in both dimensions
    num_horizontal = math.ceil(original_width / sub_image_shape[0])
    num_vertical = math.ceil(original_height / sub_image_shape[1])

    sub_images = []

    for i in range(num_vertical):
        sub_images_row = []

        for j in range(num_horizontal):
            # Calculate the coordinates of the top-left corner of the sub-image
            left = j * sub_image_shape[0]
            upper = i * sub_image_shape[1]

            # Calculate the coordinates of the bottom-right corner of the sub-image
            right = left + sub_image_shape[0]
            lower = upper + sub_image_shape[1]

            # Crop the sub-image from the original image
            sub_image = image.crop((left, upper, right, lower))
            sub_image = PIL.ImageOps.invert(sub_image)

            sub_images_row.append(transformer(sub_image))

        sub_images.append(sub_images_row)

    return sub_images, image.size


def flatten_tensor_list(tensors: list[list[torch.Tensor]]) -> torch.Tensor:
    ts = []
    for row in tensors:
        for t in row:
            ts.append(t)

    return torch.stack(ts)


def undo_preprocessing(tensor: torch.Tensor) -> Image.Image:
    tensor_np = ((0.5 * tensor.detach().cpu().numpy() + 0.5) * 255).astype(np.uint8).squeeze()
    return Image.fromarray(tensor_np).convert('RGB')


def restore_image(sub_images: list[list[torch.Tensor]], result_size: tuple[int, int]) -> Image:
    concatenated_image = Image.new('RGB', result_size)
    image_height, image_width = sub_images[0][0].shape[1:3]

    for i, image_row in enumerate(sub_images):
        for j, image in enumerate(image_row):
            concatenated_image.paste(undo_preprocessing(image), (j * image_width, i * image_height))

    return concatenated_image
