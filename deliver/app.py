import math

import PIL
import numpy as np
from flask import Flask, send_file
from flask import render_template, request
from werkzeug.datastructures import FileStorage
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms

import zipfile

from model import ConvAutoencoderDenoiser

transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

model = ConvAutoencoderDenoiser()
model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))

app = Flask(__name__)


def preprocess_file(path: str, filename: str, sub_image_shape=(310, 440)):
    image = Image.open(f"{path}/{filename}").convert('L')
    return preprocess_file(image, sub_image_shape)


def preprocess_image(image: PIL.Image, sub_image_shape=(310, 440)):
    image = PIL.ImageOps.invert(image)
    original_width, original_height = (2480, 3508)
    image = image.resize((original_width, original_height))

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


@app.route('/')
def index():  # put application's code here
    return render_template("index.html")


def apply_model(file: FileStorage) -> tuple[str, BytesIO]:
    image = Image.open(file.stream).convert("L")

    tensors, shape = preprocess_image(image, (416, 308))

    tensors_flat = flatten_tensor_list(tensors)

    with torch.no_grad():
        prediction = model(tensors_flat)

    prediction_reshape = []
    for i in range(len(tensors)):
        row = []
        for j in range(len(tensors[0])):
            row.append(prediction[i * len(tensors[0]) + j])
        prediction_reshape.append(row)

    prediction_image = restore_image(prediction_reshape, shape)

    image_bytes = BytesIO()
    prediction_image.save(image_bytes, format='png')
    image_bytes.seek(0)

    return file.filename, image_bytes


@app.route('/upload', methods=['POST'])
def upload():
    applied_files = []

    for file in request.files.getlist("files"):
        print(file.filename)
        applied_files.append(apply_model(file))

    if len(applied_files) == 1:
        filename, image_bytes = applied_files[0]
        return send_file(image_bytes, mimetype='image/png', as_attachment=True, download_name=filename)

    else:
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for filename, image_bytes in applied_files:
                zip_file.writestr(filename, image_bytes.getvalue())

        zip_buffer.seek(0)

        return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name="images.zip")


if __name__ == '__main__':
    app.run()
