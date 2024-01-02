from flask import Flask, send_file
from flask import render_template, request
from PIL import Image
from io import BytesIO
import zipfile

from werkzeug.datastructures import FileStorage

app = Flask(__name__)


@app.route('/')
def index():  # put application's code here
	return render_template("index.html")


def apply_model(file: FileStorage) -> tuple[str, BytesIO]:
	image = Image.open(file.stream).convert("L")

	image_bytes = BytesIO()
	image.save(image_bytes, format='png')
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
