import os
import time

import verovio
import cairosvg
from pypdf import PdfMerger


def files_to_pdf(files, output_pdf="output.pdf"):
	merger = PdfMerger()
	temp_dir = f"tmp_{time.time_ns()}"
	os.makedirs(temp_dir)

	pdf_files = []
	for i, file in enumerate(files):
		pdf_file = f"{temp_dir}/sheet_{i}.pdf"
		pdf_files.append(pdf_file)
		cairosvg.svg2pdf(url=file, write_to=pdf_file)
		merger.append(pdf_file)

	merger.write(output_pdf)
	merger.close()

	for file in pdf_files:
		os.remove(file)

	os.rmdir(temp_dir)


def musicxml_to_svg(input_file, output_path, name, first_page_only=False):
	tk = verovio.toolkit()
	tk.loadFile(input_file)

	output_files = []
	os.makedirs(f"{output_path}/{name}", exist_ok=True)

	for i in range(1 if first_page_only else tk.getPageCount()):
		file = f"{output_path}/{name}/sheet_{i}.svg"
		tk.renderToSVGFile(file, (i+1))
		output_files.append(file)

	return output_files


def convert_sheets(input_dir, prefix):
	for element in os.listdir(input_dir):
		file_path = os.path.join(input_dir, element)

		if os.path.isdir(file_path):
			convert_sheets(file_path, prefix + [element])
		else:
			print("Converting:", prefix, element)
			name = "".join(map(lambda x: x+"_", prefix)) + element.split(".")[0]
			musicxml_to_svg(file_path, "../dataset/existing/render", name)


def convert_svg_to_png(input_file, output_file, width=4916, height=7016):
	os.system(f"inkscape -w {width} -h {height} '{input_file}' -o '{output_file}'")


if __name__ == '__main__':
	import re

	files = []

	path_in = "../dataset/existing/render"
	path_out = "../dataset/pairs/perfect"

	for folder in os.listdir(path_in):
		for file in os.listdir(f"{path_in}/{folder}/"):
			path_file_in = os.path.join(path_in, folder, file)

			name = f"{folder}_{'_'.join(file.split('.')[0].split('_')[1:])}".replace(" ", "_")
			name = re.sub(r'\W+', '', name)
			while "__" in name:
				name = name.replace("__", "_")

			path_file_out = os.path.join(path_out, name+".png")

			files.append((path_file_in, path_file_out))

	for i, (file_in, file_out) in enumerate(files):
		print(f"Converting {i+1} / {len(files)} ({float((i+1)*100)/len(files):.2f}%)")
		convert_svg_to_png(file_in, file_out)