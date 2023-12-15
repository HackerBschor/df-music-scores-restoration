import os
import re

import subprocess
from typing import List, Tuple

from verovio import toolkit


def convert_mxl_to_svgs(input_file: str, output_path: str, name: str) -> List[str]:
	"""
	Convert a musicXML file into a list of SVG files and save them in the output_path.
	Each file contains one sheet of the converted musicXML file.
	"""
	tk: toolkit = toolkit()
	tk.loadFile(input_file)

	output_files: List[str] = []
	os.makedirs(f"{output_path}/{name}", exist_ok=True)

	for i in range(tk.getPageCount()):
		file: str = f"{output_path}/{name}/sheet_{i}.svg"
		tk.renderToSVGFile(file, (i + 1))
		output_files.append(file)

	return output_files


def convert_mxls_to_svg(input_path: str, output_path: str, prefix: List[str] = ()) -> None:
	"""
	Convert all musicXML sheets in a folder (containing sub-folders) into SVG files.
	Add the folders as prefixes to output files. Example:
	input_dir
	├───song.mxl
	└───a
		└───b
			└───c.mxl
	-> song_1.svg, song_2.svg, ..., song_n.svg, a_b_c_N.svg, a_b_c_1.svg, ..., a_b_c_M.svg
	"""
	# Walk through every element in the input folder
	for element in os.listdir(input_path):
		file_path: str = os.path.join(input_path, element)

		# If the element is a folder, add the folder name to the prefix and recursively execute the function
		if os.path.isdir(file_path):
			convert_mxls_to_svg(file_path, output_path, list(prefix) + [element])
		else:
			name: str = "".join(map(lambda x: x + "_", prefix)) + element.split(".")[0]
			convert_mxl_to_svgs(file_path, output_path, name)


def convert_svg_to_png(input_file: str, output_file: str, width: int = 2480, height: int = 3508) -> Tuple[bool, str]:
	args: List[str] = ["inkscape", "-b", "FFFFFF", "-w", str(width), "-h", str(height), input_file, "-o", output_file]

	result: subprocess.CompletedProcess = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	success: bool = len(result.stderr) < 100 and "Background RRGGBBAA" in str(result.stderr)
	success = success or ("Background RRGGBBAA" not in str(result.stderr) and len(result.stderr) == 0)

	return success, result.stderr.decode("utf-8")


def convert_svgs_to_png(path_in: str, path_out: str, tmp_dir: str = "tmp"):
	"""
	Convert all svg files in the subfolders of path_in into PNG files. Example:
	input_dir
	├─── a
	│	├─── sheet_1.svg
	│	└─── sheet_2.svg
	└─── b
		├─── sheet_1.svg
		└─── sheet_2.svg
	-> a_sheet_1.png, a_sheet_2.png, b_sheet_1.png, b_sheet_2.png
	"""
	os.makedirs(tmp_dir, exist_ok=True)

	files: List[Tuple[str, str]] = []

	# record all svg files in the subfolders of a path_in into
	for folder in os.listdir(path_in):
		for file in os.listdir(f"{path_in}/{folder}/"):

			path_file_in = os.path.join(path_in, folder, file)

			# sanitizes filename
			name = f"{folder}_{'_'.join(file.split('.')[0].split('_')[1:])}".replace(" ", "_")
			name = re.sub(r'\W+', '', name)
			while "__" in name:
				name = name.replace("__", "_")

			path_file_out = os.path.join(path_out, name + ".png")

			if os.path.exists(path_file_out):
				continue

			files.append((path_file_in, path_file_out))

	# To save the progress, read the already converted files from the `progress_done.txt` file
	try:
		with open(os.path.join(tmp_dir, "progress_done.txt"), "r") as fr:
			files_done = set(fr.read().split("\n"))
	except FileNotFoundError:
		files_done = set()

	# Convert files (if not already done) and append them in the `progress_done.txt` file
	with open(os.path.join(tmp_dir, "progress_done.txt"), "a") as fw:
		for i, (file_in, file_out) in enumerate(files):
			print(f"Converting {i + 1} / {len(files)} ({float((i + 1) * 100) / len(files):.2f}%)", end="")

			if file_out in files_done:
				print(" - Already done")
				continue

			success, error = convert_svg_to_png(file_in, file_out)

			if success:
				fw.write(file_out + "\n")
				files_done.add(file_out)
				print(" - Done")
			else:
				print(f"- Error: ({error})")

	# Clean TMP files
	os.remove(os.path.join(tmp_dir, "progress_done.txt"))
	os.rmdir(tmp_dir)
