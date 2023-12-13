import os
import re

import subprocess

import verovio


def musicxml_to_svg(input_file, output_path, name):
	tk = verovio.toolkit()
	tk.loadFile(input_file)

	output_files = []
	os.makedirs(f"{output_path}/{name}", exist_ok=True)

	for i in range(tk.getPageCount()):
		file = f"{output_path}/{name}/sheet_{i}.svg"
		tk.renderToSVGFile(file, (i + 1))
		output_files.append(file)

	return output_files


def convert_sheets(input_dir, prefix):
	for element in os.listdir(input_dir):
		file_path = os.path.join(input_dir, element)

		if os.path.isdir(file_path):
			convert_sheets(file_path, prefix + [element])
		else:
			print("Converting:", prefix, element)
			name = "".join(map(lambda x: x + "_", prefix)) + element.split(".")[0]
			musicxml_to_svg(file_path, "../dataset/existing/render", name)


def convert_svg_to_png(input_file, output_file, width=2480, height=3508):
	result = subprocess.run(
		["inkscape", "-b", "FFFFFF", "-w", str(width), "-h", str(height), input_file, "-o", output_file],
		stdout=subprocess.PIPE, stderr=subprocess.PIPE
	)
	success = (len(result.stderr) < 100 and "Background RRGGBBAA" in str(result.stderr)) or ("Background RRGGBBAA" not in str(result.stderr) and len(result.stderr) == 0)

	return success, result.stderr.decode("utf-8")


def convert_files(path_in="../dataset/existing/render", path_out="../dataset/pairs/perfect", tmp_dir="../tmp"):
	try:
		os.mkdir(tmp_dir)
	except FileExistsError:
		pass

	files = []

	for folder in os.listdir(path_in):
		for file in os.listdir(f"{path_in}/{folder}/"):
			path_file_in = os.path.join(path_in, folder, file)

			name = f"{folder}_{'_'.join(file.split('.')[0].split('_')[1:])}".replace(" ", "_")
			name = re.sub(r'\W+', '', name)
			while "__" in name:
				name = name.replace("__", "_")

			path_file_out = os.path.join(path_out, name+".png")

			if os.path.exists(path_file_out):
				continue

			files.append((path_file_in, path_file_out))

	with open(os.path.join(tmp_dir, "files.txt"), "w") as f:
		f.write("\n".join(map(lambda x: f"{x[0]}{chr(1)}{x[1]}", files)))

	with open(os.path.join(tmp_dir, "files.txt"), "r") as fr:
		lines = fr.read()
		if lines != "":
			files = list(map(lambda x: x.split(chr(1)), lines.split("\n")))
		else:
			files = []

	try:
		with open(os.path.join(tmp_dir, "progress_done.txt"), "r") as fr:
			files_done = set(fr.read().split("\n"))
	except FileNotFoundError:
		files_done = set()

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

	os.remove(os.path.join(tmp_dir, "files.txt"))
	os.remove(os.path.join(tmp_dir, "progress_done.txt"))
	os.rmdir(tmp_dir)