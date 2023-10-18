import verovio
from cairosvg import svg2png, svg2pdf


if __name__ == '__main__':
	tk = verovio.toolkit()
	tk.loadFile("examples/Mozart-Don_Giovanni.mxl")

	for i in range(tk.getPageCount()):
		tk.renderToSVGFile(f"examples/render/Mozart-Don_Giovanni/sheet_{i}.svg", (i+1))



