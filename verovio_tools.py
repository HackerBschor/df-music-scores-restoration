import os
import verovio

from pydub import AudioSegment

if __name__ == '__main__':
	tk = verovio.toolkit()
	tk.loadFile("examples/Mozart-Don_Giovanni.mxl")

	for i in range(tk.getPageCount()):
		tk.renderToSVGFile(f"examples/render/Mozart-Don_Giovanni_{i}.svg", (i+1))
