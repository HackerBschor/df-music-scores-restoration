import verovio
from cairosvg import svg2png, svg2pdf


def save_page_as_pdf(tk, page_num):
	svg = tk.renderToSVG(page_num)
	svg2pdf(bytestring=svg, write_to=f"examples/render/Mozart-Don_Giovanni_{page_num}.pdf")


def save_all_pages(tk):
	for i in range(tk.getPageCount()):
		save_page_as_pdf(tk, (i+1))


if __name__ == '__main__':
	verovio_tk = verovio.toolkit()
	verovio_tk.loadFile("examples/Mozart-Don_Giovanni.mxl")
	save_page_as_pdf(verovio_tk, 66)



