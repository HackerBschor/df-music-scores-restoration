import verovio
from scamp import NoteProperties, StartSlur, Session, StopSlur
def musicxml_to_svg(input_path, output_path, name):
	tk = verovio.toolkit()
	tk.loadFile(f"{input_path}/{name}.mxl")

	tk.renderToSVGFile(f"{output_path}/{name}.svg")

	for i in range(tk.getPageCount()):
		tk.renderToSVGFile(f"{output_path}/{name}/sheet_{i}.svg", (i+1))


def play(inst):
	harmonic = NoteProperties("notehead: harmonic", "pitch + 12")

	for i, pitch in enumerate(range(70, 79)):
		if i % 3 == 0:
			inst.play_note(pitch, 1, 1 / 3, [harmonic, "staccato", StartSlur()])
		elif i % 3 == 2:
			inst.play_note(pitch, 1, 1 / 3, [harmonic, "staccato", StopSlur()])
		else:
			inst.play_note(pitch, 1, 1 / 3, [harmonic, "staccato"])

	inst.play_chord([67, 79], 1.0, 1, "accent, fermata")


def play_music_to_xml():
	s = Session()

	s.fast_forward_to_beat(100)

	piano = s.new_part("piano")

	s.start_transcribing()
	s.fork(lambda _: play(inst=piano))
	s.wait_for_children_to_finish()
	performance = s.stop_transcribing()

	performance.to_score().to_music_xml().export_to_file("test.xml")
	performance.to_score().show()
	musicxml_to_svg(".", ".", "test")


if __name__ == '__main__':
	play_music_to_xml()
	# musicxml_to_svg(".", ".", "test")