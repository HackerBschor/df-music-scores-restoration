import os

import verovio
from scamp import NoteProperties, StartSlur, Session, StopSlur
import random

characters = [chr(97+i) for i in range(26)] + [" " for _ in range(3)] + [str(i) for i in range(10)]


def musicxml_to_svg(input_path, output_path, name):
	tk = verovio.toolkit()
	tk.loadFile(f"{input_path}/{name}.xml")

	os.makedirs(f"{output_path}/{name}", exist_ok=True)

	for i in range(tk.getPageCount()):
		tk.renderToSVGFile(f"{output_path}/{name}/sheet_{i}.svg", (i+1))


def generate_note_properties():
	properties = []

	harmonic = NoteProperties("notehead: harmonic", "pitch + 12")

	if random.randint(0, 1) == 0:
		properties.append(harmonic)
	if random.randint(0, 1) == 0:
		properties.append("staccato")
	if random.randint(0, 1) == 0:
		if random.randint(0, 1) == 0:
			properties.append(StartSlur())
		else:
			properties.append(StopSlur())

	return properties


def play(inst):
	for i in range(random.randint(0, 1000)):
		tone = random.randint(0, 1)

		if tone == 0:
			properties = generate_note_properties()
			inst.play_note(random.randint(60, 100), random.random(), random.random()*16, )
		elif tone == 1:
			notes = [random.randint(60, 100) for _ in range(0, random.randint(0, 3))]
			volume = random.random()
			length = random.random()*16
			properties = generate_note_properties()
			inst.play_chord(notes, volume, length, properties, "fermata")


def gen_rand_str(max_length=30):
	string = "".join([characters[random.randint(0, len(characters)-1)] for _ in range(random.randint(0, max_length))])
	return " ".join(map(lambda x: x if random.random() < 0.5 else x.capitalize(), string.split(" "))).strip()


def play_music_to_xml(path, name):
	s = Session()
	s.fast_forward_to_beat(100000)
	num_instruments = random.randint(1, 4)
	instruments = [s.new_part(gen_rand_str(15)) for _ in range(num_instruments)]

	s.start_transcribing()

	for i in range(num_instruments):
		s.fork(lambda _: play(inst=instruments[i]))

	s.wait_for_children_to_finish()
	performance = s.stop_transcribing()
	performance.to_score(title=gen_rand_str(), composer=gen_rand_str()).to_music_xml().export_to_file(f"{path}/{name}.xml")
	musicxml_to_svg(path, f"{path}/render", name)


if __name__ == '__main__':
	play_music_to_xml("examples", "random_sheets")