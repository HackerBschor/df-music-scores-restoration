from scamp import NoteProperties, StartSlur, Session, StopSlur
import random

from dataset_creation.conversion_tools import musicxml_to_svg, files_to_pdf


def gen_rand_str(max_length=30):
	size = range(random.randint(0, max_length))
	string = "".join(
		[RandomMusicGenerator.characters[random.randint(0, len(RandomMusicGenerator.characters) - 1)] for _ in size])
	return " ".join(map(lambda x: x if random.random() < 0.5 else x.capitalize(), string.split(" "))).strip()


class RandomMusicGenerator:
	characters = [chr(97 + i) for i in range(26)] + [" " for _ in range(3)] + [str(i) for i in range(10)]
	lengths = [1/4, 1/3, 2/3, 1/2, 3/4, 1]

	def __init__(self, num_instruments=1, note_range=(50, 100)):
		self.num_instruments = num_instruments
		self.note_range = note_range

		self.s = Session()
		self.s.fast_forward_to_beat(100000)
		self.is_slur = [0 for _ in range(num_instruments)]
		self.instruments = [self.s.new_part(gen_rand_str(15)) for _ in range(num_instruments)]

	def generate_note_properties(self, num_instrument, slur):
		properties = []

		harmonic = NoteProperties("notehead: harmonic", "pitch + 12")

		if random.random() < 0.5:
			properties.append(harmonic)

		if random.random() < 0.5:
			properties.append("staccato")

		if slur and self.is_slur[num_instrument] == -1:
			if random.random() < 0.05:
				self.is_slur[num_instrument] = random.randint(1, 8)
				properties.append(StartSlur())
		else:
			self.is_slur[num_instrument] -= 1

			if self.is_slur[num_instrument] == 0:
				properties.append(StopSlur())

		return properties

	def generate_note(self):
		return random.randint(self.note_range[0], self.note_range[1])

	def play(self, num_instrument, num_notes):
		inst = self.instruments[num_instrument]

		for i in range(random.randint(0, num_notes)):
			tone = random.randint(0, 1)

			volume = random.random()
			length = RandomMusicGenerator.lengths[random.randint(0, len(RandomMusicGenerator.lengths) - 1)]

			if tone == 0:
				properties = self.generate_note_properties(num_instrument, True)
				inst.play_note(self.generate_note(), volume, length, properties, "fermata")
			elif tone == 1:
				notes = [self.generate_note() for _ in range(0, random.randint(0, 3))]
				properties = self.generate_note_properties(num_instrument, False)
				inst.play_chord(notes, volume, length, properties, "fermata")

	def play_music_to_xml(self, path, name):
		self.s.start_transcribing()
		for i in range(self.num_instruments):
			self.s.fork(lambda _: self.play(i, num_notes=1000))
		self.s.wait_for_children_to_finish()

		performance = self.s.stop_transcribing()
		performance.export_to_midi_file(f"{path}/{name}.mid")
		performance.to_score(title=gen_rand_str(), composer=gen_rand_str()).to_music_xml().export_to_file(f"{path}/{name}.mxl")


if __name__ == '__main__':
	path = "dataset/generated"
	name = "test"
	generator = RandomMusicGenerator()
	generator.play_music_to_xml(f"{path}/musicxml", name)
	files = musicxml_to_svg(f"{path}/musicxml/{name}.mxl", f"{path}/render/", name)
	files_to_pdf(files, f"{path}/render/{name}.pdf")
