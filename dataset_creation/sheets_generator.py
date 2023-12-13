from scamp import StartSlur, Session, StopSlur, wait, SpellingPolicy
import random

characters = [chr(97 + i) for i in range(26)] + [" " for _ in range(3)] + [str(i) for i in range(10)]
lengths = [1 / 4, 1 / 3, 2 / 3, 1 / 2, 3 / 4, 1]


def gen_rand_str(max_length: int = 30):
	size = range(random.randint(0, max_length))
	string = "".join([characters[random.randint(0, len(characters) - 1)] for _ in size])
	return " ".join(map(lambda x: x if random.random() < 0.5 else x.capitalize(), string.split(" "))).strip()


def get_rand_from_list(element_list):
	return element_list[random.randint(0, len(element_list) - 1)]


def calc_length_to_full_beat(beat):
	return 0 if int(beat) == beat else (int(beat) + 1) - beat


class RandomMusicGenerator:
	def __init__(self, num_instruments=(1, 10), bass_notes=(20, 60), violin_notes=(60, 100), num_beats=(100, 1000)):
		self.num_instruments = random.randint(num_instruments[0], num_instruments[1])
		self.note_range = [get_rand_from_list([bass_notes, violin_notes]) for _ in range(self.num_instruments)]
		self.num_beats = random.randint(num_beats[0], num_beats[1])

		self.s = Session(tempo=(random.random() * 100 + 40))
		self.s.fast_forward_to_beat(self.num_beats)
		self.is_slur = [0 for _ in range(self.num_instruments)]
		self.instruments = [self.s.new_part(gen_rand_str(15)) for _ in range(self.num_instruments)]
		self.curr_beat = [0 for _ in range(self.num_instruments)]

	def generate_note_properties(self, num_instrument, slur):
		properties = []
		if random.random() < 0.2:
			properties.append(SpellingPolicy.from_string("#" if random.random() < 0.5 else "b"))

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

	def generate_note(self, num_instrument):
		return random.randint(self.note_range[num_instrument][0], self.note_range[num_instrument][1])

	def play(self, num_instrument):
		inst = self.instruments[num_instrument]

		while self.curr_beat[num_instrument] < self.num_beats:
			tone = random.random()

			volume = random.random()
			length = lengths[random.randint(0, len(lengths) - 1)]

			if 0 <= tone < 0.48:
				properties = self.generate_note_properties(num_instrument, True)
				inst.play_note(self.generate_note(num_instrument), volume, length, properties) # "fermata"
			elif 0.48 <= tone == 0.96:
				notes = [self.generate_note(num_instrument) for _ in range(0, random.randint(0, 3))]
				properties = self.generate_note_properties(num_instrument, False)
				inst.play_chord(notes, volume, length, properties)
			else:
				self.is_slur[num_instrument] = 1 if self.is_slur[num_instrument] > 1 else -1
				properties = self.generate_note_properties(num_instrument, False)
				length = calc_length_to_full_beat(self.curr_beat[num_instrument])
				if length > 0:
					inst.play_note(self.generate_note(num_instrument), volume, length, properties)
				wait(1)
				length += 1

			self.curr_beat[num_instrument] += length

	def play_music_to_xml(self, path, name, export_midi=False):
		self.s.start_transcribing()
		for i in range(self.num_instruments):
			self.s.fork(lambda _: self.play(i))
		self.s.wait_for_children_to_finish()

		performance = self.s.stop_transcribing()

		if export_midi:
			performance.export_to_midi_file(f"{path}/{name}.mid")

		score_xml = None
		while score_xml is None:
			tim_sig = f"{get_rand_from_list([i for i in range(32)])} / {get_rand_from_list([2 ** i for i in range(6)])}"
			try:
				score = performance.to_score(title=gen_rand_str(), composer=gen_rand_str(), time_signature=tim_sig)
				score_xml = score.to_music_xml()
			except ValueError:
				pass

		score_xml.export_to_file(f"{path}/{name}.mxl")

		self.s.kill()