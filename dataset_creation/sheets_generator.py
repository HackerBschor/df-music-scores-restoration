from typing import List, Any, Tuple

import pymusicxml
from scamp import StartSlur, Session, StopSlur, wait, SpellingPolicy, ScampInstrument, Performance, Score
import random

allowed_characters: List[chr] = [chr(97 + i) for i in range(26)] + [" " for _ in range(3)] + [str(i) for i in range(10)]
lengths: List[float] = [1 / 4, 1 / 3, 2 / 3, 1 / 2, 3 / 4, 1]


def gen_rand_str(max_length: int = 30) -> str:
    size: range = range(random.randint(0, max_length))
    rand_string: str = "".join([allowed_characters[random.randint(0, len(allowed_characters) - 1)] for _ in size])
    return " ".join(map(lambda x: x if random.random() < 0.5 else x.capitalize(), rand_string.split(" "))).strip()


def get_rand_from_list(element_list: List[Any]) -> Any:
    return element_list[random.randint(0, len(element_list) - 1)]


def calc_length_to_full_beat(beat: float) -> float:
    return 0.0 if int(beat) == beat else (int(beat) + 1) - beat


def convert_to_xml(performance: Performance) -> pymusicxml.Score:
    while True:
        tim_sig = f"{get_rand_from_list([i for i in range(32)])} / {get_rand_from_list([2 ** i for i in range(6)])}"

        try:
            score: Score = performance.to_score(
                title=gen_rand_str(), composer=gen_rand_str(), time_signature=tim_sig)
            return score.to_music_xml()
        except ValueError:
            pass


class RandomMusicGenerator:
    def __init__(self, lim_num_instruments: Tuple[int] = (1, 10), lim_bass_notes: Tuple[int] = (20, 60),
                 lim_violin_notes: Tuple[int] = (60, 100), lim_num_beats: Tuple[int] = (100, 1000)):

        self.num_instruments = random.randint(lim_num_instruments[0], lim_num_instruments[1])
        self.note_ranges: List[Tuple[int]] = [get_rand_from_list([lim_bass_notes, lim_violin_notes])
                                              for _ in range(self.num_instruments)]
        self.num_beats: int = random.randint(lim_num_beats[0], lim_num_beats[1])

        self.s: Session = Session(tempo=(random.random() * 100 + 40))
        self.s.fast_forward_to_beat(self.num_beats)
        self.is_slur: List[int] = [0 for _ in range(self.num_instruments)]
        self.instruments: List[ScampInstrument] = [self.s.new_part(gen_rand_str(15))
                                                   for _ in range(self.num_instruments)]
        self.curr_beat: List[int] = [0 for _ in range(self.num_instruments)]

    def generate_note_properties(self, num_instrument: int, slur: bool) -> List[Any]:
        properties: List[Any] = []

        # Add pitch
        if random.random() < 0.2:
            properties.append(SpellingPolicy.from_string("#" if random.random() < 0.5 else "b"))

        # Add staccato
        if random.random() < 0.2:
            properties.append("staccato")

        # Add or remove slur
        if slur and self.is_slur[num_instrument] == -1:
            if random.random() < 0.05:
                # Add slur for 1-8 notes
                self.is_slur[num_instrument] = random.randint(1, 8)
                properties.append(StartSlur())
        else:
            # Decrease slur counter
            self.is_slur[num_instrument] -= 1
            # Stop Slur if counter is 0
            if self.is_slur[num_instrument] == 0:
                properties.append(StopSlur())

        return properties

    def generate_note(self, num_instrument: int) -> int:
        return random.randint(self.note_ranges[num_instrument][0], self.note_ranges[num_instrument][1])

    def play(self, num_instrument: int) -> None:
        inst: ScampInstrument = self.instruments[num_instrument]

        while self.curr_beat[num_instrument] < self.num_beats:
            tone: float = random.random()
            volume: float = random.random()
            length: float = lengths[random.randint(0, len(lengths) - 1)]

            # Add Note
            if 0 <= tone < 0.48:
                properties: List[Any] = self.generate_note_properties(num_instrument, True)
                inst.play_note(self.generate_note(num_instrument), volume, length, properties)
            # Add Chord
            elif 0.48 <= tone == 0.96:
                notes = [self.generate_note(num_instrument) for _ in range(0, random.randint(0, 3))]
                properties: List[Any] = self.generate_note_properties(num_instrument, False)
                inst.play_chord(notes, volume, length, properties)
            # Finishes slur if necessary and plays add pause
            else:
                self.is_slur[num_instrument] = 1 if self.is_slur[num_instrument] > 1 else -1
                properties: List[Any] = self.generate_note_properties(num_instrument, False)
                length = calc_length_to_full_beat(self.curr_beat[num_instrument])
                if length > 0:
                    inst.play_note(self.generate_note(num_instrument), volume, length, properties)

                wait(1)
                length += 1

            self.curr_beat[num_instrument] += length

    def play_music_to_xml(self, path: str, name: str, export_midi=False):
        # Play all instruments and transform them into performance
        self.s.start_transcribing()

        for i in range(self.num_instruments):
            self.s.fork(lambda _: self.play(i))

        self.s.wait_for_children_to_finish()

        performance: Performance = self.s.stop_transcribing()

        # Saves audio if required
        if export_midi:
            performance.export_to_midi_file(f"{path}/{name}.mid")

        # Export to MusicXML
        score_xml: pymusicxml.Score = convert_to_xml(performance)
        score_xml.export_to_file(f"{path}/{name}.mxl")

        self.s.kill()
