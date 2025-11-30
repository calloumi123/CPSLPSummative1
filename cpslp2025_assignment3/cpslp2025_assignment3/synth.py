"""A simple diphone synthesiser programme. With extensions A,B,C,D,E"""

import re
import wave
from pathlib import Path
from typing import List, Dict

from nltk.corpus import cmudict
import numpy as np
import simpleaudio

from synth_args import process_commandline

def strip_stress_markers(phone: str) -> str:
    """Remove stress markers (0-9) from the end of a phone."""
    return phone.rstrip("0123456789")

# import numpy
# ...others?  (only modules that come as standard with Python3, so
# they will be available on marker machines)

# Below you'll find a pair of classes (Synth and Utterance) to start you off
# (you can change these as you like, adding extra methods/attributes/code, but
# don't change the API of what is provided initially here - ask if in doubt!)


class Synth:
    """A simple diphone synthesiser class."""

    def __init__(self, wav_folder: str) -> None:
        self.diphones, self.sample_rate = self.get_wavs(wav_folder)

    def get_wavs(self, wav_folder):
        """Loads all the waveform data contained in WAV_FOLDER.
        Returns a dictionary, with unit names as keys and the corresponding
        loaded audio data as values."""
    
        diphones: Dict[str, np.ndarray] = {}
        wav_files: list[Path] = list(Path(wav_folder).glob("*.wav"))
        
        if not wav_files:
            raise ValueError(f"No wav files found in '{wav_folder}'")
        
        first_audio = simpleaudio.Audio()
        first_audio.load(str(wav_files[0]))
        sample_rate: int = first_audio.rate
        diphones[wav_files[0].stem] = first_audio.data
        
        for wav_file in wav_files[1:]:
            audio_data = simpleaudio.Audio()
            audio_data.load(str(wav_file))
            
            if audio_data.rate != sample_rate:
                raise ValueError(f"Sample rate mismatch in '{wav_file.name}'")
            diphones[wav_file.stem] = audio_data.data
            
        return diphones, sample_rate

    def crossfade(self, list_of_diphone_waves: List[np.ndarray]) -> np.ndarray:
        """
        Crossfades a list of diphone waveforms into a single waveform.
        :param list_of_diphone_waves: List of diphone waveforms (list of np.ndarray)
        :return: Crossfaded waveform (np.ndarray)
        """
        combined: np.ndarray = np.array([], dtype=np.int16)
        overlap_size = int(self.sample_rate * 0.01)  # 10 ms crossfade
        fade_window = np.hanning(
            overlap_size * 2
        )  # 2 because we need fade in and fade out
        fade_in = fade_window[:overlap_size]
        fade_out = fade_window[overlap_size:]

        for i, a_wave in enumerate(list_of_diphone_waves[:-1]):  # All except last
            next_wave = list_of_diphone_waves[i + 1]
            end_wave = a_wave[-overlap_size:]
            start_next_wave = next_wave[:overlap_size]

            # Apply Hann and average
            faded_end = end_wave * fade_out
            faded_start = start_next_wave * fade_in
            overlapped = (faded_end + faded_start) / 2

            # Update both diphones
            list_of_diphone_waves[i] = np.concatenate(
                (a_wave[:-overlap_size], overlapped)
            )
            list_of_diphone_waves[i + 1] = next_wave[
                overlap_size:
            ]  
        # Concatenate all
        combined = np.concatenate(list_of_diphone_waves)
        return combined

    def synthesise(
        self, phones: List[str], reverse: str | None = None, crossfade: bool = False
    ) -> simpleaudio.Audio:
        """
        Synthesises a phone list into an audio utterance.
        :param phones: list of phones (list of strings)
        :param reverse: "words", "phones", "signal", or None
        :param smooth_concat: use crossfading
        :return: synthesised utterance (Audio instance)
        """
        diphones_seq: List[str] = self.phones_to_diphones(phones)
        combined: np.ndarray = np.array([], dtype=np.int16)

        if crossfade:
            smooth_list: List[np.ndarray] = []
            for diphone in diphones_seq:
                if diphone.lower() in self.diphones:
                    smooth_list.append(self.diphones[diphone.lower()])
            print(smooth_list)
            combined = self.crossfade(smooth_list)
        else:
            diphones_to_be_said: List[np.ndarray] = []
            for diphone in diphones_seq:
                if diphone.lower() in self.diphones:
                    diphones_to_be_said.append(self.diphones[diphone.lower()])
            combined = np.concatenate(
                        (diphones_to_be_said)
                    )

        if reverse == "signal":
            combined = combined[::-1]

        audio: simpleaudio.Audio = simpleaudio.Audio()
        audio.data = combined
        return audio

    def phones_to_diphones(self, phones: List[str]) -> List[str]:
        """
        Converts a list of phones to the corresponding diphone units.
        :param phones: list of phones (list of strings)
        :return: list of diphones (list of strings)
        """
        diphones_seq: List[str] = []
        for i in range(len(phones) - 1):
            diphones_seq.append(f"{phones[i]}-{phones[i+1]}")
        return diphones_seq

class Utterance:
    """Class to represent and process an utterance."""

    def __init__(self, phrase: str) -> None:
        """
        Constructor takes a phrase to process.
        :param phrase: a string which contains the phrase to process.
        """
        print(f"Processing phrase: {phrase}")  
        self.phrase: str = phrase

    def normalize_text(self, text: str) -> List[str]:
        """Normalize and split text into words."""
        text = text.lower()
        puncts: str = ".,;:!?\"'()[]{}—–-"

        for sym in puncts:
            text = text.replace(sym, " ")
        textlist: List[str] = text.split()
        return textlist

    def load_pronunciations(self, pron_file: str | Path) -> Dict[str, List[List[str]]]:
        """Load custom pronunciations from a file.
        File format: word:pos:pos_tag:index:{ PHONE1 PHONE2 ... }::
        """
        custom_prons: Dict[str, List[List[str]]] = {}
        try:
            with open(pron_file, "r", encoding="utf-8") as f:
                content: str = f.read() 
            pattern: re.Pattern = re.compile(r"^(\w+):.*?\{([^}]+)\}", re.MULTILINE)
            matches: List[tuple] = pattern.findall(content)

            def strip_stress_markers(phone: str) -> str:
                """Remove stress markers (0-9) from the end of a phone."""
                return phone.rstrip("0123456789")

            for word, phones_str in matches:
                word_lower: str = word.lower()
                phones: List[str] = [
                    strip_stress_markers(p.strip()) for p in phones_str.split()
                ]
                custom_prons[word_lower] = [phones]

            if custom_prons:
                print(
                    f"Loaded {len(custom_prons)} custom pronunciations: {custom_prons}"
                )
            else:
                print("No pronunciations found in file")
        except Exception as e:
            print(f"Error loading custom pronunciations: {e}")

        return custom_prons

    def text_to_phones(
        self, words: List[str], addpron: str | Path | None = None
    ) -> List[str]:
        """Convert words to phone sequence with PAU."""
        cmu_dict: dict = cmudict.dict()
        addpron_dict: Dict[str, List[List[str]]] = {}

        if isinstance(addpron, (str, Path)):
            addpron_dict = self.load_pronunciations(str(addpron))

        phones: List[str] = ["PAU"]  

        for word in words:
            if word in addpron_dict:
                word_phones: List[str] = addpron_dict[word][0] # Already cleaned in load_pronunciations, no stripping needed
                phones.extend(word_phones)
            elif word in cmu_dict:
                # Strip stress markers from CMU dict
                word_phones: List[str] = [
                    phone.rstrip("0123456789") for phone in cmu_dict[word][0]
                ]
                phones.extend(word_phones)
            else:
                print(f"Warning: '{word}' not found")

        phones.append("PAU")  # End silence
        return phones

    def get_phone_seq(
        self, reverse: str | None = None, addpron: str | None = None
    ) -> List[str]:
        """
        Returns the phone sequence corresponding to the text in this Utterance (i.e. self.phrase)
        :param reverse:  Whether to reverse something.  Either "words", "phones" or None
        :return: list of phones (as strings)
        """
        words: List[str] = self.normalize_text(self.phrase)

        if reverse == "words":
            words = words[::-1]
        elif reverse not in ["phones", "signal", None]:
            print(
                "You entered something that was not 'words' or 'phones', so we set it to None"
            )
            reverse = None

        phones_seq: List[str] = self.text_to_phones(words, addpron)
        if reverse == "phones":
            phones_seq = phones_seq[::-1]

        return phones_seq

def run_synthesis(phone_seq: List[str], args, wav_folder: str) -> simpleaudio.Audio:
    diphone_synth: Synth = Synth(wav_folder=wav_folder)
    audio: simpleaudio.Audio = diphone_synth.synthesise(phone_seq, reverse=args.reverse, crossfade=args.crossfade)
    if args.volume is not None:
        if not 0 <= args.volume <= 100:
            raise ValueError("Volume must be between 0 and 100")
        audio.rescale(args.volume / 100.0)
    return audio

def save_and_play(audio: simpleaudio.Audio, args) -> None:
    if args.outfile:
        audio.save(args.outfile)
    if args.play:
        audio.play()

def apply_volume(audio: simpleaudio.Audio, volume: int | None) -> None:
    """Apply volume scaling to audio if specified."""
    if volume is not None:
        if not 0 <= volume <= 100:
            raise ValueError("Volume must be between 0 and 100")
        audio.rescale(volume / 100.0)

def process_file(textfile: str, args) -> None:
    """
    Takes the path to a text file and synthesises each sentence it contains
    :param textfile: the path to a text file (string)
    :param args: the parsed command line argument object giving options
    :return: None
    """
    synth: Synth = Synth(wav_folder=args.diphones)
    
    with open(textfile, "r", encoding="utf-8") as f:
        sentences: List[str] = re.split(r"(?<=[.!?])\s+", f.read().strip())
    
    silence: np.ndarray = np.zeros(int(synth.sample_rate * 0.4), dtype=np.int16)
    chunks: List[np.ndarray] = []
    
    for phrase in sentences:
        phone_seq: List[str] = Utterance(phrase).get_phone_seq(reverse=args.reverse, addpron=args.addpron)
        audio: simpleaudio.Audio = synth.synthesise(phone_seq, reverse=args.reverse, crossfade=args.crossfade)
        apply_volume(audio, args.volume)
        
        if chunks:
            chunks.append(silence)
        chunks.append(audio.data)
    
    combined: simpleaudio.Audio = simpleaudio.Audio(rate=synth.sample_rate)
    combined.data = np.concatenate(chunks)
    save_and_play(combined, args)

def main(args) -> None:
    """Main function."""
    print(args)
    if args.fromfile:
        process_file(args.fromfile, args)
    else:
        phone_seq: List[str] = Utterance(args.phrase).get_phone_seq(reverse=args.reverse, addpron=args.addpron)
        audio: simpleaudio.Audio = run_synthesis(phone_seq, args, args.diphones)
        apply_volume(audio, args.volume)
        save_and_play(audio, args)

if __name__ == "__main__":
    main(process_commandline())
    # Store or process audio as needed

# Make this the top-level "driver" function for your programme.  There are some hints here
# to get you going, but you will need to add all the code to make your programme behave
# correctly according to the commandline options given in args (and assignment description!).

    # do what you like with "out"...

# DO NOT change or add anything below here
# (it just parses the commandline and calls your "main" function
# with the options given by the user)
if __name__ == "__main__":
    main(process_commandline())
