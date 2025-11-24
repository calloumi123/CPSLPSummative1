### Speech Synthesiser Implementation
A Python-based speech synthesis system that converts text input into intelligible synthesised speech using diphone concatenation with multiple extensions.
#### Overview
This project implements a waveform concatenation synthesiser where acoustic units are pre-recorded diphones (audio segments capturing transitions between speech sounds). 
The system normalises input text, converts words to phonemes using the CMU pronunciation dictionary, maps phonemes to diphones, and concatenates corresponding audio files 
to produce synthesised speech output.

### Features
#### Core Functionality

Text-to-speech synthesis: Convert written text directly into audio waveforms
CMU dictionary integration: Automatic word-to-phoneme conversion using NLTK
Diphone concatenation: Seamless joining of pre-recorded speech segments
Playback and file saving: Play synthesised speech or save to .wav files
Robust error handling: Graceful handling of out-of-vocabulary words, missing diphones, and malformed input

#### Extensions Implemented

##### Extension A - Volume Control ✓
  Adjust output amplitude between 0-100 using the --volume flag to control loudness.
##### Extension B - Speaking Backwards ✓
  Three reversal modes:
    --reverse signal: Reverse the entire waveform back-to-front
    --reverse words: Reverse the order of words before synthesis
    --reverse phones: Reverse the phoneme sequence

##### Extension D - Synthesising from File ✓
  Use --fromfile to process entire text files sentence by sentence. Automatically inserts 400ms silence between sentences. 
  All output can be saved to a single file with --outfile.
##### Extension E - Smoother Concatenation ✓
  Cross-fade between adjacent diphones using 10ms Hann window overlap to reduce audible glitches at concatenation boundaries.

#### Installation
  
  Requirements (pip install them) 
  
  Python 3.7+
  numpy
  nltk (with cmudict corpus)
  scipy

#### Setup
bash# Clone the repository
git clone https://github.com/calloumi123/CPSLPSummative1.git
cd CPSLPSummative1

#### Install dependencies
pip install numpy nltk scipy

#### Download required NLTK data
    python -c "import nltk; nltk.download('cmudict')"
    Usage
    Basic Synthesis
    bash# Play synthesised speech
    python synth.py -p "hello nice to meet you"
  
#### Save to file
  python synth.py -o output.wav "hello nice to meet you"
  With Extensions
  Volume Control:
  bashpython synth.py -p "hello" --volume 75
  Reversal Effects:
  bashpython synth.py -p "hello" --reverse signal   # Reverse waveform
  python synth.py -p "hello" --reverse words    # Reverse word order
  python synth.py -p "hello" --reverse phones   # Reverse phoneme order
  Custom Pronunciations:
  bashpython synth.py -p "hello scunner" --addpron pronunciations.txt
  File Processing:
  bashpython synth.py --fromfile input.txt -o output.wav
  Smoother Audio:
  bashpython synth.py -p "hello" --crossfade
  Combined:
  bashpython synth.py --fromfile input.txt -o output.wav --volume 80 --crossfade


## Project Structure
```
CPSLPSummative1/
├── synth.py              # Main synthesiser implementation
├── synth_args.py         # Command-line argument parsing (provided)
├── simpleaudio.py        # Audio file handling utilities (provided)
├── test_synth.py         # Unit tests
├── diphones/             # Pre-recorded diphone audio files
├── examples/             # Reference output examples
└── README.md
```
Implementation Details
Core Architecture
Synth Class

Loads all diphone .wav files into memory as numpy arrays
Provides phones_to_diphones() to convert phoneme sequences to diphone pairs
Implements synthesise() which concatenates diphones and applies effects
Includes crossfade() method for smooth audio transitions

Utterance Class

Handles text normalisation (lowercase, punctuation removal)
Converts word sequences to phoneme sequences using CMU dictionary
Supports custom pronunciation loading from external files
Manages reversal operations at word or phoneme level

#### Main Processing Pipeline

Normalise input text to extract words
Look up words in CMU dictionary (or custom pronunciations if provided)
Frame phoneme sequence with silence markers (PAU)
Convert phonemes to diphone pairs
Load and concatenate corresponding audio files
Apply effects (volume, reversal, cross-fading)
Save to file or play audio

#### Key Implementation Decisions
Type Hints: Full type annotations throughout for clarity and maintainability, appropriate for data science workflows.
Error Handling: Words not found in the dictionary are logged as warnings rather than causing crashes, allowing partial synthesis of mixed-vocabulary input.
Custom Pronunciations: Regular expression parsing of pronunciation file format allows users to extend the dictionary with domain-specific or non-English words.
Cross-fading Algorithm: Uses Hann windowing with 10ms overlap at diphone boundaries to create smooth transitions and reduce audible artifacts.
File Processing: Sentences are detected by sentence-ending punctuation (., !, ?) and processed individually, with 400ms silence inserted between them as specified.
Testing
Run the provided test suite to verify implementation:
bashpython -m pytest test_synth.py -v
Technical Notes

Sample rate: 48kHz
Audio format: 16-bit PCM mono
Crossfade window: 10ms Hann window
Inter-sentence silence: 400ms
Stress markers automatically stripped from CMU phonemes for diphone matching

##### License
This project was completed as part of a university coursework assignment at the School of Informatics, University of Edinburgh.
