import os
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import play

AUDIO_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))

fly_heading = AudioSegment.from_wav(AUDIO_ROOT / "fly_heading.wav")
numeric = {
    '0': AudioSegment.from_wav(AUDIO_ROOT / "0.wav"),
    '1': AudioSegment.from_wav(AUDIO_ROOT / "1.wav"),
    '2': AudioSegment.from_wav(AUDIO_ROOT / "2.wav"),
    '3': AudioSegment.from_wav(AUDIO_ROOT / "3.wav"),
    '4': AudioSegment.from_wav(AUDIO_ROOT / "4.wav"),
    '5': AudioSegment.from_wav(AUDIO_ROOT / "5.wav"),
    '6': AudioSegment.from_wav(AUDIO_ROOT / "6.wav"),
    '7': AudioSegment.from_wav(AUDIO_ROOT / "7.wav"),
    '8': AudioSegment.from_wav(AUDIO_ROOT / "8.wav"),
    '9': AudioSegment.from_wav(AUDIO_ROOT / "9.wav"),
}

def play_fly_heading(heading: str):
    """
    heading in string ex) "030"
    """
    play(fly_heading)
    for s in heading:
        play(numeric[s])
