import os

import pyaudio
"""
install pyaudio:
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
sudo apt-get install ffmpeg libav-tools
sudo pip install pyaudio

"""

EMOTIONS = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 0:'surprise'}
EMOTION_COLOR = {'neutral':0xffffff, 'calm':0xc4e1e7, 'happy':0xfff600, 'sad':0x666666, 'angry':0xe31616, 'fear':0x000000, 'disgust':0x32cd32, 'surprise':0xf58729}

MODEL_PATH = './models/cnn_transf_parallel_model.pt'
AUDIO_PATH = './audio'

SAMPLE_RATE = 48000
AUDIO_DURATION = 3

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = SAMPLE_RATE
RECORD_SECONDS = AUDIO_DURATION
WAVE_OUTPUT_FILENAME = os.path.join(AUDIO_PATH, './output.wav')

if not os.path.exists(AUDIO_PATH):
    os.mkdir(AUDIO_PATH)
    print(f'Creating audio folder {AUDIO_PATH} 🎤')


RECORD_KEY = 'r'