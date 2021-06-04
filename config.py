import os

import pyaudio

"""
install pyaudio:
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
sudo apt-get install ffmpeg libav-tools
sudo pip install pyaudio

"""

EMOTIONS = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 0: 'surprise'}
EMOTION_COLOR = {'neutral': (255, 255, 255), 'calm': (196, 225, 231), 'happy': (255, 246, 0), 'sad': (102, 102, 102),
                 'angry': (227, 22, 22), 'fear': (0, 0, 0), 'disgust': (50, 205, 50), 'surprise': (245, 135, 41)}

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

WINDOW_W = 200
WINDOW_H = 200

if not os.path.exists(AUDIO_PATH):
    os.mkdir(AUDIO_PATH)
    print(f'Creating audio folder {AUDIO_PATH} 🎤')

RECORD_KEY = 'r'
