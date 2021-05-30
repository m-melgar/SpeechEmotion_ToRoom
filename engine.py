import os
import wave

import librosa
import numpy as np
import pyaudio

import config as cfg


def getMELspectrogram(audio, sample_rate):
    """
    retuns mel spectogram
    :param audio:
    :param sample_rate:
    :return:
    """
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sample_rate,
                                              n_fft=1024,
                                              win_length = 512,
                                              window='hamming',
                                              hop_length = 256,
                                              n_mels=128,
                                              fmax=sample_rate/2
                                             )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def load_audio_signal():
    """
    gets audio from AUDIO_PATH and return its mel spectrogram
    :return: mel spectrogram
    """
    audio, sample_rate = librosa.load(cfg.AUDIO_PATH, duration=cfg.AUDIO_DURATION, offset=0.5, sr=cfg.SAMPLE_RATE)
    signal = np.zeros((int(cfg.SAMPLE_RATE * 3, )))
    signal[:len(audio)] = audio
    mel_spectrogram = getMELspectrogram(signal, sample_rate=cfg.SAMPLE_RATE)

    return mel_spectrogram

def get_mic_audio():


    p = pyaudio.PyAudio()

    stream = p.open(format=cfg.FORMAT,
                    channels=cfg.CHANNELS,
                    rate=cfg.RATE,
                    input=True,
                    frames_per_buffer=cfg.CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(cfg.RATE / cfg.CHUNK * cfg.RECORD_SECONDS)):
        data = stream.read(cfg.CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(cfg.WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(cfg.CHANNELS)
    wf.setsampwidth(p.get_sample_size(cfg.FORMAT))
    wf.setframerate(cfg.RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def delete_audio():

    if os.path.exists(cfg.AUDIO_PATH):
        os.remove(cfg.AUDIO_PATH)