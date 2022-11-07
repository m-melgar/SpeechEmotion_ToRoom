import os
import wave

import cv2
import librosa
import numpy as np
import pyaudio

import config as cfg


def getMELspectrogram(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Computes mel spectrogram from audio np.ndarray

    :param audio: np.ndarray
                Audio section
    :param sample_rate: int
                sample rate for mel spectrogram
    :return: np.ndarray
                mel spectrogram
    """
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sample_rate,
                                              n_fft=1024,
                                              win_length=512,
                                              window='hamming',
                                              hop_length=256,
                                              n_mels=128,
                                              fmax=sample_rate / 2
                                              )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def load_audio_signal() -> np.ndarray:
    """
    Computes mel spectrogram from AUDIO_OPATH
    :return: np.ndarray
            mel spectrogram from AUDIO_PATH
    """
    audio, sample_rate = librosa.load(cfg.WAVE_OUTPUT_FILENAME, duration=cfg.AUDIO_DURATION, offset=0.5,
                                      sr=cfg.SAMPLE_RATE)
    signal = np.zeros((int(cfg.SAMPLE_RATE * 3, )))
    signal[:len(audio)] = audio
    mel_spectrogram = getMELspectrogram(signal, sample_rate=cfg.SAMPLE_RATE)

    # print("Mel spectogram shape", mel_spectrogram.shape)
    return mel_spectrogram


def get_mic_audio() -> None:
    """
    Records audio from microphone and stores it at WAVE_OUTPUT_FILENAME
    """
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

    with wave.open(cfg.WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(cfg.CHANNELS)
        wf.setsampwidth(p.get_sample_size(cfg.FORMAT))
        wf.setframerate(cfg.RATE)
        wf.writeframes(b''.join(frames))

    # wf = wave.open(cfg.WAVE_OUTPUT_FILENAME, 'wb')
    # wf.setnchannels(cfg.CHANNELS)
    # wf.setsampwidth(p.get_sample_size(cfg.FORMAT))
    # wf.setframerate(cfg.RATE)
    # wf.writeframes(b''.join(frames))
    # wf.close()


def delete_audio() -> None:
    """
    Deletes audio at WAVE_OUTPUT_FILENAME
    """
    if os.path.exists(cfg.AUDIO_PATH):
        os.remove(cfg.WAVE_OUTPUT_FILENAME)


def create_color(rgb_color: tuple[int, int, int] = (0, 0, 0),
                 width: int = cfg.WINDOW_W,
                 height: int = cfg.WINDOW_H) -> np.ndarray:
    """Create new image(numpy array) filled with certain color in RGB

    :param rgb_color: tuple[int, int, int]
                tuple representing an RGB color value
    :param width: int
                width of image array
    :param height: int
                height of image array
    :return: np.ndarray
                image array filled with solid color
    """
    # Create black blank image
    image = np.zeros((height, width, 3), dtype=np.int8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(rgb_color)
    # Fill image with color
    image[:] = color

    return image


def colored_window(rgbcolor: tuple[int, int, int]) -> None:
    """
    Displays rgbcolor with CV2
    """
    color_array = create_color(rgb_color=rgbcolor)
    cv2.imshow("emotion", color_array)
