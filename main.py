""""
reference: https://github.com/Data-Science-kosta/Speech-Emotion-Classification-with-PyTorch
"""
import keyboard
import torch

import config as cfg
import engine
from model import ParallelModel

if __name__ == '__main__':

    # SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOAD MODEL
    model = ParallelModel(len(cfg.EMOTIONS)).to(device)
    model.load_state_dict(torch.load(cfg.MODEL_PATH))
    print(f'Model is loaded from {cfg.MODEL_PATH} 🎉🎉')

    model.eval()

    while True:
        try: # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed(cfg.RECORD_KEY):
                engine.get_mic_audio()


                # RUN INFERENCE
                mel_spec_data = engine.load_audio_signal().to(device)
                output_logits, output_softmax = model(mel_spec_data)
                predictions = torch.argmax(output_softmax, dim=1)

                print(f'Predictions:{predictions} Predicted emotion: {cfg.EMOTIONS[predictions.item()]} Predicted color: {cfg.EMOTION_COLOR[cfg.EMOTIONS[predictions.item()]]}')

                engine.delete_audio()
        except:
            pass

