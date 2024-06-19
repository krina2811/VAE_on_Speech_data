
from preprocess import MinMaxNormaliser
import librosa
import torch
import numpy as np

class soundgenerator:
    def __init__(self, vae_model, hop_length):
        self.vae_model = vae_model
        self.hop_length = hop_length
        self._min_max_normaliser = MinMaxNormaliser(0, 1)


    def generate_ori_npy(self, spectrograms, min_max_values):                     
        reconstructed_sound, _, _ = self.vae_model(spectrograms)                    
        # reconstructed_audio = self.convert_sound_to_audio(reconstructed_sound, min_max_values)     
        return reconstructed_sound
    
    def generate_sound(self, spectrograms, min_max_values):                     
        reconstructed_sound, _, _ = self.vae_model(spectrograms)                    
        reconstructed_audio = self.convert_sound_to_audio(reconstructed_sound, min_max_values)     
        return reconstructed_audio


    def convert_sound_to_audio(self, spectrograms, min_max_values):
        signals = []
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):

            #convert 3d shape into 2d (remove channels)
            log_spectrogram = spectrogram[0, :, :]

            # denormalized the sound
            denorm_log_spec = self._min_max_normaliser.denormalize(log_spectrogram, min_max_value["min"], min_max_value["max"])

            ## convert log to amplitude
            spec = librosa.db_to_amplitude(denorm_log_spec.detach().cpu())

            # apply Griffin-Lim
            signal = librosa.istft(np.array(spec), hop_length=self.hop_length)
            # append signal to "signals"
            signals.append(signal)
        return signals




