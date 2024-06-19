import librosa
import numpy as np
import os
import pickle


class Loader:
    """load the audio file using librosa"""
    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono
    
    def load(self, file_path):
        signal = librosa.load(file_path,
                            sr=self.sample_rate,
                            duration=self.duration,
                            mono=self.mono )[0]
        return signal

class Padder:
    """add padding is it necessary"""
    def __init__(self, mode="constant"):
        self.mode = mode
    
    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array, (num_missing_items, 0), self.mode )
        return padded_array
    
    def right_pad(self, array, num_missing_items):
        
        padded_array = np.pad(array,
                              (0, num_missing_items),
                              mode=self.mode)
        return padded_array
    



class LogSpectrogramExtractor:
    """extract mel-spectrogram frpm signal"""
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal, 
                            n_fft = self.frame_size,
                            hop_length = self.hop_length
                            )[:-1]    ### return shape ---- (1 + frame_size/2, n_fft)--- so here we got 513 as first shape but we want to deal with even number so do slicing
    
        spectogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectogram)
        return log_spectrogram

class MinMaxNormaliser:
    "apply minmax normalization"
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value
    
    def normalize(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())  ## this return the array bet [0,1] if we want the array bet min max value have to do next step
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array
    
    def denormalize(self, array, original_min, original_max):  ## reverse process of normlization
        array = (array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array



class Saver:
    """save features and min_max values"""
    def __init__(self, feature_save_dir, min_max_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_save_dir = min_max_save_dir

    def save_feature(self, feature, audio_path):
        save_path = self._generate_save_path(audio_path)
        # np.save(save_path, feature)
        return save_path

    def save_min_max_value(self, min_max_values):
        save_path = os.path.join(self.min_max_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    
    def _generate_save_path(self, audio_path):
        
        file_name = audio_path.split("/")[3].split(".")[0]
        
        save_path = os.path.join(self.feature_save_dir, file_name + '.npy')
        return save_path


class PreprocessingPipeline:
    """load files from directory and apply each above steps"""
    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normalizer = None
        self.saver = None
        self.min_max_value = {}
        self._loader = None

        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = loader.sample_rate * loader.duration
    
    
    def process(self, audio_file_dir):
        for root, _, files in os.walk(audio_file_dir):
            for audio_path in files:
                audio_path = os.path.join(root, audio_path)
                self._process_file(audio_path)
                print(f"Process file {audio_path}")
            
        self.saver.save_min_max_value(self.min_max_value)

    def _process_file(self, audio_path):
        signal = self.loader.load(audio_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normalizer.normalize(feature)
        # print(norm_feature.shape)
        save_path = self.saver.save_feature(norm_feature, audio_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        else:
            return False
        
    
    def _apply_padding(self, signal):
        num_missing_samples = int(self._num_expected_samples - len(signal))
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal


    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_value[save_path] = {
            "min": min_val,
            "max": max_val
        }


if __name__ == "__main__":

    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74
    SAMPLE_RATE = 22050
    MONO = True

    audio_dir = "./dataset/recordings"
    spectrogram_save_dir = "./dataset/spectrograms"
    min_max_save_dir = "./dataset/min_max_values"

    ## initantiate all objects
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normlizer = MinMaxNormaliser(0, 1)
    saver = Saver(spectrogram_save_dir, min_max_save_dir)

    preprocess_pipeline = PreprocessingPipeline()
    preprocess_pipeline.loader = loader
    preprocess_pipeline.padder = padder
    preprocess_pipeline.extractor = extractor
    preprocess_pipeline.normalizer = min_max_normlizer
    preprocess_pipeline.saver = saver

    preprocess_pipeline.process(audio_dir)