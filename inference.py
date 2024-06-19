import os
import pickle
import torch
import numpy as np
import soundfile as sf
import glob
from soundgenerator import soundgenerator
from autoencoder import Autoencoder
import shutil
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

HOP_LENGTH = 256
spectrogram_path = "./dataset/spectrograms/"
SAVE_DIR_GENERATED = "./dataset/generated_ori_sound/"
MIN_MAX_VALUES_PATH = "./dataset/min_max_values/min_max_values.pkl"
test_data_dir = "./dataset/test_data/"
wav_file_path = "./dataset/recordings/"

os.makedirs(SAVE_DIR_GENERATED, exist_ok=True)
os.makedirs(test_data_dir, exist_ok=True)


def select_spectrograms(file_paths,
                        min_max_values,
                        num_spectrograms=8):
    
    sampled_indexes = np.random.choice(range(len(file_paths)), num_spectrograms)
    
    file_paths = [file_paths[index] for index in sampled_indexes]
    
    sampled_min_max_values = [min_max_values[file_path] for file_path in
                           file_paths]
    print(file_paths)
    print(sampled_min_max_values)
    return sampled_min_max_values, file_paths

def save_signals(signals, audio_path, save_dir, sample_rate=22050):
    for signal, file_path in zip(signals, audio_path):
        save_path = file_path.split("/")[-1].split(".")[0]
        save_path = os.path.join(save_dir, save_path + "_generated.wav")
        sf.write(save_path, signal, sample_rate)

class AudioDataset_test(Dataset):

    def __init__(self, spectrogram_path):
        super(AudioDataset_test, self).__init__()
        self.spectrogram_path = spectrogram_path
        self.spectrogram_list =glob.glob(os.path.join(spectrogram_path, "*.npy"))
        
    def __getitem__(self, index):
        spec_npy_path = self.spectrogram_list[index]
        spectrogram = np.load(spec_npy_path)
        mel_sp = torch.tensor(spectrogram).unsqueeze(0)
        return mel_sp, spec_npy_path

    def __len__(self):
        return len(self.spectrogram_list)


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = Autoencoder(input_shape=(1, 256, 64),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2,1)),
        latent_space_dim=128)


    model.load_state_dict(torch.load('./checkpoint/model.pt'))
    model.to(device)
    model.eval()

    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)

    
    sound_generator = soundgenerator(model, HOP_LENGTH)


    file_path_list = glob.glob(os.path.join(spectrogram_path, "*.npy"))
    sample_min_max_value, sample_file_paths = select_spectrograms(file_path_list, min_max_values)

    for audio_path in sample_file_paths:
        shutil.copyfile(audio_path, os.path.join(test_data_dir, audio_path.split("/")[-1]))

    valid_set = AudioDataset_test(test_data_dir)
    valid_loader = DataLoader(
            valid_set,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            num_workers=3,
        )

    tbar = tqdm(valid_loader, ncols=80)


    ##################################### find the difference between original npu and generated npy ################################################
    ori_array_list = []
    gen_array_list = []
    
    for i, (waveform_spec, audio_path) in enumerate(tbar):
        signals = sound_generator.generate_ori_npy(torch.tensor(waveform_spec).to(device), sample_min_max_value)
        ori_array_list.append(waveform_spec)
        gen_array_list.append(signals)

    for ori_audio, gen_audio in zip(ori_array_list, gen_array_list):
        gen_audio = gen_audio.detach().cpu()       
        norm_vector1 = np.linalg.norm(ori_audio)
        norm_vector2 = np.linalg.norm(gen_audio)
        print("diff:", np.mod(norm_vector1, norm_vector2))



    ###################################### generate the original audio wav file and save it into directory  ############################################
    for i, (waveform_spec, audio_path) in enumerate(tbar):
        audio_list = sound_generator.generate_sound(torch.tensor(waveform_spec).to(device), sample_min_max_value)
        save_signals(audio_list, audio_path, SAVE_DIR_GENERATED)


    for audio_path in sample_file_paths:
        file_name = audio_path.split("/")[-1].split(".")[0] + '.wav'
        shutil.copyfile(os.path.join(wav_file_path, file_name), os.path.join(SAVE_DIR_GENERATED, file_name))




