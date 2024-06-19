import torch
import glob
import numpy as np
import os
from autoencoder import Autoencoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim


# def load_spectrogram(spectrogram_path):
#     x_train = []
#     for file_path in glob.glob(os.path.join(spectrogram_path, "*.npy")):
#         spectrogram = np.load(file_path)
#         x_train.append(spectrogram)
#     x_train = np.array(x_train)
#     return x_train


class AudioDataset(Dataset):

    def __init__(self, spectrogram_path):
        super(AudioDataset, self).__init__()
        self.spectrogram_path = spectrogram_path
        self.spectrogram_list =glob.glob(os.path.join(spectrogram_path, "*.npy"))
        
    def __getitem__(self, index):
        spec_npy_path = self.spectrogram_list[index]
        spectrogram = np.load(spec_npy_path)
        mel_sp = torch.tensor(spectrogram).unsqueeze(0)
        return mel_sp

    def __len__(self):
        return len(self.spectrogram_list)

def KLD_loss(mean, log_var):
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return KLD



if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    spectrogram_path = "./dataset/spectrograms"
    batch_size = 8
    epochs = 150
    save_checkpoint_path = "./checkpoint/"
    os.makedirs(save_checkpoint_path, exist_ok=True)


    train_set = AudioDataset(spectrogram_path)
    train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=3,
        )

    ########### prepare model #####################
    model = Autoencoder(input_shape=(1, 256, 64),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2,1)),
        latent_space_dim=128)


    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    

    ########################### traning epochs ##################################
    min_valid_loss = None
    best_epoch = 0
    
    reconstruction_loss_weight = 1000000   ######### hyper-parameter ########
    for epoch in range(epochs):
            tbar = tqdm(train_loader, ncols=80)
            tbar.set_description("Train epoch %s" % epoch)

            running_loss = 0
            
            for i, waveform_spec in enumerate(tbar):
                waveform_spec = waveform_spec.to(device)
                
                output, mean, logvar = model(waveform_spec)
                mseloss = criterion(waveform_spec, output)
                kldloss = KLD_loss(mean, logvar)
                loss = reconstruction_loss_weight * mseloss + kldloss
                loss.backward()

                # Updating TQDM
                running_loss += loss.item()
                if i % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {i + 1:5d}] ----------loss: {running_loss / 100:.3f} -------reconstruction_loss: {mseloss:.3f} ---------kldloss: {kldloss:.3f}')                  
                    running_loss = 0.0
                
                
                # Updating Loss & Updating Weights
                optimizer.step()
                optimizer.zero_grad()
                tbar.set_postfix(loss=loss.item())
                

            torch.save(model.state_dict(), os.path.join(save_checkpoint_path, f"{epoch}.pt"))
    
    


