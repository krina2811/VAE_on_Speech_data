# Variational AutoEncoder for Speech Data

The main use of autoencoder is dimensionality reduction. The purpose of this model is speech feature extraction. 


Download the dataset from drive......
https://drive.google.com/file/d/1gIpUdKDWof0T1ixQSs1BFJ77wqN-VFch/view?usp=drive_link


git first step, convert the (.wav) data into the spectrogram format.
<br>
python preprocess.py

Next step, during training encoder is used to convert the speech data into latent space or lower dim. The decoder take latent space as input and convert it to original shape. The loss is calculated between decoder output and original input. Model try to reduce the loss and generate the same data like input data. So autoencoder is basically a generative model.
<br>
python trin.py

------------- hyperparameters -------------
<br>
batch_size = 8
<br>
epochs = 150
<br>
input_shape=(1, 256, 64)
<br>
latent_space_dim=128
<br>


 
