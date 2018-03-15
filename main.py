from neural_net.data_access.read_project_data import DataReader, DataLoader
from neural_net.architecture.fc_nets import ThreeLayerFcNet
from neural_net.architecture.Model import StackedLSTM, CNNLSTM, TemporalCNNLSTM
from neural_net.data_process.data_processor import DataProcessor
from neural_net.architecture.autoencoder import AutoEncoder
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

# dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_freq_all.npz")
# dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_freq_subj1.npz")
# dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_freq_subj1-8.npz")
# dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_all.npz")
#dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_subj1.npz")
#dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_freq_subj1.npz")
dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_freq_all.npz")
#dl = DataLoader("neural_net\\data_access\\processed_datasets\\winsor_all_img_split.npz")
#dl = DataLoader("neural_net\\data_access\\processed_datasets\\winsor_all.npz")
#dl = DataLoader("neural_net\\data_access\\processed_datasets\\sample_winsor_1.npz")
#dl = DataLoader("neural_net\\data_access\\processed_datasets\\winsor_img_sample_split_4x6.npz")
#dl = DataLoader("neural_net\\data_access\\processed_datasets\\winsor_all.npz")
# dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_subj2.npz")
# dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_subj1-8.npz")

X_train, y_train, X_val, y_val, X_test, y_test = dl.load()

#X_train = np.absolute(np.fft.fft(X_train, axis=1))[:,:500,:]
#X_train_mean = np.mean(X_train, axis = 1)
#X_train_std = np.std(X_train, axis = 1)
#X_train_ = np.zeros(X_train.shape)
#for i in range(X_train.shape[0]):
#  for j in range(X_train.shape[2]):
#    X_train_[i, :, j] = (X_train[i, :, j] - X_train_mean[i, j]) / X_train_std[i,j]
#X_train = X_train_
#
#X_val = np.absolute(np.fft.fft(X_val, axis=1))[:,:500,:]
#
#X_val_mean = np.mean(X_val, axis = 1)
#X_val_std = np.std(X_val, axis = 1)
#X_val_ = np.zeros(X_val.shape)
#for i in range(X_val.shape[0]):
#  for j in range(X_val.shape[2]):
#    X_val_[i, :, j] = (X_val[i, :, j] - X_val_mean[i, j]) / X_val_std[i,j]
#X_val = X_val_

#ae = AutoEncoder()
#ae.train(X_train, X_val)
#X_train = ae.encode(X_train)
#X_val = ae.encode(X_val)

print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
data = {
  'X_train': X_train,
  'y_train': y_train,
  'X_val': X_val,
  'y_val': y_val
}

""" HYPERPARAMS """
hyperparams = {}
hyperparams['optimizer'] = 'adam'
#hyperparams['learning_rate'] = 1e-2
hyperparams['learning_rate'] = 5e-5
hyperparams['lr_decay'] = 0
hyperparams['loss_function'] = 'categorical_crossentropy'
hyperparams['batch_size'] = 32
hyperparams['epochs'] = 20
hyperparams['verbose'] = 1


""" ARCHPARAMS """
archparams = {}
###############
# StackedLSTM #
###############
#archparams['lstm_activation'] = 'tanh'
#archparams['lstm_dropout'] = 0.002
#archparams['lstm_units'] = [128, 128]#, 64]
#archparams['dense_units'] = [1024]
#archparams['dense_dropout'] = 0.002
#archparams['kernel_regularizer'] = 0.001
#archparams['initializer'] = 'he_normal'
#archparams['input_dim'] = X_train.shape
#
#net = StackedLSTM(hyperparams, archparams)
#net.train(data)

###########
# CNNLSTM #
###########
archparams['input_dim'] = X_train.shape
archparams['channels'] = 3
archparams['kernel_regularizer'] = 0.001
archparams['initializer'] = 'he_normal'
archparams['conv_units'] = [128, 128]
archparams['conv_activation'] = 'relu'
archparams['kernel_size'] = (2, 2)
archparams['pool_size'] = (1, 1)
archparams['conv_dropout'] = [0.05, 0.05]
archparams['lstm_units'] = [64, 64]
archparams['lstm_activation'] = 'tanh'
archparams['lstm_dropout'] = 0
archparams['dense_units'] = [1024]
archparams['dense_dropout'] = 0
archparams['dilation_rate'] = (2,2)

net = CNNLSTM(hyperparams, archparams)
net.train(data)

###################
# TemporalCNNLSTM #
###################
#archparams['input_dim'] = X_train.shape
#archparams['kernel_regularizer'] = 0.001
#archparams['initializer'] = 'he_normal'
##archparams['conv_units'] = [64, 64, 128, 128]
#archparams['conv_units'] = [128, 128]
#archparams['conv_activation'] = 'relu'
#archparams['kernel_size'] = 2
#archparams['strides'] = 1
#archparams['pool_size'] = 1
#archparams['conv_dropout'] = 0
#archparams['lstm_units'] = [64, 64]
#archparams['lstm_activation'] = 'tanh'
#archparams['lstm_dropout'] = 0.2
#archparams['dense_units'] = [1024]
#archparams['dense_dropout'] = 0
#archparams['dilation_rate'] = 2
#
#net = TemporalCNNLSTM(hyperparams, archparams)
#net.train(data)

# Process the data.
data_processor = DataProcessor(net)
data_processor.save_data_csv(save_history=True)
