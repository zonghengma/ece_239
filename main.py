from neural_net.data_access.read_project_data import DataReader, DataLoader
from neural_net.architecture.fc_nets import ThreeLayerFcNet
from neural_net.architecture.Model import StackedLSTM, CNNLSTM, TemporalCNNLSTM
from neural_net.data_process.data_processor import DataProcessor
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

# dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_freq_all.npz")
# dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_freq_subj1.npz")
# dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_freq_subj1-8.npz")
# dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_all.npz")
dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_subj1.npz")
# dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_subj2.npz")
# dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_subj1-8.npz")

X_train, y_train, X_val, y_val, X_test, y_test = dl.load()

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
hyperparams['learning_rate'] = 1e-2
hyperparams['lr_decay'] = 0.001
hyperparams['loss_function'] = 'categorical_crossentropy'
hyperparams['batch_size'] = 16
hyperparams['epochs'] = 100
hyperparams['verbose'] = 1


""" ARCHPARAMS """
archparams = {}
###############
# StackedLSTM #
###############
# archparams['lstm_activation'] = 'tanh'
# archparams['lstm_dropout'] = 0
# archparams['lstm_units'] = [32, 32, 32]
# archparams['dense_units'] = [1024]
# archparams['dense_dropout'] = 0
# archparams['kernel_regularizer'] = 0.001
# archparams['initializer'] = 'he_normal'
# archparams['input_dim'] = X_train.shape

# net = StackedLSTM(hyperparams, archparams)
# net.train(data)

###########
# CNNLSTM #
###########
archparams['input_dim'] = X_train.shape
archparams['channels'] = 1
archparams['kernel_regularizer'] = 0.001
archparams['initializer'] = 'he_normal'
archparams['conv_units'] = [128, 128]
archparams['conv_activation'] = 'relu'
archparams['kernel_size'] = (2, 2)
archparams['pool_size'] = (2, 2)
archparams['conv_dropout'] = 0
archparams['lstm_units'] = [64, 64]
archparams['lstm_activation'] = 'tanh'
archparams['lstm_dropout'] = 0
archparams['dense_units'] = [1024]
archparams['dense_dropout'] = 0

net = CNNLSTM(hyperparams, archparams)
net.train(data)

###################
# TemporalCNNLSTM #
###################
# archparams['input_dim'] = X_train.shape
# archparams['kernel_regularizer'] = 0.001
# archparams['initializer'] = 'he_normal'
# archparams['conv_units'] = [64, 64, 128, 128]
# archparams['conv_activation'] = 'relu'
# archparams['kernel_size'] = 3
# archparams['strides'] = 1
# archparams['pool_size'] = 2
# archparams['conv_dropout'] = 0
# archparams['lstm_units'] = [128, 64]
# archparams['lstm_activation'] = 'tanh'
# archparams['lstm_dropout'] = 0
# archparams['dense_units'] = [1024]
# archparams['dense_dropout'] = 0

# net = TemporalCNNLSTM(hyperparams, archparams)
# net.train(data)

# Process the data.
data_processor = DataProcessor(net)
data_processor.save_data_csv(save_history=True)
