from neural_net.data_access.read_project_data import DataReader, DataLoader
from neural_net.architecture.fc_nets import ThreeLayerFcNet
from neural_net.architecture.Model import StackedLSTM, CNNLSTM, TemporalCNNLSTM
from neural_net.data_process.data_processor import DataProcessor
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

# dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_freq_all.npz")
# dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_freq_subj1.npz")
dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_freq_subj1-8.npz")
# dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_all.npz")
# dl = DataLoader("neural_net\\data_access\\processed_datasets\\nan_winsor_normalized_subj1.npz")
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


# Set architecture-specific parameters.
archparams = {}
archparams['input_dim'] = X_train.shape
archparams['channels'] = 3

archparams['kernel_regularizer'] = 0.001
archparams['conv_units'] = [32, 32]
archparams['kernel_size'] = (3, 3)
archparams['pool_size'] = (1, 1)

archparams['lstm_units'] = [8, 8, 8, 8, 8]

archparams['dense_units'] = [1024, 512]
archparams['dense_dropout'] = 0.4

# archparams['input_dim'] = X_train.shape
# archparams['lstm_activation'] = 'tanh'
# archparams['lstm_dropout'] = 0.4
# archparams['lstm_units'] = [32, 32, 32]
# archparams['dense_units'] = [1024, 512]
# archparams['dense_dropout'] = 0.4
# archparams['kernel_regularizer'] = 0.001

# Set regular hyperparameters.
hyperparams = {}
hyperparams['optimizer'] = 'adam'
hyperparams['learning_rate'] = 1e-2
hyperparams['lr_decay'] = 0.001
hyperparams['loss_function'] = 'categorical_crossentropy'
hyperparams['batch_size'] = 16
hyperparams['epochs'] = 100
hyperparams['verbose'] = 1

# Create the network.
net = CNNLSTM(hyperparams, archparams)
net.train(data)
# net = StackedLSTM(hyperparams, archparams)
# net.train(data)
# net = TemporalCNNLSTM(hyperparams, archparams)
# net.train(data)

# Process the data.
data_processor = DataProcessor(net)
data_processor.save_data_csv(save_history=True)
