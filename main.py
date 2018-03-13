from neural_net.data_access.read_project_data import DataReader, DataLoader
from neural_net.architecture.fc_nets import ThreeLayerFcNet
from neural_net.architecture.Model import StackedLSTM, CNNLSTM
from neural_net.data_process.data_processor import DataProcessor
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

# Get mnist data.
#(X_main_raw, y_main_raw), (X_test_raw, y_test_raw) = mnist.load_data()
#shuffle_train_idx = np.random.permutation(X_main_raw.shape[0])
#shuffle_val_idx = np.random.permutation(X_test_raw.shape[0])

# Reshape the data into proper format.
#num_pixels = X_main_raw.shape[1] * X_main_raw.shape[2]
#X_main = X_main_raw.reshape(X_main_raw.shape[0], num_pixels).astype('float32')
#X_test = X_test_raw.reshape(X_test_raw.shape[0], num_pixels).astype('float32')
#y_main = np_utils.to_categorical(y_main_raw)
#y_test = np_utils.to_categorical(y_test_raw)
#<<<<<<< HEAD
#dl = DataLoader("neural_net/data_access/processed_datasets/img_sample.npz")
#=======
#dl = DataLoader("neural_net\\data_access\\processed_datasets\\norm_sample.npz")
#dl = DataLoader("neural_net\\data_access\\processed_datasets\\sample.npz")
dl = DataLoader("neural_net\\data_access\\processed_datasets\\img_sample.npz")
#dl = DataLoader("neural_net\\data_access\\processed_datasets\\img_split_sample2.npz")
#dl = DataLoader("neural_net\\data_access\\processed_datasets\\augment_sample.npz")
#dl = DataLoader("neural_net\\data_access\\processed_datasets\\no_cue_sample.npz")
#>>>>>>> 02f24cb2aa27da9dffc2fc13176bfd0c1a7da21f
#num_class = y_test.shape[1]
#num_class = y_test.shape[1]
X_train, y_train, X_val, y_val, X_test, y_test = dl.load()
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)
X_val = X_val.reshape(X_val.shape[0],X_val.shape[1], X_val.shape[2], X_val.shape[3], 1)
#data = {
#  'X_train': X_main[shuffle_train_idx],
#  'y_train': y_main[shuffle_train_idx],
#  'X_val': X_main[shuffle_val_idx],
#  'y_val': y_main[shuffle_val_idx]
#}
data = {
  'X_train': X_train,
  'y_train': y_train,
  'X_val': X_val,
  'y_val': y_val
}


# Set architecture-specific parameters.
archparams = {}
archparams['hidden_units'] = [50, 100, 10]
archparams['input_dim'] = X_train.shape
#<<<<<<< HEAD
#archparams['dense_units'] = [1024, 1024]
archparams['dense_units'] = [512]
#archparams['dense_dropout'] = 0.1
archparams['channels'] = 1
#archparams['channels'] = 40
#=======
#archparams['dense_units'] = [1024]
#archparams['dense_dropout'] = 0.4
#>>>>>>> 568e201b1e6e03305ac2517b48ecc3c84b5a4edc
#archparams['lstm_dropout'] = 0.4
archparams['lstm_units'] = [128, 128, 128]
#archparams['kernel_regularizer'] = 0.001
archparams['kernel_size'] = (2,2)
archparams['conv_units'] = [128, 128]
print(X_train.shape)
#archparams['dummy_val'] = 10230

# Set regular hyperparameters.
hyperparams = {}
hyperparams['optimizer'] = 'adam'
hyperparams['learning_rate'] = 1e-3
#hyperparams['learning_rate'] = 1e-4
#hyperparams['learning_rate'] = 3e-5
hyperparams['lr_decay'] = 0.001
hyperparams['loss_function'] = 'categorical_crossentropy'
hyperparams['batch_size'] = 128
#hyperparams['batch_size'] = 32
hyperparams['epochs'] = 30
hyperparams['verbose'] = 2

# Create the network.
#fc_net = ThreeLayerFcNet(hyperparams, archparams)
#fc_net.train(data)
#net = StackedLSTM(hyperparams, archparams)
#net.train(data)
cnn = CNNLSTM(hyperparams, archparams)
cnn.train(data)

# Process the data.
#data_processor = DataProcessor(fc_net)
data_processor = DataProcessor(cnn)
#data_processor = DataProcessor(net)
data_processor.save_data_csv(save_history=True)
