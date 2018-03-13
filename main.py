from neural_net.data_access.read_project_data import DataReader, DataLoader
from neural_net.architecture.fc_nets import ThreeLayerFcNet
from neural_net.architecture.Model import StackedLSTM, CNNLSTM, TemporalCNNLSTM
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
dl = DataLoader("neural_net/data_access/processed_datasets/all.npz")
#=======
#dl = DataLoader("neural_net\\data_access\\processed_datasets\\norm_sample.npz")
#dl = DataLoader("neural_net\\data_access\\processed_datasets\\sample.npz")
#dl = DataLoader("neural_net\\data_access\\processed_datasets\\img_sample.npz")
#>>>>>>> 02f24cb2aa27da9dffc2fc13176bfd0c1a7da21f
#num_class = y_test.shape[1]
#num_class = y_test.shape[1]
X_train, y_train, X_val, y_val, X_test, y_test = dl.load()

print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
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
#archparams['hidden_units'] = [50, 100, 10]
archparams['input_dim'] = X_train.shape
archparams['dense_units'] = [1024, 1024]
archparams['dense_dropout'] = 0.4
#archparams['lstm_dropout'] = 0.4
#archparams['lstm_units'] = [128, 128]
#archparams['kernel_regularizer'] = 0.001
#archparams['conv_units'] = [128, 128]
print(X_train.shape)
#archparams['dummy_val'] = 10230

# Set regular hyperparameters.
hyperparams = {}
hyperparams['optimizer'] = 'adam'
hyperparams['learning_rate'] = 1e-3
hyperparams['lr_decay'] = 0.01
hyperparams['loss_function'] = 'categorical_crossentropy'
hyperparams['batch_size'] = 16
hyperparams['epochs'] = 10
hyperparams['verbose'] = 2

# Create the network.
#fc_net = ThreeLayerFcNet(hyperparams, archparams)
#fc_net.train(data)
#net = StackedLSTM(hyperparams, archparams)
#net.train(data)
cnn1d = TemporalCNNLSTM(hyperparams, archparams)
cnn1d.train(data)

# Process the data.
#data_processor = DataProcessor(fc_net)
data_processor = DataProcessor(cnn1d)
#data_processor = DataProcessor(net)
data_processor.save_data_csv(save_history=True)
