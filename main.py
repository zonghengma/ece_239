from neural_net.data_access.read_project_data import DataReader
from neural_net.architecture.fc_nets import ThreeLayerFcNet
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

# Get mnist data.
(X_main_raw, y_main_raw), (X_test_raw, y_test_raw) = mnist.load_data()
shuffle_train_idx = np.random.permutation(X_main_raw.shape[0])
shuffle_val_idx = np.random.permutation(X_test_raw.shape[0])

# Reshape the data into proper format.
num_pixels = X_main_raw.shape[1] * X_main_raw.shape[2]
X_main = X_main_raw.reshape(X_main_raw.shape[0], num_pixels).astype('float32')
X_test = X_test_raw.reshape(X_test_raw.shape[0], num_pixels).astype('float32')
y_main = np_utils.to_categorical(y_main_raw)
y_test = np_utils.to_categorical(y_test_raw)
num_class = y_test.shape[1]
data = {
  'X_train': X_main[shuffle_train_idx],
  'y_train': y_main[shuffle_train_idx],
  'X_val': X_main[shuffle_val_idx],
  'y_val': y_main[shuffle_val_idx]
}

# Set architecture-specific parameters.
archparams = {}
archparams['hidden_units'] = [50, 100, 10]
archparams['input_dim'] = 784

# Set regular hyperparameters.
hyperparams = {}
hyperparams['optimizer'] = 'adam'
hyperparams['learning_rate'] = 1e-3
hyperparams['lr_decay'] = 0.001
hyperparams['loss_function'] = 'categorical_crossentropy'
hyperparams['batch_size'] = 64
hyperparams['epochs'] = 10
hyperparams['verbose'] = 0

# Create the network.
fc_net = ThreeLayerFcNet(hyperparams, archparams)
fc_net.train(data)

# Process the data.
data_processor = DataProcessor(fc_net)
data_processor.save_data()
