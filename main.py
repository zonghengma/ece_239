from neural_net.data_access.read_project_data import DataReader, DataLoader
from neural_net.architecture.fc_nets import ThreeLayerFcNet
from neural_net.architecture.Model import StackedLSTM, CNNLSTM, TemporalCNNLSTM, vanilaCNN2D
from neural_net.data_process.data_processor import DataProcessor
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

# dl = DataLoader("neural_net/data_access/processed_datasets/nan_freq_normalizedbyfreq_all.npz")
# dl = DataLoader("neural_net/data_access/processed_datasets/nan_winsor_normalized_freq_1split_all.npz")
# dl = DataLoader("neural_net/data_access/processed_datasets/nan_winsor_normalized_freq_all.npz")
dl = DataLoader("neural_net/data_access/processed_datasets/new_preprocess_clean_image_nx8x6x7x3.npz")
# dl = DataLoader("neural_net/data_access/processed_datasets/nan_winsor_normalized_freq_subj1.npz")
# dl = DataLoader("neural_net/data_access/processed_datasets/nan_winsor_normalized_freq_subj1-8.npz")
# dl = DataLoader("neural_net/data_access/processed_datasets/nan_winsor_normalized_all.npz")
# dl = DataLoader("neural_net/data_access/processed_datasets/nan_winsor_normalized_img_sub1-5,7-8.npz")
# dl = DataLoader("neural_net/data_access/processed_datasets/nan_winsor_normalized_img_subj1.npz")
# dl = DataLoader("neural_net/data_access/processed_datasets/nan_winsor_normalized_img8x125_subj1_.npz")
# dl = DataLoader("neural_net/data_access/processed_datasets/nan_winsor_trial_channel_normalize_channel_cnnlstm_all.npz")
# dl = DataLoader("neural_net/data_access/processed_datasets/nan_winsor_trial_channel_normalize_channel_normalformat_all.npz")
# dl = DataLoader("neural_net/data_access/processed_datasets/nan_winsor_trial_channel_normalize_channel_normalformat_subj1.npz")
# dl = DataLoader("neural_net/data_access/processed_datasets/nan_winsor_normalized_subj1.npz")
# dl = DataLoader("neural_net/data_access/processed_datasets/nan_winsor_normalized_subj2.npz")
# dl = DataLoader("neural_net/data_access/processed_datasets/nan_winsor_normalized_subj1-8.npz")

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
hyperparams['learning_rate'] = 5e-5
hyperparams['lr_decay'] = 0.001
hyperparams['loss_function'] = 'categorical_crossentropy'
hyperparams['batch_size'] = 32
hyperparams['epochs'] = 300
hyperparams['verbose'] = 2


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
archparams['channels'] = 3
archparams['kernel_regularizer'] = 0.0001
archparams['initializer'] = 'glorot_uniform'
archparams['conv_units'] = [32, 32, 64, 64]
archparams['conv_activation'] = 'relu'
archparams['kernel_size'] = (2, 2)
archparams['pool_size'] = (1, 2, 1, 1)
archparams['pool_strides'] = archparams['pool_size']
archparams['dilation_rate'] = 1
archparams['conv_dropout'] = [0, 0, 0, 0]
archparams['lstm_units'] = [128]
archparams['lstm_activation'] = 'tanh'
archparams['lstm_dropout'] = 0
archparams['dense_units'] = [512]
archparams['dense_dropout'] = 0.5

net = CNNLSTM(hyperparams, archparams)
net.train(data)

data_processor = DataProcessor(net)
data_processor.save_data_csv(save_history=True)

###################
# TemporalCNNLSTM #
###################
# archparams['input_dim'] = X_train.shape
# archparams['kernel_regularizer'] = 0.00001
# archparams['initializer'] = 'he_normal'
# archparams['conv_units'] = [32, 32, 32, 32, 64, 64]
# archparams['conv_activation'] = 'relu'
# archparams['kernel_size'] = 10
# archparams['strides'] = 1
# archparams['pool_size'] = 5
# archparams['conv_dropout'] = 0
# archparams['lstm_units'] = [128]
# archparams['lstm_activation'] = 'tanh'
# archparams['lstm_dropout'] = 0
# archparams['dense_units'] = [512]
# archparams['dense_dropout'] = 0.4

# net = TemporalCNNLSTM(hyperparams, archparams)
# net.train(data)

###############
# TemporalCNN #
###############
# archparams['input_dim'] = X_train.shape
# archparams['kernel_regularizer'] = 0.00001
# archparams['initializer'] = 'he_normal'
# archparams['conv_units'] = [32, 32, 32, 32, 64, 64]
# archparams['conv_activation'] = 'relu'
# archparams['kernel_size'] = 10
# archparams['strides'] = 1
# archparams['pool_size'] = 5
# archparams['conv_dropout'] = 0
# archparams['lstm_units'] = [128]
# archparams['lstm_activation'] = 'tanh'
# archparams['lstm_dropout'] = 0
# archparams['dense_units'] = [512]
# archparams['dense_dropout'] = 0.4

# net = TemporalCNN(hyperparams, archparams)
# net.train(data)

###############
# vanilaCNN2D #
###############
# archparams['input_dim'] = X_train.shape
# archparams['channels'] = 3
# archparams['kernel_regularizer'] = 0.001
# archparams['initializer'] = 'he_normal'
# archparams['conv_units'] = [8, 8]
# archparams['conv_activation'] = 'relu'
# archparams['kernel_size'] = (2,2)
# archparams['strides'] = (1,1)
# archparams['pool_size'] = (1,1)
# archparams['conv_dropout'] = 0
# archparams['dense_units'] = [16, 16]
# archparams['dense_dropout'] = 0.5

# net = vanilaCNN2D(hyperparams, archparams)
# net.train(data)


# Process the data.
# data_processor = DataProcessor(net)
# data_processor.save_data_csv(save_history=True)
