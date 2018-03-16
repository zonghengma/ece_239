from neural_net.data_access.read_project_data import DataReader, DataLoader
from neural_net.architecture.fc_nets import ThreeLayerFcNet
from neural_net.architecture.Model import StackedLSTM, CNNLSTM, TemporalCNNLSTM, vanilaCNN2D
from neural_net.data_process.data_processor import DataProcessor
from neural_net.architecture.autoencoder import AutoEncoder
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import csv


data_file_path = "neural_net\\data_access\\processed_datasets\\new_preprocess_clean_image_nx8x6x7x3.npz"
config_file_path = "config.csv"

dl = DataLoader(data_file_path)
X_train, y_train, X_val, y_val, X_test, y_test = dl.load()

print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
data = {
  'X_train': X_train,
  'y_train': y_train,
  'X_val': X_val,
  'y_val': y_val,
  'X_test': X_test,
  'y_test': y_test
}

csv_file = open(config_file_path, 'r', newline='')
csv_reader = csv.DictReader(csv_file)

for row in csv_reader:
  hyperparams = {}
  hyperparams['optimizer'] = 'adam'
  hyperparams['learning_rate'] = float(row['learning_rate'])
  hyperparams['lr_decay']      = float(row['lr_decay'])
  hyperparams['loss_function'] = 'categorical_crossentropy'
  hyperparams['batch_size']    = int(row['batch_size'])
  hyperparams['epochs']        = 300
  hyperparams['verbose']       = 1
  
  
  archparams = {}
  archparams['input_dim'] = X_train.shape
  archparams['channels'] = 3
  archparams['kernel_regularizer'] = 0.0001
  archparams['initializer']        = 'glorot_uniform'       
  archparams['conv_units']         = eval(row['conv_units'])
  archparams['conv_activation']    = 'relu'
  archparams['kernel_size']        = eval(row['kernel_size'])
  archparams['pool_size']          = eval(row['pool_size'])
  archparams['pool_strides']       = eval(row['pool_strides'])
  archparams['dilation_rate']      = int(row['dilation_rate'])
  archparams['conv_dropout']       = eval(row['conv_dropout']) 
  archparams['lstm_units']         = eval(row['lstm_units']) 
  archparams['lstm_activation']    = 'tanh'   
  archparams['lstm_dropout']       = float(row['lstm_dropout'])
  archparams['dense_units']        = eval(row['dense_units'])
  archparams['dense_dropout']      = float(row['dense_dropout']) 
  
  net = CNNLSTM(hyperparams, archparams)
  net.train(data)
  
  data_processor = DataProcessor(net)
  data_processor.save_data_csv(save_history=True)
  
csv_file.close()
