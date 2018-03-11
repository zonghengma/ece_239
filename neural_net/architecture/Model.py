import numpy as np
from neural_net.architecture.base_model import BaseModel
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Conv2D, Dropout, MaxPooling2D, Flatten

class StackedLSTM(BaseModel):
  def __init__(self, hyperparams, archparams):
    """ initialize the StackedLSTM with hyperparams
    architecture: [LSTM]xN - [Dense]xM

    archparams:
      lstm_activation: activation function, default is tanh
      lstm_dropout: dropout rate, default is 0
      lstm_units: number of units in each LSTM layer, default structure is 
                  [32, 32, 32], at least one layer
      dense_units: number of units in each hidden dense layer, default is [1024]
      dense_dropout: dense layer dropout rate, default is 0
      input_dim: input dimensions, default is 288x1000x25
    """
    self.lstm_act = archparams.get('lstm_activation', 'tanh')
    self.lstm_dropout = archparams.get('lstm_dropout', 0)
    self.lstm_units = archparams.get('lstm_units', [32, 32, 32])
    self.dense_units = archparams.get('dense_units', [1024])
    self.dense_dropout = archparams.get('dense_dropout', 0)
    self.input_dim = archparams.get('input_dim', (288, 1000, 25))
    self.timestep = self.input_dim[1]
    self.channels = self.input_dim[2]
    model = self.construct_model()

    super().__init__(model, 'StackedLSTM', hyperparams, archparams)

  def construct_model(self):
    model = Sequential()

    # at least one LSTM layer
    model.add(LSTM(self.lstm_units[0], 
                          return_sequences=len(self.lstm_units) > 1,
                          activation=self.lstm_act,
                          dropout=self.lstm_dropout,
                          input_shape=(self.timestep, self.channels)))
    
    # LSTM layers in the middle
    for i in range(1, len(self.lstm_units)-1):
      model.add(LSTM(self.lstm_units[i], 
                            return_sequences=True,
                            activation=self.lstm_act,
                            dropout=self.lstm_dropout))
    
    # last LSTM layer
    if len(self.lstm_units) > 1:
      model.add(LSTM(self.lstm_units[-1],
                            activation=self.lstm_act,
                            dropout=self.lstm_dropout))

    # dense layer for classification
    for i in range(len(self.dense_units)):
      model.add(Dense(self.dense_units[i], activation='relu'))
      model.add(Dropout(self.dense_dropout))
    model.add(Dense(4, activation='softmax'))
    return model

class CNNLSTM(BaseModel):
  def __init__(self, hyperparams, archparams):
    """
    architecture: [conv-maxpool]xN - [flatten] - [lstm]xM - [Dense]xK
    archparams:
      input_dim: input dimensions(rows, cols, timestep), default is (4, 6, 1)
      conv_units: number of units in each conv layer, default is [64, 64, 64]
      conv_activation: convolution activation, default is relu
      kernel_size: size of conv kernel, default is (3, 3)
      pool_size: size of pooling layer, default is (2, 2)
      conv_dropout: convolution layer dropout rate, ranges from [0, 1], default no dropout

      lstm_units: number of units in each lstm layer, default is [32, 32, 32]
      lstm_activation: lstm activation, default is tanh
      lstm_dropout: lstm dropout, ranges from [0, 1], default is no dropout

      dense_units: number of units in each hidden dense layer, default is [1024]
      dense_dropout: dense layer dropout rate, default is 0
    """
    self.input_dim = archparams.get('input_dim', (4, 6, 1))
    self.conv_units = archparams.get('conv_units', [64, 64, 64])
    self.conv_act = archparams.get('conv_activation', 'relu')
    self.kernel_size = archparams.get('kernel_size', (3, 3))
    self.pool_size = archparams.get('pool_size', (2, 2))
    self.conv_dropout = archparams.get('conv_dropout', 0)

    self.lstm_units = archparams.get('lstm_units', [32, 32, 32])
    self.lstm_act = archparams.get('lstm_activation', 'tanh')
    self.lstm_dropout = archparams.get('lstm_dropout', 0)

    self.dense_units = archparams.get('dense_units', [1024])
    self.dense_dropout = archparams.get('dense_dropout', 0)

    model = self.construct_model()
    super().__init__(model, 'CNNLSTM', hyperparams, archparams)

  def construct_model(self):
    model = Sequential()
    # define CNN model
    for i in range(len(self.conv_units)):
      if i == 0:
        model.add(TimeDistributed(Conv2D(self.conv_units[i],
                                        kernel_size=self.kernel_size,
                                        activation=self.conv_act,
                                        padding='same',
                                        input_shape=self.input_dim)))
      else:
        model.add(TimeDistributed(Conv2D(self.conv_units[i],
                                        kernel_size=self.kernel_size,
                                        activation=self.conv_act,
                                        padding='same')))
      model.add(TimeDistributed(MaxPooling2D(pool_size=self.pool_size)))
    model.add(TimeDistributed(Flatten()))
    # define LSTM model
    for i in range(len(self.lstm_units)-1):
      model.add(LSTM(self.lstm_units[i], 
                    return_sequences=True,
                    activation=self.lstm_act,
                    dropout=self.lstm_dropout))
    model.add(LSTM(self.lstm_units[-1],
                  activation=self.lstm_act,
                  dropout=self.lstm_dropout))
    # add hidden dense layers
    for i in range(len(self.dense_units)):
      model.add(Dense(self.dense_units[i], activation='relu'))
      model.add(Dropout(self.dense_dropout))
    model.add(Dense(4, activation="softmax"))
    return model
