import numpy as np
from neural_net.architecture.base_model import BaseModel
from keras.models import Sequential
from keras import regularizers
from keras.layers import LSTM, Dense, TimeDistributed, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Conv3D, MaxPooling3D
from keras.layers import BatchNormalization
import pdb

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
      kernel_regularizer: penalty rate of l2 regularization
      initializer: initialization
      input_dim: input dimensions, default is 288x1000x25
    """
    self.lstm_act = archparams.get('lstm_activation', 'tanh')
    self.lstm_dropout = archparams.get('lstm_dropout', 0)
    self.lstm_units = archparams.get('lstm_units', [32, 32, 32])
    self.dense_units = archparams.get('dense_units', [1024])
    self.dense_dropout = archparams.get('dense_dropout', 0)
    self.kernel_regularizer = archparams.get('kernel_regularizer', 0.001)
    self.initializer = archparams.get('initializer', 'he_normal')
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
                          kernel_initializer=self.initializer,
                          input_shape=(self.timestep, self.channels)))
    
    # LSTM layers in the middle
    for i in range(1, len(self.lstm_units)-1):
      model.add(LSTM(self.lstm_units[i], 
                            return_sequences=True,
                            activation=self.lstm_act,
                            kernel_initializer=self.initializer,
                            dropout=self.lstm_dropout))
    
    # last LSTM layer
    if len(self.lstm_units) > 1:
      model.add(LSTM(self.lstm_units[-1],
                            activation=self.lstm_act,
                            kernel_initializer=self.initializer,
                            dropout=self.lstm_dropout))

    # dense layer for classification
    for i in range(len(self.dense_units)):
      model.add(Dense(self.dense_units[i], 
                      activation='relu',
                      kernel_initializer=self.initializer,
                      kernel_regularizer=regularizers.l2(self.kernel_regularizer)))
      model.add(Dropout(self.dense_dropout))
      model.add(BatchNormalization())
    model.add(Dense(4, kernel_initializer=self.initializer, 
                    activation='softmax'))
    return model

class CNNLSTM(BaseModel):
  def __init__(self, hyperparams, archparams):
    """
    architecture: [conv-maxpool]xN - [flatten] - [lstm]xM - [Dense]xK
    archparams:
      input_dim: input dimensions(rows, cols, timestep), default is (6, 7, 1)
      kernel_regularizer: penalty rate of l2 regularization
      channels: channels of the conv layer input
      initializer: initialization

      conv_units: number of units in each conv layer, default is [64, 64, 64]
      conv_activation: convolution activation, default is relu
      kernel_size: size of conv kernel, default is (3, 3)
      pool_size: size of pooling layer, default is (2, 2)
      pool_strides: size of strides in pooling layer, default is (1, 1)
      conv_dropout: convolution layer dropout rate, ranges from [0, 1], default no dropout
        should be a length-N array, where N is number of conv layers

      lstm_units: number of units in each lstm layer, default is [32, 32, 32]
      lstm_activation: lstm activation, default is tanh
      lstm_dropout: lstm dropout, ranges from [0, 1], default is no dropout

      dense_units: number of units in each hidden dense layer, default is [1024]
      dense_dropout: dense layer dropout rate, default is 0
    """
    self.input_dim = archparams.get('input_dim', (288, 1000, 6, 7, 1))
    self.channels = archparams.get('channels', 1)
    self.input_shape = (self.input_dim[1], self.input_dim[2], self.input_dim[3], self.channels)
    self.kernel_regularizer = archparams.get('kernel_regularizer', 0.001)
    self.initializer = archparams.get('initializer', None)

    self.conv_units = archparams.get('conv_units', [128, 128])
    self.conv_act = archparams.get('conv_activation', 'relu')
    self.kernel_size = archparams.get('kernel_size', (2, 2))
    self.pool_size = archparams.get('pool_size', [(1,1), (1,1)])
    self.pool_strides = archparams.get('pool_strides', [None, None])
    self.dilation_rate = archparams.get('dilation_rate', (1, 1))
    self.conv_dropout = archparams.get('conv_dropout', [0, 0])

    self.lstm_units = archparams.get('lstm_units', [64, 64])
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
                                        dilation_rate=self.dilation_rate,
                                        kernel_initializer=self.initializer,
                                        kernel_regularizer=regularizers.l2(self.kernel_regularizer)),
                                        input_shape=self.input_shape))
        #pdb.set_trace()
      else:
        model.add(TimeDistributed(Conv2D(self.conv_units[i],
                                        kernel_size=self.kernel_size,
                                        activation=self.conv_act,
                                        dilation_rate=self.dilation_rate,
                                        kernel_initializer=self.initializer,
                                        kernel_regularizer=regularizers.l2(self.kernel_regularizer),
                                        padding='same')))
      #pdb.set_trace()
      #model.add(BatchNormalization())
      model.add(Dropout(self.conv_dropout[i]))
      model.add(TimeDistributed(MaxPooling2D(pool_size=self.pool_size[i], strides=self.pool_strides[i])))
      #model.add(BatchNormalization())
    model.add(TimeDistributed(Flatten()))
    # define LSTM model
    for i in range(len(self.lstm_units)-1):
      model.add(LSTM(self.lstm_units[i], 
                    return_sequences=True,
                    activation=self.lstm_act,
                    kernel_regularizer=regularizers.l2(self.kernel_regularizer),
                    kernel_initializer=self.initializer,
                    dropout=self.lstm_dropout))
    model.add(LSTM(self.lstm_units[-1],
                  activation=self.lstm_act,
                  kernel_regularizer=regularizers.l2(self.kernel_regularizer),
                  kernel_initializer=self.initializer,
                  dropout=self.lstm_dropout))
    # add hidden dense layers
    for i in range(len(self.dense_units)):
      model.add(Dense(self.dense_units[i], 
                      activation='relu',
                      kernel_initializer=self.initializer,
                      kernel_regularizer=regularizers.l2(self.kernel_regularizer)))
      model.add(Dropout(self.dense_dropout))
      model.add(BatchNormalization())
    model.add(Dense(4, kernel_initializer=self.initializer,
                    activation="softmax"))
    model.summary()
    return model

class TemporalCNNLSTM(BaseModel):
  def __init__(self, hyperparams, archparams):
    """
    architecture: [Conv1D-maxpool]xN - [flatten] - [lstm]xM - [Dense]xK
    archparams:
      input_dim: input dimensions(rows, cols, timestep), default is (6, 7, 1)
      kernel_regularizer: penalty rate of l2 regularization
      channels: channels of the conv layer input
      initializer: initialization

      conv_units: number of units in each conv layer, default is [64, 64, 64]
      conv_activation: convolution activation, default is relu
      kernel_size: size of conv kernel, default is 3
      strides: size of strides, default is 1
      pool_size: size of pooling layer, default is (2, 2)
      conv_dropout: convolution layer dropout rate, ranges from [0, 1], default no dropout

      lstm_units: number of units in each lstm layer, default is [32, 32, 32]
      lstm_activation: lstm activation, default is tanh
      lstm_dropout: lstm dropout, ranges from [0, 1], default is no dropout

      dense_units: number of units in each hidden dense layer, default is [1024]
      dense_dropout: dense layer dropout rate, default is 0
    """
    self.input_dim = archparams.get('input_dim', (288, 1000, 22))
    self.input_shape = (self.input_dim[1], self.input_dim[2])
    self.kernel_regularizer = archparams.get('kernel_regularizer', 0.001)
    self.initializer = archparams.get('initializer', 'he_normal')

    self.conv_units = archparams.get('conv_units', [64, 64, 128, 128])
    self.conv_act = archparams.get('conv_activation', 'relu')
    self.kernel_size = archparams.get('kernel_size', 3)
    self.strides = archparams.get('strides', 1)
    self.pool_size = archparams.get('pool_size', 2)
    self.dilation_rate = archparams.get('dilation_rate', 1)
    self.conv_dropout = archparams.get('conv_dropout', 0)

    self.lstm_units = archparams.get('lstm_units', [128, 64])
    self.lstm_act = archparams.get('lstm_activation', 'tanh')
    self.lstm_dropout = archparams.get('lstm_dropout', 0)

    self.dense_units = archparams.get('dense_units', [1024])
    self.dense_dropout = archparams.get('dense_dropout', 0)

    model = self.construct_model()
    super().__init__(model, 'TemporalCNNLSTM', hyperparams, archparams)

  def construct_model(self):
    model = Sequential()
    # define CNN model
    for i in range(len(self.conv_units)):
      if i == 0:
        model.add(Conv1D(self.conv_units[i],
                        kernel_size=self.kernel_size,
                        activation=self.conv_act,
                        padding='same',
                        dilation_rate=self.dilation_rate,
                        strides=self.strides,
                        kernel_initializer=self.initializer,
                        kernel_regularizer=regularizers.l2(self.kernel_regularizer),
                        input_shape=self.input_shape))
        #pdb.set_trace()
      else:
        model.add(Conv1D(self.conv_units[i],
                        kernel_size=self.kernel_size,
                        activation=self.conv_act,
                        padding='same',
                        dilation_rate=self.dilation_rate,
                        strides=self.strides,
                        kernel_initializer=self.initializer,
                        kernel_regularizer=regularizers.l2(self.kernel_regularizer)))
      #pdb.set_trace()
      if (i+1)%2 == 0: 
        model.add(MaxPooling1D(pool_size=self.pool_size))

    # define LSTM model
    for i in range(len(self.lstm_units)-1):
      model.add(LSTM(self.lstm_units[i], 
                    return_sequences=True,
                    activation=self.lstm_act,
                    kernel_regularizer=regularizers.l2(self.kernel_regularizer),
                    kernel_initializer=self.initializer,
                    dropout=self.lstm_dropout))
    model.add(LSTM(self.lstm_units[-1],
                  activation=self.lstm_act,
                  kernel_regularizer=regularizers.l2(self.kernel_regularizer),
                  kernel_initializer=self.initializer,
                  dropout=self.lstm_dropout))
    # add hidden dense layers
    for i in range(len(self.dense_units)):
      model.add(Dense(self.dense_units[i], 
                      activation='relu',
                      kernel_initializer=self.initializer,
                      kernel_regularizer=regularizers.l2(self.kernel_regularizer)))
      model.add(Dropout(self.dense_dropout))
      model.add(BatchNormalization())
    #model.add(Flatten())
    model.add(Dense(4, kernel_initializer=self.initializer,
                    activation="softmax"))
    # pdb.set_trace()
    return model

class vanilaCNN2D(BaseModel):
  def __init__(self, hyperparams, archparams):
    """
    architecture: [[Conv2D-Conv2D]-maxpooling]xN - [Dense]xM
    """
    self.input_dim = archparams.get('input_dim', (288*40, 6, 7, 25))
    self.channels = archparams.get('channels', 25)
    self.input_shape = (self.input_dim[1], self.input_dim[2], self.channels)
    self.kernel_regularizer = archparams.get('kernel_regularizer', 0.001)
    self.initializer = archparams.get('initializer', 'he_normal')

    self.conv_units = archparams.get('conv_units', [64, 64, 128, 128])
    self.conv_act = archparams.get('conv_activation', 'relu')
    self.kernel_size = archparams.get('kernel_size', (2,2))
    self.strides = archparams.get('strides', (1,1))
    self.pool_size = archparams.get('pool_size', (2,2))
    self.conv_dropout = archparams.get('conv_dropout', 0)

    self.dense_units = archparams.get('dense_units', [1024])
    self.dense_dropout = archparams.get('dense_dropout', 0)

    model = self.construct_model()
    super().__init__(model, 'vanilaCNN2D', hyperparams, archparams)

  def construct_model(self):
    model = Sequential()
    # define CNN model
    for i in range(len(self.conv_units)):
      if i == 0:
        model.add(Conv2D(self.conv_units[i],
                        kernel_size=self.kernel_size,
                        activation=self.conv_act,
                        padding='same',
                        strides=self.strides,
                        kernel_initializer=self.initializer,
                        kernel_regularizer=regularizers.l2(self.kernel_regularizer),
                        input_shape=self.input_shape))
        #pdb.set_trace()
      else:
        model.add(Conv2D(self.conv_units[i],
                        kernel_size=self.kernel_size,
                        activation=self.conv_act,
                        padding='same',
                        strides=self.strides,
                        kernel_initializer=self.initializer,
                        kernel_regularizer=regularizers.l2(self.kernel_regularizer)))
      #pdb.set_trace()
      if (i+1)%2 == 0: 
        model.add(MaxPooling2D(pool_size=self.pool_size))

    for i in range(len(self.dense_units)):
      model.add(Dense(self.dense_units[i], 
                      activation='relu',
                      kernel_initializer=self.initializer,
                      kernel_regularizer=regularizers.l2(self.kernel_regularizer)))
      model.add(Dropout(self.dense_dropout))
      model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(4, kernel_initializer=self.initializer,
                    activation="softmax"))
    pdb.set_trace()
    return model

class vanilaCNN3D(BaseModel):
  def __init__(self, hyperparams, archparams):
    """
    architecture: [[Conv3D-Conv3D]-maxpooling]xN - [Dense]xM
    """
    self.input_dim = archparams.get('input_dim', (288, 1000, 6, 7, 1))
    self.input_shape = (self.input_dim[1], self.input_dim[2], self.input_dim[3], self.input_dim[4])
    self.kernel_regularizer = archparams.get('kernel_regularizer', 0.001)
    self.initializer = archparams.get('initializer', 'he_normal')

    self.conv_units = archparams.get('conv_units', [64, 64, 128, 128])
    self.conv_act = archparams.get('conv_activation', 'relu')
    self.kernel_size = archparams.get('kernel_size', (2,2,2))
    self.strides = archparams.get('strides', (1,1,1))
    self.pool_size = archparams.get('pool_size', (2,1,1))
    self.conv_dropout = archparams.get('conv_dropout', 0)

    self.dense_units = archparams.get('dense_units', [1024])
    self.dense_dropout = archparams.get('dense_dropout', 0)

    model = self.construct_model()
    super().__init__(model, 'vanilaCNN3D', hyperparams, archparams)

  def construct_model(self):
    model = Sequential()
    # define CNN model
    for i in range(len(self.conv_units)):
      if i == 0:
        model.add(Conv3D(self.conv_units[i],
                        kernel_size=self.kernel_size,
                        activation=self.conv_act,
                        padding='same',
                        strides=self.strides,
                        kernel_initializer=self.initializer,
                        kernel_regularizer=regularizers.l2(self.kernel_regularizer),
                        input_shape=self.input_shape))
        #pdb.set_trace()
      else:
        model.add(Conv3D(self.conv_units[i],
                        kernel_size=self.kernel_size,
                        activation=self.conv_act,
                        padding='same',
                        strides=self.strides,
                        kernel_initializer=self.initializer,
                        kernel_regularizer=regularizers.l2(self.kernel_regularizer)))
      #pdb.set_trace()
      if (i+1)%2 == 0: 
        model.add(MaxPooling3D(pool_size=self.pool_size))

    for i in range(len(self.dense_units)):
      model.add(Dense(self.dense_units[i], 
                      activation='relu',
                      kernel_initializer=self.initializer,
                      kernel_regularizer=regularizers.l2(self.kernel_regularizer)))
      model.add(Dropout(self.dense_dropout))
      model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(4, kernel_initializer=self.initializer,
                    activation="softmax"))
    pdb.set_trace()
    return model
