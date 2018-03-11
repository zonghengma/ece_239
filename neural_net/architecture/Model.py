import numpy as np
from neural_net.architecture.base_model import BaseModel
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Conv2D

class StackedLSTM(BaseModel):
  def __init__(self, hyperparams, archparams):
    """ initialize the StackedLSTM with hyperparams
    architecture: [LSTM]xN - Dense

    archparams:
      activation: activation function, default is tanh
      dropout: dropout rate, default is 0
      hidden_units: number of units in each LSTM layer, default structure is 
                  [32, 32, 32], at least one layer
      input_dim: input dimensions, default is 288x1000x25
    """
    self.activation = archparams.get('activation', 'tanh')
    self.dropout = archparams.get('dropout', 0)
    self.hidden_units = archparams.get('hidden_units', [32, 32, 32])
    self.input_dim = archparams.get('input_dim', (288, 1000, 25))
    self.timestep = self.input_dim(1)
    self.channels = self.input_dim(2)
    model = self.construct_model()

    super().__init__(model, 'StackedLSTM', hyperparams, archparams)

  def construct_model(self):
    model = Sequential()

    # at least one LSTM layer
    model.add(LSTM(self.hidden_units[0], 
                          return_sequence=len(self.hidden_units) > 1,
                          activation=self.activation,
                          dropout=self.dropout,
                          input_shape=(self.timestep, self.channels)))
    
    # LSTM layers in the middle
    for i in range(1, len(self.hidden_units)-1):
      model.add(LSTM(self.hidden_units[i], 
                            return_sequence=True,
                            activation=self.activation,
                            dropout=self.dropout))
    
    # last LSTM layer
    if len(self.hidden_units) > 1:
      model.add(LSTM(self.hidden_units[-1],
                            activation=self.activation,
                            dropout=self.dropout))

    # dense layer for classification
    model.add(Dense(4, activation='softmax'))
    return model

class CNNLSTM(BaseModel):
  def __init__(self, hyperparams, archparams):
    """
    architecture: [conv]xN - [lstm]xM - Dense
    
    """
    self.input_dim = archparams.get('input_dim', (5, 5, 1))
    self.conv_units = archparams.get('conv_hidden_units', [64, 64, 64])
    self.conv_act = archparams.get('conv_activation', 'relu')
    self.kernel_size = archparams.get('kernel_size', (3, 3))
    self.pool_size = archparams.get('pool_size', (2, 2))
    self.conv_dropout = archparams.get('conv_dropout', 0)

    self.lstm_units = archparams.get('lstm_hidden_units', [32, 32, 32])
    self.lstm_act = archparams.get('lstm_activation', 'tanh')
    self.lstm_dropout = archparams.get('lstm_dropout', 0)
    model = self.construct_model()
    super().__init__(model, 'CNNLSTM', hyperparams, archparams)

  def construct_model(self):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(self.conv_units[0],
                                    kernel_size=self.kernel_size,
                                    activation=self.conv_act,
                                    padding='same',
                                    input_shape=self.input_dim)))