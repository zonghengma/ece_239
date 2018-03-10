import numpy as np
from base_model import BaseModel
from keras.models import Sequential
from keras.layers import LSTM, Dense

class StackedLSTM(object):
  def __init__(self, hyperparams):
    self.activation = hyperparams.get('activation', 'tanh')
    self.dropout = hyperparams.get('dropout', 0)
    self.hidden_dims = hyperparams.get('hidden_dims', [32, 32, 32])
    self.timestep = hyperparams.get('timestep', 1000)
    self.channels = hyperparams.get('channels', 25)
    self.model = self.construct_model()

  def construct_model(self):
    model = Sequential()
    model.add(LSTM(self.hidden_dims[0], 
                          return_sequence=len(self.hidden_dims) > 1,
                          activation=self.activation,
                          dropout=self.dropout,
                          input_shape=(self.timestep, self.channels)))
    for i in range(1, len(self.hidden_dims)-1):
      model.add(LSTM(self.hidden_dims[i], 
                            return_sequence=True,
                            activation=self.activation,
                            dropout=self.dropout))
    if len(self.hidden_dims) > 1:
      model.add(LSTM(self.hidden_dims[-1],
                            activation=self.activation,
                            dropout=self.dropout))
    model.add(Dense(4, activation='softmax'))
    return model

  def get_model(self):
    return self.model

class CNNLSTM(object):
  def __init__(self, ):
    pass

  def get_model(self):
    pass




