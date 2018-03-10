import numpy as np
from base_model import BaseModel
from keras.models import Sequential
from keras.layers import LSTM, Dense

class StackedLSTM(BaseModel):
  def __init__(self, hyperparams):
    self.activation = hyperparams.get('activation', 'tanh')
    self.dropout = hyperparams.get('dropout', 0)
    self.hidden_dims = hyperparams['hidden_dims']
    self.timestep = 1000
    self.channels = 25

  def get_model(self):
    self.__model = Sequential()
    self.__model.add(LSTM(self.hidden_dims[0], 
                          return_sequence=len(self.hidden_dims) > 1,
                          activation=self.activation,
                          dropout=self.dropout,
                          input_shape=(self.timestep, self.channels)))
    for i in range(1, len(self.hidden_dims)-1):
      self.__model.add(LSTM(self.hidden_dims[i], 
                            return_sequence=True,
                            activation=self.activation,
                            dropout=self.dropout))
    if len(self.hidden_dims) > 1:
      self.__model.add(LSTM(self.hidden_dims[-1],
                            activation=self.activation,
                            dropout=self.dropout))
    self.__model.add(Dense(4, activation='softmax'))

    return self.__model

class CNNLSTM(BaseModel):
  def __init__():
    pass

  def get_model(self):
    pass




