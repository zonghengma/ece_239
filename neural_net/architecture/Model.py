import numpy as np
from neural_net.architecture.base_model import BaseModel
from keras.models import Sequential
from keras.layers import LSTM, Dense

class StackedLSTM(BaseModel):
  def __init__(self, hyperparams, archparams):
    """ initialize the StackedLSTM with hyperparams
    archparams:
      activation: activation function, default is tanh
      dropout: dropout rate, default is 0
      hidden_dims: number of units in each LSTM layer, default structure is 
                  [32, 32, 32], at least one layer
      input_dim: input dimensions, default is 288x1000x25
    """
    self.activation = archparams.get('activation', 'tanh')
    self.dropout = archparams.get('dropout', 0)
    self.hidden_dims = archparams.get('hidden_dims', [32, 32, 32])
    self.input_dim = archparams.get('input_dim', (288, 1000, 25))
    self.timestep = self.input_dim(1)
    self.channels = self.input_dim(2)
    model = self.construct_model()

    super().__init__(model, 'StackedLSTM', hyperparams, archparams)

  def construct_model(self):
    model = Sequential()

    # at least one LSTM layer
    model.add(LSTM(self.hidden_dims[0], 
                          return_sequence=len(self.hidden_dims) > 1,
                          activation=self.activation,
                          dropout=self.dropout,
                          input_shape=(self.timestep, self.channels)))
    
    # LSTM layers in the middle
    for i in range(1, len(self.hidden_dims)-1):
      model.add(LSTM(self.hidden_dims[i], 
                            return_sequence=True,
                            activation=self.activation,
                            dropout=self.dropout))
    
    # last LSTM layer
    if len(self.hidden_dims) > 1:
      model.add(LSTM(self.hidden_dims[-1],
                            activation=self.activation,
                            dropout=self.dropout))

    # dense layer for classification
    model.add(Dense(4, activation='softmax'))
    return model