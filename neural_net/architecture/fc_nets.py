from neural_net.architecture.base_model import BaseModel
import keras.models as models
import keras.layers as layers

class ThreeLayerFcNet(BaseModel):
  def __init__(self, hyperparams, archparams):
    # Name the network.
    name = 'three-layer-fc-net'

    # Grab architecture parameters.
    hidden_units = archparams['hidden_units']
    input_dim = archparams['input_dim']

    # Set up the model architecture>
    model = models.Sequential()
    model.add(layers.Dense(hidden_units[0], input_dim=input_dim))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(hidden_units[1], input_dim=input_dim))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(hidden_units[2], input_dim=input_dim))
    model.add(layers.Activation('softmax'))

    # Super function will handle the rest of the generic model setup.
    super().__init__(model, name, hyperparams, archparams)
