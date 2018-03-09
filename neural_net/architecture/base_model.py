import keras.optimizers as optimizers

class BaseModel(object):
  """Base class representing a complete neural net architecture.

  Use this BaseModel class as the base class of a fully defined
  neural network architecture. Contains a keras model, a unique name for
  the architecture, and the input parameters that make this model unique.

  Attributes:
    model: The keras model to save.
    name: String of unique name for the model. Every different architecture
          should have a uniquely identifying name.
    params: The params that are unique to this architecture that will be
            used to compare to other runs of this same architecture.
            A dictionary of {'param': value, 'param': value}.

  """
  def __init__(self, keras_model, name, hyperparams):
    """Set and compile the model, everything before model.fit().

    Args:
      keras_model: A keras Sequential() model with layers already added.
      name: String of unique identifier for this architecture.
      hyperparams: Dictionary of hyperparameters for this run. Note that this
              should be a common set of hyperparameters for ALL models,
              completely independent of the architecture, such as batch_size,
              learning_rate, lr_decay, etc.
              {<param_name>: <param_value>, <param_name>: <param_value>, ...}
    """
    self.__model = keras_model
    self.__name = name
    self.__params = hyperparams
    self.__compile()

  def get_model(self):
    return self.__model

  def get_parameters(self):
    return self.__params

  def get_name(self):
    return self.__name

  def train(self, training_data):
    """Performs the network training step using the input data.

    Args:
      training_data: Training data to provide to the model. Should be a
                     dictionary of:
                     {'X_train': <>,
                      'y_train': <>,
                      'X_val': <>
                      'y_val': <>}

    """
    # Gather parameters>
    model = self.__model
    params = self.__params
    X_train = training_data['X_train']
    y_train = training_data['y_train']
    validation_data = (training_data['X_val'], training_data['y_val'])
    batch_size = params['batch_size']
    epochs = params['epochs']
    verbose = params['verbose']

    # Perform the actual training.
    model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs,
              verbose=verbose, validation_data=validation_data)

  def __compile(self):
    """Helper function that calls the compile function of the model."""
    params = self.__params
    optimizer = self.__create_optimizer()
    loss_function = params['loss_function']

    model = self.__model
    model.compile(optimizer=optimizer, loss=loss_function,
                  metrics=['accuracy'])

  def __create_optimizer(self):
    """Manually creates optimizer so parameters can be tuned.

    Args:
      hyperparams: Same hyperparams dictionary passed into the class in
                   the constructor.
    Returns:
      A keras.optimizer object of the desired type to be directly plugged
      into the model. Add to the model at compile time by calling
      model.compile(..., optimizer=<returned value>)
    """
    hyperparams = self.__params
    # The name of the optimizer to use is set by optimizer param.
    optimizer = hyperparams['optimizer']
    # Load parameters common to all optimizers.
    learning_rate = hyperparams['learning_rate']
    lr_decay = hyperparams['lr_decay']

    # We return different types of optimizers based on what optimizer was
    # selected and then set the appropriate values.
    if optimizer == 'adam':
      # Load parameters for this optimizer, but provide default values too.
      beta_1 = hyperparams['beta1'] if 'beta1' in hyperparams else 0.9
      beta_2 = hyperparams['beta2'] if 'beta2' in hyperparams else 0.999
      return optimizers.Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2,
                             decay=lr_decay)
    else:
      raise Error('Optimizer: ' + optimizer + ' has not been implemented ' +
                  'yet in base_model.py. Implement it in __create_optimizer ' +
                  'method to be used in the model.')
