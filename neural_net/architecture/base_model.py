import keras.optimizers as optimizers
import keras.callbacks

class BaseModel(object):
  """Base class representing a complete neural net architecture.

  Use this BaseModel class as the base class of a fully defined
  neural network architecture. Contains a keras model, a unique name for
  the architecture, and the input parameters that make this model unique.

  Attributes:
    model: The keras model to save.
    name: String of unique name for the model. Every different architecture
          should have a uniquely identifying name.
    hyperparams: The params that are unique to this architecture that will be
                 used to compare to other runs of this same architecture.
                 A dictionary of {'param': value, 'param': value}.
    archparams: The params that are unique the actual model. These are not
                directly used by the BaseModel class, but only saved so
                that they can be reported by the base class's save function.
                A dictionary of {'param': value, 'param': value}.
    history: Keras History object for saving the model performance history.
  """
  def __init__(self, keras_model, name, hyperparams, archparams):
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
    self.__hyperparams = hyperparams
    self.__archparams = archparams
    self.__compile()

  def get_model(self):
    return self.__model

  def get_name(self):
    return self.__name

  def get_history(self):
    return self.__history

  def get_params(self):
    """Returns all the parameters used in this model.

    Returns both the hyperparams and archparams in a single dictionary.
    Performs escaping of data so that it can be directly input into a CSV
    writer and written line-by-line.

    Returns:
      Dictionary of {param: value, param: value}
    """
    params = {}
    for k, v in self.__hyperparams.items():
      # Unconditionally add these since can't have any errors.
      params[k] = v

    for k, v in self.__archparams.items():
      # Raise an error if there's any parameter overlap.
      if k in params:
        raise Error('Found duplicate key in archparams and hyperparams: ' +
                    k + ', cannot have duplicate keys')
      params[k] = v

    # Insert name.
    params['model_name'] = self.__name

    return params

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
    params = self.__hyperparams
    X_train = training_data['X_train']
    y_train = training_data['y_train']
    validation_data = (training_data['X_val'], training_data['y_val'])
    batch_size = params['batch_size']
    epochs = params['epochs']
    verbose = params['verbose']

    # Perform the actual training.
    self.__history = model.fit(x=X_train, y=y_train, batch_size=batch_size,
                               epochs=epochs, verbose=verbose,
                               validation_data=validation_data)

  def __compile(self):
    """Helper function that calls the compile function of the model."""
    params = self.__hyperparams
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
    hyperparams = self.__hyperparams
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
