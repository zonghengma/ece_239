from datetime import datetime
import os
import matplotlib.pyplot as plt
from neural_net.data_process.csv_saver import CsvSaver

class DataProcessor(object):
  """Data processing class.

  Given a trained network, characterizes the performance of the network,
  provides data saving functions, and data plotting functions.
  """

  def __init__(self, network):
    self.__network = network
    self.__post_process()

  def save_data_csv(self, save_history=False, filename='output.csv'):
    """Saves the parameters as csv file with fully quoted csv data.

    Default save format will save to the given file. If save_history is
    enabled, then graphs of the loss and accuracy histories will also be
    saved in a directory with name "<datetime--networkname>/".

    Args:
      filename: Filename of the file to save.
    """
    # Create the save file base name.
    network_name = self.__network.get_name()
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    basename = date + '-' + network_name

    # Check that output/ exists and create if not.
    outputs_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.isdir(outputs_dir):
      os.mkdir(outputs_dir)

    # Save the model's parameters to the main save file in output/.
    model_info = self.__network.get_params()
    model_info['run_id'] = basename
    # Stuff with the training and validation results.
    model_info['min_train_loss'] = self.__results['min_train_loss']
    model_info['max_train_acc'] = self.__results['max_train_acc']
    model_info['min_val_loss'] = self.__results['min_val_loss']
    model_info['max_val_acc'] = self.__results['max_val_acc']
    csv_saver = CsvSaver()
    csv_saver.append_to_file(outputs_dir, filename, model_info)

    # History saving functionality.
    if save_history:
      # Create the specific directory if it doesn't exist.
      dirpath = os.path.join(outputs_dir, basename)
      if not os.path.isdir(dirpath):
        os.mkdir(dirpath)

      self.__save_history_images(dirpath, basename)

  def __save_history_images(self, dirpath, basename):
    """Creates the desired directory and then puts image history data into it.

    Creates a directory, saves the parameter values in it in a CSV file,
    and saves the history data into images.

    Args:
      dirpath: Directory to save the files to.
      basename: The base name of these files, which is the directory name.
    """
    # Training history.
    results = self.__results
    fig, ax1 = plt.subplots()
    ax1.plot(results['train_loss_history'], label='Loss', color='b')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training history')
    ax2 = ax1.twinx()
    ax2.plot(results['train_acc_history'], label='Accuracy', color='r')
    ax2.set_ylabel('Accuracy')
    fig.legend()
    plt.savefig(os.path.join(dirpath, 'training_history.png'))

    # Validation history.
    fig, ax1 = plt.subplots()
    ax1.plot(results['val_loss_history'], label='Loss', color='b')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Validation history')
    ax2 = ax1.twinx()
    ax2.plot(results['val_acc_history'], label='Accuracy', color='r')
    ax2.set_ylabel('Accuracy')
    fig.legend()
    plt.savefig(os.path.join(dirpath, 'validation_history.png'))

  def __post_process(self):
    """Post processes the data into a common format.

    Loss and accuracy histories are retrieved from the model itself. The
    history data should be saved for later use and a single value loss
    and accuracy value should be saved.
    """
    model = self.__network.get_model()
    history = self.__network.get_history()

    # Gather the data from the network history.
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']

    # Save the output data.
    results = {}
    results['train_loss_history'] = loss
    results['train_acc_history'] = acc
    results['val_loss_history'] = val_loss
    results['val_acc_history'] = val_acc
    results['min_train_loss'] = min(loss)
    results['max_train_acc'] = max(acc)
    results['min_val_loss'] = min(val_loss)
    results['max_val_acc'] = max(val_acc)
    self.__results = results
