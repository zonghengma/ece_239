import numpy as np
import h5py
import os
import random

class DataReader(object):
  ''' DataReader knows where to find the project datasets in the project,
      finds the .mat files in the directory, and allows the user to access
      the data.

      Attributes:
          dataset_filepath: The filepath for the project_dataset
                              directory that contains the EEG data.
          raw_data: List of dictionaries of data that is in the same format as
                the sample provided by the professor. The list format is:
                [
                  {'X', x_data,
                   'y', y_data
                  },
                  {'X', x_data,
                   'y', y_data
                  },
                  etc
                ]
                Each list item is a subject, [A01T_slice, ..., A09T_slice]
  '''

  def __init__(self, path = "project_datasets/"):
    ''' Accesses the project dataset directory and saves the data.
    '''
    self.dataset_filepath = path
    self.raw_data = dict()
    #df = self.__dataset_filepath
    ## Grab only the mat files in the directory.
    #dir_files = os.listdir(self.__dataset_filepath)
    #files = [file for file in dir_files if '.mat' in file]
    #mat_file_refs = [os.path.join(df, file) for file in files]

    ## Go through each mat file, open it, get the data in the sample
    ## data format, and then save it into self.data.
    #self.raw_data = []
    #for file_ref in mat_file_refs:
    #    X, y = self.__read_file(file_ref)
    #    self.raw_data.append({'X': X, 'y': y})

    for i in range(1, 10):
      file_name = "A0" + str(i) + "T_slice.mat"
      file_path = os.path.join(self.dataset_filepath, file_name)
    
      X, y = self.__read_file(file_path)
      self.raw_data["X" + str(i)] = X
      self.raw_data["y" + str(i)] = y

  def preprocess(self, train_subject, test_subject, file_name="processed_data", file_path="processed_datasets/"):
    if train_subject != test_subject and len([i for i in test_subject if i in train_subject]) > 0:
      raise ValueError("Invalid input")

    if train_subject == test_subject:
      X_all = np.concatenate([self.raw_data["X" + str(i)] for i in train_subject])
      y_all = np.concatenate([self.raw_data["y" + str(i)] for i in train_subject])
      test_index, train_val_index = self.__reservoir_sampling(X_all.shape[0], int(X_all.shape[0] * 0.1))
      X_test = X_all[test_index]
      y_test = y_all[test_index]
      X_train_val = X_all[train_val_index]
      y_train_val = y_all[train_val_index]
    else:
      X_train_val = np.concatenate([self.raw_data["X" + str(i)] for i in train_subject])
      y_train_val = np.concatenate([self.raw_data["y" + str(i)] for i in train_subject])
      X_test = np.concatenate([self.raw_data["X" + str(i)] for i in test_subject])
      y_test = np.concatenate([self.raw_data["y" + str(i)] for i in test_subject])

    val_index, train_index = self.__reservoir_sampling(X_train_val.shape[0], int(X_train_val.shape[0] * 0.1))
    X_val = X_train_val[val_index]
    y_val = y_train_val[val_index]
    X_train = X_train_val[train_index]
    y_train = y_train_val[train_index]

    self.__write_file(X_train, y_train, X_val, y_val, X_test, y_test, file_name, file_path)

  
  def __reservoir_sampling(self, data_length, output_length):
    rate = output_length / data_length
    reservoir = [i for i in range(data_length)]
    for i in range(output_length, data_length):
      rand = random.randint(0, i)
      if rand < output_length:
        reservoir[rand] = i
    return sorted(reservoir[:output_length]), sorted(reservoir[output_length:])

  def __read_file(self, filepath):
    ''' Reads the input file and returns the X and y datasets.

        Args:
            filepath: Filepath of the data set to read.
        Returns:
            X,y; where X is the input data and y is the classifier info
    '''
    file = h5py.File(filepath, 'r')
    X = np.copy(file['image'])
    y = np.copy(file['type'])
    y = y[0,0:X.shape[0]:1]
    y = np.asarray(y, dtype=np.int32)
    return X, y

  def __write_file(self, X_tr, y_tr, X_va, y_va, X_te, y_te, file_name, file_path):
    if ".npz" not in file_name:
      file_name += ".npz"
    path = os.path.join(file_path, file_name)
    np.savez(path, X_train = X_tr, y_train = y_tr, X_val = X_va, y_val = y_va, X_test = X_te, y_test = y_te)

class DataLoader(object):
  pass
