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
          raw_data: Dictionaries of data that is in the same format as
                the sample provided by the professor. The list format is:
                  {
                    'X1': X of subject 1,
                    'y1': y of subject 1,
                    'X2': X of subject 2,
                    'y2': y of subject 2,
                    ...
                  }
                Each list item is a subject, [A01T_slice, ..., A09T_slice]
  '''

  def __init__(self, path = "project_datasets/", split = False, image = False):
    ''' Accesses the project dataset directory and saves the data.
    '''
    self.dataset_filepath = path
    self.raw_data = dict()

    for i in range(1, 10):
      file_name = "A0" + str(i) + "T_slice.mat"
      file_path = os.path.join(self.dataset_filepath, file_name)
    
      X, y = self.__read_file(file_path)
      y = self.__one_hot_encode(y)
      if split:
        X, y = self.__split(X, y)
      if image:
        X = self.__image(X)
      self.raw_data["X" + str(i)] = X
      self.raw_data["y" + str(i)] = y

  def preprocess(self, train_subjects, test_subjects, file_path=""):
    if train_subjects != test_subjects and len([i for i in test_subjects if i in train_subjects]) > 0:
      raise ValueError("Invalid input")

    if train_subjects == test_subjects:
      X_all = np.concatenate([self.raw_data["X" + str(i)] for i in train_subjects])
      y_all = np.concatenate([self.raw_data["y" + str(i)] for i in train_subjects])
      test_index, train_val_index = self.__reservoir_sampling(X_all.shape[0], int(X_all.shape[0] * 0.1))
      X_test = X_all[test_index]
      y_test = y_all[test_index]
      X_train_val = X_all[train_val_index]
      y_train_val = y_all[train_val_index]
    else:
      X_train_val = np.concatenate([self.raw_data["X" + str(i)] for i in train_subjects])
      y_train_val = np.concatenate([self.raw_data["y" + str(i)] for i in train_subjects])
      X_test = np.concatenate([self.raw_data["X" + str(i)] for i in test_subjects])
      y_test = np.concatenate([self.raw_data["y" + str(i)] for i in test_subjects])

    val_index, train_index = self.__reservoir_sampling(X_train_val.shape[0], int(X_train_val.shape[0] * 0.1))
    X_val = X_train_val[val_index]
    y_val = y_train_val[val_index]
    X_train = X_train_val[train_index]
    y_train = y_train_val[train_index]

    self.__write_file(X_train, y_train, X_val, y_val, X_test, y_test, file_path)

  def __one_hot_encode(self, y):
    res = np.zeros((y.shape[0], 4))
    res[np.arange(y.shape[0]), y.astype(int)-769] = 1
    return res

  def __split(self, X, y): 
    #resX = np.concatenate(np.split(X, 8, axis=2))
    #resY = np.zeros((y.shape[0] * 8, y.shape[1]))
    #for i in range(y.shape[0] * 8):
    #  resY[i, :] = y[i % 8, :]
    #return resX.transpose(0,2,1), resY
    assert False
        
  def __image(self, X):
#    img = [[2,3,4,1,5,6],[7,8,9,10,11,12],[14,15,16,17,18,13],[19,20,21,22,0,0]]
    img = np.array([[0,0,0,1,0,0,0],[0,2,3,4,5,6,0],[7,8,9,10,11,12,13],[0,14,15,16,17,18,0],[0,0,19,20,21,0,0],[0,0,0,22,0,0,0]])
    res = np.zeros((X.shape[0], 40, img.shape[0], img.shape[1], 25))
    for i in range(X.shape[0]):
      for j in range(X.shape[1]):
        for ii in range(img.shape[0]):
          for jj in range(img.shape[1]):
            if img[ii,jj] != 0:
              #for k in range(40):
              res[i,j//25,ii,jj,j%25] = X[i, j, img[ii,jj]-1]
    return res 
        
  
  def __reservoir_sampling(self, data_length, output_length):
    rate = output_length / data_length
    reservoir = [i for i in range(data_length)]
    for i in range(output_length, data_length):
      rand = random.randint(0, i)
      if rand < output_length:
        reservoir[i], reservoir[rand] = reservoir[rand], reservoir[i]
        #reservoir[rand] = i
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
    return X.transpose(0,2,1)[:,:,:22], y

  def __write_file(self, X_tr, y_tr, X_va, y_va, X_te, y_te, file_path):
    if ".npz" not in file_path:
      file_path += ".npz"
    np.savez(file_path, X_train = X_tr, y_train = y_tr, X_val = X_va, y_val = y_va, X_test = X_te, y_test = y_te)

class DataLoader(object):
  def __init__(self, path = ""):
    self.processed_filepath = path
    if self.processed_filepath == "":
      self.processed_filepath = os.path.join("processed_datasets", "processed_data")
    if ".npz" not in self.processed_filepath:
      self.processed_filepath += ".npz"
    
  def load(self):
    data = np.load(self.processed_filepath)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    #return (X_train.transpose(0,2,1), y_train, X_val.transpose(0,2,1), y_val, X_test.transpose(0,2,1), y_test) 
    return (X_train, y_train, X_val, y_val, X_test, y_test) 
