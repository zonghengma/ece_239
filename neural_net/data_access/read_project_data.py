import numpy as np
import h5py
import os

class DataReader(object):
    ''' DataReader knows where to find the project datasets in the project,
        finds the .mat files in the directory, and allows the user to access
        the data.

        Attributes:
            __dataset_filepath: The default filepath for the project_dataset
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

    def __init__(self):
        ''' Accesses the project dataset directory and saves the data.
        '''
        self.__dataset_filepath = 'project_datasets/'
        df = self.__dataset_filepath
        # Grab only the mat files in the directory.
        dir_files = os.listdir(self.__dataset_filepath)
        files = [file for file in dir_files if '.mat' in file]
        mat_file_refs = [os.path.join(df, file) for file in files]

        # Go through each mat file, open it, get the data in the sample
        # data format, and then save it into self.data.
        self.raw_data = []
        for file_ref in mat_file_refs:
            X, y = self.__read_file(file_ref)
            self.raw_data.append({'X': X, 'y': y})


    # def preprocess(self):


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

