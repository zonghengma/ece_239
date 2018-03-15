import new_preprocess_utilities as utilities
import pdb
import keras.optimizers as optimizers
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
import numpy as np

# Input parameters.
subjects = range(1,10)
output_filename = 'new_preprocess_freq_image_nx8x6x7x6.npz'

# Preprocessing.
X_no_nan, y_no_nan = utilities.import_data_remove_nans(subjects)
X_no_eog, y_no_eog = utilities.remove_eog_and_transpose(X_no_nan, y_no_nan)
X_one_hot, y_one_hot = utilities.one_hot(X_no_eog, y_no_eog)
X, y = utilities.pre_process(X_one_hot, y_one_hot)

# Data shuffle and split.
X_shuffle, y_shuffle = shuffle(X, y)
X_test = X_shuffle[0:50, :, :]
y_test = y_shuffle[0:50]
total_training_size = X.shape[0] - 50
validation_size = int(total_training_size * .9)
X_val = X_shuffle[50: validation_size + 50, :, :]
y_val = y_shuffle[50: validation_size + 50]
X_train = X_shuffle[validation_size + 50:, :, :]
y_train = y_shuffle[validation_size + 50:]

# Data save.
base_output_directory = '../data_access/processed_datasets/'
np.savez(base_output_directory + output_filename, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
