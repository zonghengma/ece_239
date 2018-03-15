import numpy as np
import h5py
import pdb
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import signal
import matplotlib.pyplot as plt
from scipy.stats import mstats
from scipy.signal import welch

def import_data_remove_nans(subjects_list):
  print('--- Importing data and removing nans ---')
  # Data import

  # Go through each dataset, retrieve the data, then just store it in the
  # list for use later.
  dataset = []
  for subj in subjects_list:
    filename = '../../project_datasets/A0' + str(subj) + 'T_slice.mat'
    file = h5py.File(filename, 'r')
    X = np.copy(file['image'])
    y = np.copy(file['type'])
    y = y[0, 0:X.shape[0]:1]
    y = np.asarray(y, dtype=np.int32)
    dataset.append([X, y])

  # Data is now in dataset[subject_num][0=image,1=type][data]

  # Remove nans
  dataset_removed_nans = []
  for dataset_i in range(len(dataset)):
    nan_rows = []
    X = dataset[dataset_i][0]
    y = dataset[dataset_i][1]
    # Detect the nans by rows, which is the 0th idx.
    for trial in range(X.shape[0]):
      if np.any(np.isnan(X[trial,:,:])):
        nan_rows.append(trial)

    # Remove the nans from the data and then the classes.
    num_trials = X.shape[0]
    good_rows = [x for x in range(num_trials) if x not in nan_rows]
    res_X = X[good_rows]
    res_y = y[good_rows]

    dataset_removed_nans.append([res_X, res_y])

  # Create the np array from the dataset.
  images = []
  classes = []
  # Go through each subject.
  for i in range(len(dataset_removed_nans)):
    # Go through each trial and add it to the final array.
    for trial in range(dataset_removed_nans[i][0].shape[0]):
      images.append(dataset_removed_nans[i][0][trial,:,:])
      classes.append(dataset_removed_nans[i][1][trial])

  np_images = np.array(images)
  np_classes = np.array(classes)
  return np_images, np_classes

def remove_eog_and_transpose(X, y):
  print('--- Removing eog data and tranposing X ---')
  assert X.shape[1] == 25 and X.shape[2] == 1000
  assert y.shape[0] == X.shape[0]

  return X[:, 0:22, :].transpose(0, 2, 1), y

def one_hot(X, y):
  print('--- One hot encoding y ---')
  y -= 769
  y_one_hot = to_categorical(y)
  return X, y_one_hot

######## PRE PROCESS ##########

def pre_process(X, y):
  print('--- Preprocessing ---')
  # Time domain stuff.
  # X, y = reshape_into_image(X, y)
  # X, y = high_pass_filter(X, y)
  # X, y = get_winsorized(X, y)
  # X, y = get_normalized_by_trial_and_channel(X, y)
  # X, y = get_normalized_by_all(X, y)

  # Freq domain stuff.
  X, y = high_pass_filter(X, y)
  X, y = reshape_into_image(X, y)
  # X is diff shape now, image format.
  X, y = split_into_timesteps(X, y)
  # X is diff shape now, timesplit and image format.
  # X, y = get_freq_domain_3channels(X, y)
  # X, y = normalize_image_0_1_by_3channel(X, y)
  X, y = get_freq_domain_6channels(X, y)
  X, y = normalize_image_0_1_by_6channel(X, y)
  # Final shape is [trials, num_timesections, rows, cols, num_freqs]
  # Full dataset:  [2558, 8, 6, 7, 3 or 6]

  # X, y = remove_1hz_and_lower(X, y)
  # X, y = normalize(X, y)
  return X, y

def subplot(X1, X2, trial, channel):
  f, axarr = plt.subplots(2, sharex=True)
  axarr[0].plot(X1[trial,:,channel])
  axarr[1].plot(X2[trial,:,channel])
  plt.show()

def normalize_image_0_1_by_3channel(X, y):
  print('--- Normalizing by channel, expecting 3 channel input ---')
  assert X.shape[4] == 3
  X_winsor = np.copy(X)
  X_winsor[:, :, :, :, 0] = mstats.winsorize(X[:, :, :, :, 0], [0, 0.1])
  X_winsor[:, :, :, :, 1] = mstats.winsorize(X[:, :, :, :, 1], [0, 0.1])
  X_winsor[:, :, :, :, 2] = mstats.winsorize(X[:, :, :, :, 2], [0, 0.1])

  X_norm = X_winsor - X_winsor.min()
  X_norm = X_norm / X_winsor.std()

  return X_norm, y

def normalize_image_0_1_by_6channel(X, y):
  print('--- Normalizing by channel, expecting 6 channel input ---')
  assert X.shape[4] == 6
  X_winsor = np.copy(X)
  # Only winsorize across 0 1 2, since 3 4 5 is already [0 1]
  X_winsor[:, :, :, :, 0] = mstats.winsorize(X[:, :, :, :, 0], [0, 0.1])
  X_winsor[:, :, :, :, 1] = mstats.winsorize(X[:, :, :, :, 1], [0, 0.1])
  X_winsor[:, :, :, :, 2] = mstats.winsorize(X[:, :, :, :, 2], [0, 0.1])

  X_norm = np.copy(X_winsor)
  X_norm[:, :, :, :, 0:3] = X_winsor[:, :, :, :, 0:3] - X_winsor[:, :, :, :, 0:3].min()
  X_norm[:, :, :, :, 0:3] = np.divide(X_norm[:, :, :, :, 0:3], X_winsor[:, :, :, :, 0:3].std(), where=X_winsor[:, :, :, :, 0:3] > 0)

  return X_norm, y

def get_normalized_by_all(X, y):
  print('--- Normalizing by all data ---')
  mean = X.sum() / X.size
  X_meaned = X - mean
  stdev = X.std()
  X_res = X_meaned / stdev
  return X_res, y

def reshape_into_image(X, y):
  img = np.array([[0,0,0,1,0,0,0],[0,2,3,4,5,6,0],[7,8,9,10,11,12,13],[0,14,15,16,17,18,0],[0,0,19,20,21,0,0],[0,0,0,22,0,0,0]])
  res = np.zeros((X.shape[0], X.shape[1], img.shape[0], img.shape[1]))
  rows = img.shape[0]
  cols = img.shape[1]
  print('--- Reshaping into image format, ' + str(rows) + ' x ' + str(cols) + ' ---')

  for ii in range(rows):
    for jj in range(cols):
      if img[ii,jj] != 0:
        res[:,:, ii, jj] = X[:, :, img[ii, jj]-1]
  return res, y

def split_into_timesteps(X, y):
  assert len(X.shape) == 4
  num_splits = 8
  print('--- Splitting into ' + str(num_splits) + ' sections of timesteps ---')
  assert X.shape[1] % num_splits == 0
  split_size = X.shape[1] // num_splits
  X_split = np.zeros((X.shape[0], num_splits, split_size, X.shape[2], X.shape[3]))
  for i in range(num_splits):
    res = X[:, i * split_size : (i + 1) * split_size, :, :]
    X_split[:, i, 0 : split_size, :, :] = res
  return X_split, y

def get_freq_domain_3channels(X, y):
  print('--- Getting frequency domain, 3 channel output ---')

  # fft method.
  # x_freq = np.absolute(np.fft.fft(X, axis=1))
  # x_freq_half = x_freq[:, 0 : x_freq.shape[1] // 2, :]
  # pdb.set_trace()

  # welch method.
  psd_welch = welch(X, nperseg=125, fs=250, axis=2)

  # Go through each bin and do the accounting for the magnitudes.
  freqs = psd_welch[0]
  theta_low = 4
  theta_high = 7
  alpha_low = 8
  alpha_high = 13
  beta_low = 14
  beta_high = 30
  welches = psd_welch[1]

  thetas_idx = (freqs >= theta_low) & (freqs <= theta_high)
  alphas_idx = (freqs >= alpha_low) & (freqs <= alpha_high)
  betas_idx  = (freqs >= beta_low)  & (freqs <= beta_high)
  thetas = welches[:, :, thetas_idx, :, :]
  alphas = welches[:, :, alphas_idx, :, :]
  betas  = welches[:, :, betas_idx,  :, :]

  # allocate space
  # 3 for the num of freq ranges.
  X_res = np.zeros((X.shape[0], X.shape[1], X.shape[3], X.shape[4], 3))
  X_res[:, :, :, :, 0] = np.sum(thetas, axis=2)
  X_res[:, :, :, :, 1] = np.sum(alphas, axis=2)
  X_res[:, :, :, :, 2] = np.sum(betas,  axis=2)

  return X_res, y

def get_freq_domain_6channels(X, y):
  print('--- Getting frequency domain, 6 channel output ---')

  # welch method.
  psd_welch = welch(X, nperseg=125, fs=250, axis=2)

  # Go through each bin and do the accounting for the magnitudes.
  freqs = psd_welch[0]
  theta_low = 4
  theta_high = 7
  alpha_low = 8
  alpha_high = 13
  beta_low = 14
  beta_high = 30
  welches = psd_welch[1]

  thetas_idx = (freqs >= theta_low) & (freqs <= theta_high)
  alphas_idx = (freqs >= alpha_low) & (freqs <= alpha_high)
  betas_idx  = (freqs >= beta_low)  & (freqs <= beta_high)
  thetas = welches[:, :, thetas_idx, :, :]
  alphas = welches[:, :, alphas_idx, :, :]
  betas  = welches[:, :, betas_idx,  :, :]

  # allocate space
  # 3 for the num of freq ranges.
  X_res = np.zeros((X.shape[0], X.shape[1], X.shape[3], X.shape[4], 6))
  X_res[:, :, :, :, 0] = np.sum(thetas, axis=2)
  X_res[:, :, :, :, 1] = np.sum(alphas, axis=2)
  X_res[:, :, :, :, 2] = np.sum(betas,  axis=2)
  # sum_freqs is the summation of the 0 1 2 channels.
  sum_freqs = np.sum(X_res[:, :, :, :, 0:3], axis=4)
  X_res[:, :, :, :, 3] = np.divide(X_res[:, :, :, :, 0], sum_freqs, where=sum_freqs > 0)
  X_res[:, :, :, :, 4] = np.divide(X_res[:, :, :, :, 1], sum_freqs, where=sum_freqs > 0)
  X_res[:, :, :, :, 5] = np.divide(X_res[:, :, :, :, 2], sum_freqs, where=sum_freqs > 0)

  return X_res, y

def get_normalized_by_trial_and_channel(X, y):
  print('--- Normalizing data ---')
  scaler = StandardScaler()
  X_norm = np.copy(X)
  for trial in range(X.shape[0]):
    X_norm[trial, :, :] = scaler.fit_transform(X[trial, :, :])
  return X_norm, y

def get_winsorized(X, y):
  print('--- Winsorizing ---')
  winsor_lims = [0.05, 0.05]
  X_winsorized = mstats.winsorize(X, limits=winsor_lims, axis=1)
  return X_winsorized, y

def high_pass_filter(X, y):
  print('--- High pass filtering ---')
  nyquist = 0.5 * 250
  cutoff = 1 / nyquist
  b, a = signal.butter(3, cutoff, btype='highpass', analog=False)
  X_filt = signal.filtfilt(b, a, X, axis=1)
  return X_filt, y

def remove_1hz_and_lower(X, y):
  X_res = X[:, 5:, :]
  return X_res, y

def normalize_MinMaxScaler(X, y):
  scaler = MinMaxScaler()
  X_res = np.copy(X)
  for i in range(X.shape[0]):
    X_res[i, :, :] = scaler.fit_transform(X[i, :, :])
  return X_res, y
