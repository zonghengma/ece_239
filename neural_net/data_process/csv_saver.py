import os
import pandas as pd
import csv

class CsvSaver(object):
  """CSV file saving class to write data to file.

  Attributes:
    full_path: The fully qualified file path for the CSV file.
  """
  def append_to_file(self, dirpath, filename, data):
    """Appends the current run info to the output file in the directory.

    File does not have to exist. It is created if it doesn't. This output
    file should be common for all models run so that models can be quickly
    compared.

    Args:
      dirpath: Path to the directory to save the file.
      data: Dictionary of parameters that characterize this run. Contains
            model name, hyperparameters, and archparams.
    """
    full_path = os.path.join(dirpath, filename)
    df = self.__init_file(full_path)

    # Create empty entries for any data header that doesn't already exist.
    for key in data.keys():
      if key not in df:
        df[key] = ''

    # Append the new row of data and save to CSV.
    df = df.append(data, ignore_index=True)
    df.to_csv(full_path, index=False, quoting=csv.QUOTE_ALL)

  def __init_file(self, filepath):
    """Return a pandas DataFrame for the data."""
    if not os.path.exists(filepath):
      return pd.DataFrame()

    return pd.read_csv(filepath)
