To generate train/validation/test data for selected subject and save the data to a file, run the following:
    python pre_process.py [train_subjects] [test_subjects] path_to_project_datsets path_to_output_file
For example:
    python pre_process.py [1,2,3,4,5,6,7,8] [9] C:\project_datasets C:\output_data\data.npz
Gives you train/validation data on subjects 1 through 8(90% and 10%) and test data on subject 9(100%), and save the data to the file "data.npz"

To load the datafile, in main.py:
  from neural_net.data_access.read_project_data import DataLoader
  dl = DataLoader("C:\output_data\data.npz")
  X_train, y_train, X_val, y_val, X_test, y_test = dl.load()
