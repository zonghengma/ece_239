import sys
import os
from read_project_data import DataReader

if __name__ == "__main__":
  train_subjects = eval(sys.argv[1])
  test_subjects = eval(sys.argv[2])
#  print(train_subjects, type(train_subjects))
  assert type(train_subjects) == list
  assert type(test_subjects) == list
  if len(sys.argv) >= 4:
    if_path = sys.argv[3]
  else:
    if_path = "project_datasets"
  if len(sys.argv) >= 5:
    of_path = sys.argv[4]  	
  else:
    of_path = os.path.join("processed_datasets", "processed_data.npz")
  dr = DataReader(if_path, False, False, True)
  dr.preprocess(train_subjects, test_subjects, of_path)
  
