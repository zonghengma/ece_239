import csv

field_names = ['learning_rate', 'lr_decay', 'batch_size', 'conv_units', 'kernel_size', 'pool_size', 'pool_strides', 'dilation_rate', 'conv_dropout', 'lstm_units', 'lstm_dropout', 'dense_units', 'dense_dropout']

learning_rates = [5e-4]
lr_decay = [0]
batch_size = [32]
conv_units = [[20,16]]
kernel_size = [[2, 2]]
pool_size = [[1, 2]]
pool_strides = [[1, 2]]
dilation_rate = [1]
conv_dropout = [[0, 0]]
lstm_units = [[86],[96],[128]]
lstm_dropout = [0.4,0.45,0.5]
dense_units = [[512],[768]]
dense_dropout = [0.5,0.6]

csv_file = open("config.csv", "w", newline='')
writer = csv.DictWriter(csv_file, field_names)
writer.writeheader()

for lr in learning_rates:
  for decay in lr_decay:
    for bs in batch_size:
      for cu in conv_units:
        for ks in kernel_size:
          for ps in pool_size:
            for pstr in pool_strides:
              for dr in dilation_rate:
                for cd in conv_dropout:
                  for lu in lstm_units:
                    for ld in lstm_dropout:
                      for du in dense_units:
                        for ddp in dense_dropout:
                          if len(cu) == len(ps) and len(ps) == len(pstr) and len(pstr) == len(cd):
                            writer.writerow({
                              'learning_rate': lr,
                              'lr_decay': decay,
                              'batch_size': bs,
                              'conv_units': cu,
                              'kernel_size': ks,
                              'pool_size': ps,
                              'pool_strides': pstr,
                              'dilation_rate': dr,
                              'conv_dropout': cd,
                              'lstm_units': lu,
                              'lstm_dropout': ld,
                              'dense_units': du,
                              'dense_dropout': ddp
                            })

#writer.writerow({
#  'learning_rate': 5e-5,
#  'lr_decay': 0.001,
#  'batch_size': 32,
#  'conv_units': [64,64,128,128],
#  'kernel_size': (2, 2),
#  'pool_size': (1,1),
#  'pool_strides': (1,1),
#  'dilation_rate': 1,
#  'conv_dropout': [0,0.25],
#  'lstm_units': [128],
#  'lstm_dropout': 0.2,
#  'dense_units': 512,
#  'dense_dropout': 0.25
#})
#writer.writerow({
#  'learning_rate': 5e-5,
#  'lr_decay': 0.001,
#  'batch_size': 32,
#  'conv_units': [64,64,128,128],
#  'kernel_size': (2, 2),
#  'pool_size': (1,1),
#  'pool_strides': (1,1),
#  'dilation_rate': 1,
#  'conv_dropout': [0,0.25],
#  'lstm_units': [128],
#  'lstm_dropout': 0.2,
#  'dense_units': 512,
#  'dense_dropout': 0.25
#})
csv_file.close()
