import os
import numpy as np
import settings as s
import sys

file_dir = os.path.join(s.OUTPUT_DIR, "timit_matrices")
first_i = 0
second_i = 0
length = len(os.listdir(file_dir))
n = 0
s_1 = np.zeros((1, 85))
s_2 = np.zeros((1, 85))

for i, phone in enumerate(os.listdir(file_dir)):
    x = np.nan_to_num(np.load(os.path.join(file_dir, phone)))
    n += x.shape[0]
    s_1 += x.sum(axis=0)
    s_2 += np.sum(np.square(x), axis=0)
    if i % 10000 == 0:
        print(f"n: {n}")
        print(f"mean: {s_1/n}")
        print(f"stddev: {np.sqrt((n/(n-1))*((s_2/n) - np.square(s_1/n)))}")
        print(f"{i}/{length}")

means = s_1/n
stds= np.sqrt((n/(n-1))*((s_2/n) - np.square(s_1/n)))
output = np.vstack((means, stds))
np.save(os.path.join(s.OUTPUT_DIR, "zscores"), output)
