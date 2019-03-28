import os
import csv
import numpy as np
import settings as s
import sys

phones = []
length = 0
with open(os.path.join("timit_data.csv"), "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        phones.append(row["numpy_X"])
        length += 1

first_i = 0
second_i = 0
n = 0
s_1 = np.zeros((1, 66))
s_2 = np.zeros((1, 66))

for i, phone in enumerate(phones):
    x = np.nan_to_num(np.load(phone))
    n += x.shape[0]
    s_1 += x.sum(axis=0)
    s_2 += np.sum(np.square(x), axis=0)

means = s_1/n
stds= np.sqrt((n/(n-1))*((s_2/n) - np.square(s_1/n)))
output = np.vstack((means, stds))
np.save(os.path.join(s.OUTPUT_DIR, "zscores_new"), output)
