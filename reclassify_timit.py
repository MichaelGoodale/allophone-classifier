import os 
import csv
import sys
import pandas as pd
import random
import numpy as np
from keras.models import load_model
import argparse 
import subprocess
from uuid import uuid1
import multiprocessing
import itertools
import settings as s

N_CPU = multiprocessing.cpu_count()
TEMP_DIR = os.path.join(s.OUTPUT_DIR, "classifier_temporary")

INPUTLIST = os.path.join(TEMP_DIR, "inputlist")
OUTPUTLIST = os.path.join(TEMP_DIR, "outputlist")
PREDICTIONS = os.path.join(TEMP_DIR, "predictions")
AUDIO_DIR = os.path.join(TEMP_DIR, "audio")
VOT_DIR = os.path.join(TEMP_DIR, "autovot_files")
NUMPY_MATRICES = os.path.join(TEMP_DIR, "numpy_matrices")

for path in [VOT_DIR, NUMPY_MATRICES, AUDIO_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

def _calculate_features(i, phone_list):
    inputlist = INPUTLIST+str(i)
    outputlist = OUTPUTLIST+str(i)
    predictions = PREDICTIONS+str(i)
    phone_list = sorted(phone_list, key=lambda x: (x[2], x[0]))
    with open(inputlist, "w") as input_f, open(outputlist, "w") as output_f:
        for begin, end, path, phone_id, *rest in phone_list:
            input_f.write("\"{}\" {:3f} {:3f} {:3f} {:3f} [seconds]\n".format(path,begin,end,begin,end))
            output_path = os.path.join(VOT_DIR, str(phone_id))
            output_f.write(output_path+"\n")
    ret = subprocess.run([os.path.join(s.PATH_TO_AUTOVOT, "VotFrontEnd2"), "-dont_normalize", inputlist, outputlist, "null"])
    subprocess.run([os.path.join(s.PATH_TO_AUTOVOT, "VotDecode"), "-pos_only", "-output_predictions", predictions, outputlist, "null", s.CLASSIFIER])
    memo = {}
    with open(predictions) as pred_f:
        for count, ((begin, end, path, phone_id, *rest), line) in enumerate(zip(phone_list, pred_f)):
            vot_conf, vot_begin, vot_end = line.split()
            matrix = np.genfromtxt(os.path.join(VOT_DIR, str(phone_id)), skip_header=1)
            matrix_file = os.path.join(NUMPY_MATRICES, "{}.npy".format(phone_id))
            matrix = np.insert(matrix, 63, [[vot_conf], [vot_begin], [vot_end]], axis=1)
            np.save(matrix_file, matrix)

def get_features(phone_list):
    '''Takes in a list of 4-tuples formatted with input, output, path and phone_id'''
    n_phones = len(phone_list)
    list_size = int(n_phones/(N_CPU))
    sub_lists = [phone_list[i:i+list_size] for i in range(0, n_phones, list_size)]
    sub_lists = [x for x in sub_lists if len(x) > 0]
    while len(sub_lists) > N_CPU:
        temp = sub_lists.pop()
        sub_lists[len(sub_lists)-1] += temp
    jobs = []
    for i in range(len(sub_lists)):
        p = multiprocessing.Process(target=_calculate_features, args=(i, sub_lists[i]))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()

phones = pd.read_csv("data/timit_phone_info.csv")
phones = phones.loc[phones["underlying_phoneme"].isin(["B", "D", "G", "P", "T", "K", "T_Y", "D_Y", "N_D", "N_T"]) ,:]
input_phones = []
for _, stop in phones.iterrows():
    input_phones.append((stop["window_begin"], stop["window_end"], stop["path"]+".WAV", stop["phone_id"]))

get_features(input_phones)
models = {"P":load_model("p_model.hd5"),
          "T":load_model("t_model.hd5"),
          "K":load_model("k_model.hd5"),
          "B":load_model("b_model.hd5"),
          "D":load_model("d_model.hd5"),
          "G":load_model("g_model.hd5")}

LABELS = {"P": ["p", "prl", "pcl", "-", "other"],
          "T": ["t", "trl", "tcl", "-", "dx", "q", "other"],
          "K": ["k", "krl", "kcl", "-", "other"],
          "B": ["b", "brl", "bcl", "-", "other"],
          "D": ["d", "drl", "dcl", "-", "dx", "other"],
          "G": ["g", "grl", "gcl", "-", "other"]}
#zscores = np.load("data/zscores.npy")[:, :63]
#zscores = np.hstack((zscores, [[88.0977, 30.28, 68.447], [515.34, 43.5, 75.344]]))
zscores = np.load("data/zscores_new.npy")

def get_prediction(row): 
    phone_id, phoneme = row[0], row[1]
    if phoneme in ["N_T", "T_Y"]:
        phoneme = "T"
    elif phoneme in ["N_D", "D_Y"]:
        phoneme = "D"
    X = np.nan_to_num(np.load(os.path.join(NUMPY_MATRICES, "{}.npy".format(phone_id))))
    X = ((X - zscores[0])/zscores[1])
    X = X[None, ...]
    pred = models[phoneme].predict(X)
    pred = pred.flatten()
    pred_i = np.argmax(pred)
    return LABELS[phoneme][pred_i], pred

def save_numpy_mat(row): 
    phone_id, phoneme = row[0], row[1]
    return os.path.join(NUMPY_MATRICES, "{}.npy".format(phone_id))

GET_DATA=False
if GET_DATA:
	phones["numpy_X"] = phones[["phone_id", "underlying_phoneme"]].apply(save_numpy_mat, axis=1)
	phones.to_csv("timit_data.csv")
else:
	phones["pred"], phones["conf"], = \
	zip(*phones[["phone_id", "underlying_phoneme"]].apply(get_prediction, axis=1))
	phones.to_csv("reclassified_timit_data.csv")
