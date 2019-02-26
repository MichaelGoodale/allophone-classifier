import os 
import csv
import sys
import textgrid
import numpy as np
from keras.models import load_model
import argparse 
import subprocess
import multiprocessing
import itertools
import settings as s

N_CPU = multiprocessing.cpu_count()
TEMP_DIR = os.path.join(s.OUTPUT_DIR, "classifier_temporary")

PRAAT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "praat_barren")
OPENSAUCE_FEATURES = ["snackF0", "praatF0", "shrF0", "reaperF0", "snackFormants", "praatFormants", "SHR"]


INPUTLIST = os.path.join(TEMP_DIR, "inputlist")
OUTPUTLIST = os.path.join(TEMP_DIR, "outputlist")
PREDICTIONS = os.path.join(TEMP_DIR, "predictions")
VOT_DIR = os.path.join(TEMP_DIR, "autovot_files")
NUMPY_MATRICES = os.path.join(TEMP_DIR, "numpy_matrices")

for path in [VOT_DIR, NUMPY_MATRICES]:
    if not os.path.exists(path):
        os.makedirs(path)


def get_glottal_features(path, memo, output_name):
    if path in memo:
        return memo[path]
    orig_dir = os.getcwd()

    os.chdir("../opensauce-python")
    subprocess.run(["python", "-m", "opensauce", path, "--praat-path", PRAAT_PATH, \
            "--no-textgrid", "--use-pyreaper", \
            "--measurements"] + OPENSAUCE_FEATURES + ["-o", output_name])
    X = np.genfromtxt(output_name, skip_header=1)[:, 1:]
    memo[path] = X
    os.chdir(orig_dir)
    return X


def _calculate_features(i, phone_list):
    inputlist = INPUTLIST+str(i)
    outputlist = OUTPUTLIST+str(i)
    predictions = PREDICTIONS+str(i)

    with open(inputlist, "w") as input_f, open(outputlist, "w") as output_f:
        for begin, end, path, phone_id in phone_list:
            input_f.write(f"\"{path}\" {begin:3f} {end:3f} {begin:3f} {end:3f} [seconds]\n")
            output_path = os.path.join(VOT_DIR, f"{phone_id}")
            output_f.write(output_path+"\n")
    subprocess.run([os.path.join(s.PATH_TO_AUTOVOT, "VotFrontEnd2"), inputlist, outputlist, "null"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([os.path.join(s.PATH_TO_AUTOVOT, "VotDecode"), "-pos_only", "-output_predictions", predictions, outputlist, "null", s.CLASSIFIER], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    memo = {}
    with open(predictions) as pred_f:
        for (begin, end, path, phone_id), line in zip(phone_list, pred_f):
            vot_conf, vot_begin, vot_end = line.split()
            matrix = np.genfromtxt(os.path.join(VOT_DIR, str(phone_id)), skip_header=1)
            matrix_file = os.path.join(NUMPY_MATRICES, f"{phone_id}.npy")

            begin_ms = int(begin*1000)
            end_ms = int(end*1000)
            X = get_glottal_features(path, memo, f"output_{i}")[begin_ms:end_ms+1]

            if X.shape[0] > matrix.shape[0]:
                X = X[:matrix.shape[0]]

            matrix = np.hstack((matrix, X))
            matrix = np.insert(matrix, 85, [[vot_conf], [vot_begin], [vot_end]], axis=1)
            np.save(matrix_file, matrix)

def get_features(phone_list):
    '''Takes in a list of 4-tuples formatted with input, output, path and phone_id'''
    n_phones = len(phone_list)
    list_size = int(n_phones/(N_CPU-1))
    sub_lists = [phone_list[i:i+list_size] for i in range(0, n_phones, list_size)]

    #Ensure that no lists have overlapping paths
    represented_paths = [set(x[2] for x in l) for l in sub_lists]
    for i, j in itertools.combinations(range(len(sub_lists)), 2):
        intersect = represented_paths[i].intersection(represented_paths[j])
        if len(intersect) != 0:
            for path in intersect:
                sub_lists[j] = sub_lists[j]+[x for x in sub_lists[i] if x[2] == path]
                sub_lists[i] = [x for x in sub_lists[i] if x[2] != path]
                represented_paths[i].remove(path)
    jobs = []
    for i in range(len(sub_lists)):
        p = multiprocessing.Process(target=_calculate_features, args=(i, sub_lists[i]))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()

input_phones = []
with open(os.path.join(s.OUTPUT_DIR, "timit_data.csv"), "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader): 
        if i > 23:
            break
        input_phones.append((float(row["window_begin"]), float(row["window_end"]), row["path"]+".WAV", i))
get_features(input_phones)
print("Feature extraction completed")
model = load_model("outputmodel.hd5")
for _, _, _, phone_id in input_phones:
    X = np.load(os.path.join(NUMPY_MATRICES, f"{phone_id}.npy"))
    print(model.predict(X[None, ...]))
