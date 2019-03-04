import os 
import csv
import sys
import textgrid
import wave
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

PRAAT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "praat_barren")
OPENSAUCE_FEATURES = ["snackF0", "praatF0", "shrF0", "reaperF0", "snackFormants", "praatFormants", "SHR"]


INPUTLIST = os.path.join(TEMP_DIR, "inputlist")
OUTPUTLIST = os.path.join(TEMP_DIR, "outputlist")
PREDICTIONS = os.path.join(TEMP_DIR, "predictions")
AUDIO_DIR = os.path.join(TEMP_DIR, "audio")
VOT_DIR = os.path.join(TEMP_DIR, "autovot_files")
NUMPY_MATRICES = os.path.join(TEMP_DIR, "numpy_matrices")

for path in [VOT_DIR, NUMPY_MATRICES, AUDIO_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

def extract_segment(path, begin, end):
    output_path = os.path.abspath(os.path.join(AUDIO_DIR, "{}.wav".format(uuid1())))
    subprocess.run(["sox", path, output_path, "trim", str(begin), str(end-begin+0.01)])
    return output_path

def get_glottal_features(path, memo, output_name):
    if path in memo:
        return memo[path]

    orig_dir = os.getcwd()
    os.chdir("../opensauce-python")
    subprocess.run(["python", "-m", "opensauce", path, "--praat-path", PRAAT_PATH, \
            "--no-textgrid", "--use-pyreaper", \
            "--measurements"] + OPENSAUCE_FEATURES + ["-o", output_name])
    os.chdir(orig_dir)
    X = np.genfromtxt(os.path.join("../opensauce-python", output_name), skip_header=1)[:, 1:]
    memo[path] = X
    return X


def _calculate_features(i, phone_list):
    inputlist = INPUTLIST+str(i)
    outputlist = OUTPUTLIST+str(i)
    predictions = PREDICTIONS+str(i)
    phone_list = sorted(phone_list, key=lambda x: (x[2], x[0]))
    with open(inputlist, "w") as input_f, open(outputlist, "w") as output_f:
        for begin, end, path, phone_id in phone_list:
            input_f.write("\"{}\" {:3f} {:3f} {:3f} {:3f} [seconds]\n".format(path,begin,end,begin,end))
            output_path = os.path.join(VOT_DIR, str(phone_id))
            output_f.write(output_path+"\n")
    ret = subprocess.run([os.path.join(s.PATH_TO_AUTOVOT, "VotFrontEnd2"), inputlist, outputlist, "null"])
    subprocess.run([os.path.join(s.PATH_TO_AUTOVOT, "VotDecode"), "-pos_only", "-output_predictions", predictions, outputlist, "null", s.CLASSIFIER])
    memo = {}
    with open(predictions) as pred_f:
        for count, ((begin, end, path, phone_id), line) in enumerate(zip(phone_list, pred_f)):
            vot_conf, vot_begin, vot_end = line.split()
            matrix = np.genfromtxt(os.path.join(VOT_DIR, str(phone_id)), skip_header=1)
            matrix_file = os.path.join(NUMPY_MATRICES, "{}.npy".format(phone_id))
            matrix = np.insert(matrix, 63, [[vot_conf], [vot_begin], [vot_end]], axis=1)
            np.save(matrix_file, matrix)

def get_features(phone_list):
    '''Takes in a list of 4-tuples formatted with input, output, path and phone_id'''
    n_phones = len(phone_list)
    list_size = int(n_phones/(N_CPU-1))
    sub_lists = [phone_list[i:i+list_size] for i in range(0, n_phones, list_size)]

    #Ensure that no lists have overlapping paths
    represented_paths = [set(x[2] for x in l) for l in sub_lists]
    combinations = list(itertools.combinations(range(len(sub_lists)), 2))
    random.shuffle(combinations)
    for i, j in combinations:
        intersect = represented_paths[i].intersection(represented_paths[j])
        if len(intersect) != 0:
            for path in intersect:
                i, j = random.sample([i, j], 2)
                sub_lists[j] = sub_lists[j]+[x for x in sub_lists[i] if x[2] == path]
                sub_lists[i] = [x for x in sub_lists[i] if x[2] != path]
                represented_paths[i].remove(path)
    jobs = []
    sub_lists = [x for x in sub_lists if len(x) > 0]
    for i in range(len(sub_lists)):
        p = multiprocessing.Process(target=_calculate_features, args=(i, sub_lists[i]))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()

input_phones = []
phone_labels = []
phone_id = 0
for directory in os.listdir(s.BUCKEYE_DIR):
    for phone_file in [x for x in os.listdir(os.path.join(s.BUCKEYE_DIR, directory)) if x.endswith("phones")]:
        path = os.path.join(s.BUCKEYE_DIR, directory, phone_file)
        phone = "begin_of_file"
        time = 0 
        #max_time = float(subprocess.check_output(["soxi", "-D", path.replace(".phones", ".wav")])) - 0.005
        with wave.open(path.replace(".phones", ".wav"), 'r') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        max_time = duration - 0.005
        with open(path, 'r') as f:
            for line in f:
                if not line.startswith(' '):
                    continue
                line = line.split(';')[0]
                line = line.strip() 
                while '  ' in line:
                    line = line.replace('  ', ' ')
                old_time, old_phone = time, phone
                time, _, phone = line.split(' ')
                time = float(time)
                #if old_phone in ["tq", "dx", "t"]:
                begin = max(old_time - 0.050, 0)
                end = min(time + 0.100, max_time)
                if end - begin > 0.030:
                    input_phones.append((begin, end, path.replace("phones", "wav"), phone_id))
                    phone_labels.append(old_phone)
                    phone_id += 1
file_list = list(set(x[2] for x in input_phones))
file_list = random.sample(file_list, 4)
input_phones = [x for x in input_phones if x[2] in file_list]
get_features(input_phones)
print("Feature extraction completed")
model = load_model("outputmodel.hd5")

LABELS = ["t", "trl", "tcl", "-", "dx", "q", "other"]
file_zscores = {path: 
    {"n" : 0,
    "s_1" : np.zeros((1, 66), dtype=np.float64),
    "s_2" : np.zeros((1, 66), dtype=np.float64)}
    for path in set(x[2] for x in input_phones)}

for phone, (_, _, path, phone_id) in zip(phone_labels, input_phones):
    try:
        X = np.load(os.path.join(NUMPY_MATRICES, "{}.npy".format(phone_id)))
    except:
        continue
    X = np.nan_to_num(X)
    file_zscores[path]["n"] += X.shape[0]
    file_zscores[path]["s_1"] += X.sum(axis=0)
    file_zscores[path]["s_2"] += np.sum(np.square(X), axis=0)

for k in file_zscores:
    n = file_zscores[k]["n"] 
    s_1 = file_zscores[k]["s_1"] 
    s_2 = file_zscores[k]["s_2"] 
    file_zscores[k]["stds"] = np.sqrt((n/(n-1))*((s_2/n) - np.square(s_1/n)))
    file_zscores[k]["mean"] = s_1/n

with open('predictions_weee', 'w') as f:
    for phone, (_, _, path, phone_id) in zip(phone_labels, input_phones):
        if phone not in LABELS:
            continue
        X = (np.nan_to_num(X) - file_zscores[path]["mean"])/file_zscores[path]["stds"]
        pred = model.predict(X[None, ...])
        f.write('{} {} {}\n'.format(phone, LABELS[np.argmax(pred)], path))
