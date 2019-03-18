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
        for begin, end, path, phone_id, *rest in phone_list:
            input_f.write("\"{}\" {:3f} {:3f} {:3f} {:3f} [seconds]\n".format(path,begin,end,begin,end))
            output_path = os.path.join(VOT_DIR, str(phone_id))
            output_f.write(output_path+"\n")
    ret = subprocess.run([os.path.join(s.PATH_TO_AUTOVOT, "VotFrontEnd2"), inputlist, outputlist, "null"])
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
        sub_lists[len(sub_lists)-1] += sub_lists.pop()
    jobs = []
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
                begin = max(old_time - 0.50, 0)
                end = min(time + 0.15, max_time)
                if phone not in ["tq", "t", "dx"]:
                    continue
                if end - begin > 0.030:
                    input_phones.append((begin, end, path.replace("phones", "wav"), uuid1(), phone_id, phone))
                    phone_labels.append(phone_id)
                    phone_id += 1
phone_labels = set(random.Random(69).sample(phone_labels, 100))
input_phones = [x for x in input_phones if x[4] in phone_labels]
get_features(input_phones)
print("Feature extraction completed")
model = load_model("outputmodel.hd5")

LABELS = ["t", "trl", "tcl", "-", "dx", "q", "other"]
zscores = np.load("data/zscores.npy")[:, :63]
zscores = np.hstack((zscores, [[88.0977, 30.28, 68.447], [515.34, 43.5, 75.344]]))
correct = 0
incorrect = 0
cor_conf = 0
incor_conf = 0
high_conf = 0
high_cor = 0
with open('predictions_weee', 'w') as f:
    for phone_id in phone_labels:
        pred = np.zeros((1, 7))
        for i, (begin, end, path, uid, phone_id, phone) in enumerate([x for x in input_phones if x[4] == phone_id]):
            X = np.load(os.path.join(NUMPY_MATRICES, "{}.npy".format(uid)))
            X = np.nan_to_num(X)
            X = ((np.nan_to_num(X) - zscores[0])/zscores[1])
            pred[i, :] = model.predict(X[None, ...])
        pred = np.mean(pred, axis=0)
        argmax = np.argmax(pred)
        #argmax = np.unravel_index(np.argmax(pred, axis=None), pred.shape)[1]
        conf = np.max(pred)
        f.write('{} {} {} {} {} {}\n'.format(phone, LABELS[argmax], conf, path, begin, end))
        pred = LABELS[argmax]
        if conf > 0.50:
           high_conf += 1
           if phone == "t":
               if pred in ["t", "tcl", "trl"]:
                  high_cor +=1
           elif phone == "tq":
               if pred == "q":
                  high_cor +=1
           elif phone == "dx":
               if pred == "dx":
                  high_cor +=1

        if phone == "t":
            if pred in ["t", "tcl", "trl"]:
                correct += 1
                cor_conf += conf
            else:
                incorrect += 1
                incor_conf += conf
        elif phone == "tq":
            if pred == "q":
                correct += 1
                cor_conf += conf
            else:
                incorrect += 1
                incor_conf += conf
        elif phone == "dx":
            if pred == "dx":
                correct += 1
                cor_conf += conf
            else:
                incorrect += 1
                incor_conf += conf
print("{}/{}".format(correct, incorrect+correct))
print("Correct conf {}".format(cor_conf/correct))
print("Incorrect conf {}".format(incor_conf/incorrect))
print("{}/{}".format(high_cor, high_conf))
