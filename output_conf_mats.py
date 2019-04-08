import os
import sys
import csv
import itertools
import pickle
import argparse
import random 
from math import ceil
import datetime
import numpy as np
import settings as s
import keras
from keras import layers, regularizers
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

BATCH_SIZE = 32
TEST_SPLIT = 0.8
MIN_LENGTH = 50
EPOCHS = 10
PHONEME_TO_TRAIN = "t"
FEATURE_SIZE=66
FEATURE_TYPE="AutoVOT"
TEST_SPEAKERS = [ "DAB0", "WBT0", "ELC0", "TAS1", "WEW0", "PAS0", "JMP0", "LNT0", 
                  "PKT0", "LLL0", "TLS0", "JLM0", "BPM0", "KLT0", "NLP0", "CMJ0",
                  "JDH0", "MGD0", "GRT0", "NJM0", "DHC0", "JLN0", "PAM0", "MLD0"]

parser = argparse.ArgumentParser(description='Train allophone classifier')
parser.add_argument('--phoneme', default=PHONEME_TO_TRAIN, help='Phoneme to train')
parser.add_argument('--epochs', default=EPOCHS, type=int, help='Number of epochs to train for')
parser.add_argument('--feature_type', default=FEATURE_TYPE, help='Which feature to use')
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='Number of batches')
parser.add_argument('--position', default="no_pos", help='Word position')

args = parser.parse_args()
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
FEATURE_TYPE = args.feature_type
POSITION = args.position
PHONEME_TO_TRAIN = args.phoneme


def get_labels(phoneme):
    if phoneme == "t":
        labels = ["t", "trl", "tcl", "-", "dx", "q", "other"]
        allophone_classes = {
                "trl":"trl",
                "t":"t",
                "tcl":"tcl",
                "-":"-",
                "dx":"dx",
                "nx":"dx",
                "q":"q"
                }
        valid_phonemes = ["t", "n_t", "t_y"]
    elif phoneme == "d":
        labels = ["d", "drl", "dcl", "-", "dx", "other"]
        allophone_classes = {
                "drl":"drl",
                "d":"d",
                "dcl":"dcl",
                "-":"-",
                "dx":"dx",
                "nx":"dx",
                }
        valid_phonemes = ["d", "n_d", "d_y"]
    else:
       labels = [phoneme, "{}rl".format(phoneme), "{}cl".format(phoneme), "-", "other"]
       allophone_classes = {
       	phoneme: phoneme,
               "{}rl".format(phoneme): "{}rl".format(phoneme),
               "-":"-",
               "{}cl".format(phoneme): "{}cl".format(phoneme),
               }
       valid_phonemes = [phoneme]
    return labels, allophone_classes, valid_phonemes


def evaluate_model(truth, predictions, classes):
    conf_mat = confusion_matrix(truth, predictions)
    output_str = """
	\\begin{{figure}}[htb!]
        \\begin{{center}}
	\\newcommand\items{{{}}}   %Number of classes
	\\arrayrulecolor{{white}} %Table line colors
	\\noindent
	\\begin{{tabular}}{{cc*{{\items}}{{|E}}|}}
	\\multicolumn{{1}}{{c}}{{}} &\\multicolumn{{1}}{{c}}{{}} &\\multicolumn{{\\items}}{{c}}{{Predicted}} \\\\ \\hhline{{~*\\items{{|-}}|}}
	\\multicolumn{{1}}{{c}}{{}} &
	\\multicolumn{{1}}{{c}}{{}} &
	""".format(len(classes))

    for i, clss in enumerate(classes):
        output_str += " \\multicolumn{{1}}{{c}}{{\\rot{{{}}}}}".format(clss)
        if i < len(classes)-1:
            output_str += " &"
    output_str  += """
        \\\\ \\hhline{~*\\items{|-}|}
        \\multirow{\\items}{*}{\\rotatebox{90}{Actual}}
        """

    for i in range(conf_mat.shape[0]):
        output_str += "&{}" .format(classes[i])
        for j in range(conf_mat.shape[1]):
            output_str += "& {}" .format(conf_mat[i, j])
        output_str += "\\\\ \\hhline{~*\\items{|-}|}\n"
    output_str += """
                   \\end{{tabular}}
                   \\caption{{Confusion matrix for /{}/, accuracy={:.2f}\\%}}
                   \\end{{center}}
                   \\end{{figure}}
                   """.format(PHONEME_TO_TRAIN, 100*(sum(truth == predictions)/len(predictions)))
    print(output_str)

def data_generator(data, batch_size=BATCH_SIZE, do_shuffle=False, sample_weight=False):
    z_scores = np.load(os.path.join(s.OUTPUT_DIR, "zscores_new.npy"))
    rand = random.Random()
    length_order = list(data.keys())
    path_cache = {}
    number_of_classes = max(allophone_mapping.values())+1
    while True:
        if do_shuffle:
            rand.shuffle(length_order)
        for length in length_order:
            if length < MIN_LENGTH:
                continue
            if do_shuffle:
                new_data_list = rand.sample(data[length], len(data[length]))
            else:
                new_data_list = data[length]
            for position in range(0, len(new_data_list), batch_size):
                batch_x = []
                batch_y = []
                batch_sample_weights = []
                for phone in new_data_list[position:position+batch_size]:
                    if phone["x_path"] not in path_cache:
                        phone_x = np.nan_to_num(np.load(phone["x_path"]))
                        phone_x = (phone_x - z_scores[0])/z_scores[1]
                        path_cache[phone["x_path"]] = phone_x
                    else:
                        phone_x = path_cache[phone["x_path"]]
                    batch_x.append(phone_x)
                    y = np.zeros((1, number_of_classes))
                    y[:, allophone_mapping[phone["allophone"]]] = 1
                    batch_y.append(y)
                batch_x = np.stack(batch_x, axis=0)
                batch_y = np.vstack(batch_y)
                yield batch_x, batch_y


data = {}
allophone_mapping = {}

LABELS, ALLOPHONE_CLASSES, VALID_PHONEMES = get_labels(PHONEME_TO_TRAIN)
previous_phoneme = "x"
count = 0
rand_p = random.Random(39)
with open(os.path.join("timit_data.csv"), "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        phoneme = row["underlying_phoneme"].lower()
        if phoneme not in VALID_PHONEMES:
            if PHONEME_TO_TRAIN != "t" or row["phone"] != "q":
                continue

        if row["boundary"] == "3" or row["boundary"] == "2":
            pos = "initial"
        elif row["boundary"] == "1":
            pos = "syllable"
        else:
            pos = "medial"

        if POSITION != "no_pos" and pos != POSITION:
            continue 

        allophone = row["phone"]
        if allophone not in allophone_mapping:
            allophone_mapping[allophone] = len(allophone_mapping)
        x = np.nan_to_num(np.load(row["numpy_X"]))

        length = x.shape[0]
        if length not in data:
            data[length] = []

        data[length].append({
            "allophone":allophone,
            "position":pos,
            "phoneme":phoneme,
            "x_path":row["numpy_X"],
            "speaker":row["path"].split("/")[-2][1:],
            })
for k, v in allophone_mapping.items():
    if k not in ALLOPHONE_CLASSES:
        allophone_mapping[k] = LABELS.index("other")
    else:
        allophone_mapping[k]=LABELS.index(ALLOPHONE_CLASSES[k])
train_data = {}
test_data = {}
numbers_test = {k:0 for k in allophone_mapping.values()}
numbers_train = {k:0 for k in allophone_mapping.values()}
for l in data:
    test_data[l] = []
    train_data[l] = []
    for x in data[l]:
        if x["speaker"] in TEST_SPEAKERS:
            test_data[l].append(x)
            numbers_test[allophone_mapping[x["allophone"]]] += 1
        else:
            numbers_train[allophone_mapping[x["allophone"]]] += 1
            train_data[l].append(x)
train_size = sum([ceil(len(v)/BATCH_SIZE) for length, v in train_data.items()])
validation_size = sum([ceil(len(v)/BATCH_SIZE) for length, v in test_data.items()])
y_s = []
for i, (x, y) in enumerate(data_generator(test_data)):
    if i >= validation_size:
        break
    y_s.append(y)
test_y = np.vstack(y_s)
model = keras.models.load_model("{}_model_max.hd5".format(PHONEME_TO_TRAIN))
pred = model.predict_generator(data_generator(test_data), validation_size)
evaluate_model(np.argmax(test_y, axis=1), np.argmax(pred, axis=1), LABELS)
