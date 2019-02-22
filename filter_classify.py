import os
import sys
import itertools
import pickle
import argparse
import random 
from math import ceil
from time import sleep
import numpy as np
import settings as s
import keras
from keras import layers, regularizers
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

BATCH_SIZE = 32
TEST_SPLIT = 0.8
MIN_LENGTH = 9
EPOCHS = 10
PHONEME_TO_TRAIN = "p"
FEATURE_SIZE=63
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

def evaluate_model(truth, predictions, classes):
    conf_mat = confusion_matrix(truth, predictions)
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix for /{PHONEME_TO_TRAIN}/")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, "{0:d}".format(conf_mat[i, j]),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    
    print(conf_mat)

def data_generator(data, batch_size=BATCH_SIZE):
    z_scores = np.load(os.path.join(s.OUTPUT_DIR, "zscores.npy"))
    length_order = list(data.keys())
    path_cache = {}
    while True:
        for length in length_order:
            if length <= MIN_LENGTH:
                continue
            for position in range(0, len(data[length]), batch_size):
                batch_x = []
                batch_y = []
                for phone in data[length][position:position+batch_size]:
                    if phone["x_path"] not in path_cache:
                        if not os.path.isfile(phone["x_path"]+".npy"):
                            if FEATURE_TYPE == "specgram":
                                phone_x = np.load(phone["x_path"])
                            elif FEATURE_TYPE == "AutoVOT":
                                phone_x = np.nan_to_num(np.genfromtxt(phone["x_path"], skip_header=1, filling_values=0))

                            phone_x = (phone_x - z_scores[0])/z_scores[1]
                            np.save(phone["x_path"]+".npy", phone_x)
                        else:
                            phone_x = np.load(phone["x_path"]+".npy")
                        path_cache[phone["x_path"]] = phone_x
                    else:
                        phone_x = path_cache[phone["x_path"]]
                    batch_x.append(phone_x)
                    y = np.zeros((1, len(allophone_mapping)))
                    y[:, allophone_mapping[phone["allophone"]]] = 1
                    #if phone["allophone"] in [f"{PHONEME_TO_TRAIN}rl", PHONEME_TO_TRAIN]:
                    #    y = 1
                    #else:
                    #    y = 0
                    batch_y.append(y)
                batch_x = np.stack(batch_x, axis=0)
                batch_y = np.vstack(batch_y)
                yield batch_x, batch_y

def sotc_size():
    file_directory = os.path.join(s.OUTPUT_DIR, "autovot_files_sotc")
    y_x_s = []
    for phone in os.listdir(file_directory):
        try:
            phoneme, measurable, type_of_measure, phone_id= phone.split("-")
        except:
            continue
        if phoneme != PHONEME_TO_TRAIN:
            continue
        x = np.nan_to_num(np.genfromtxt(os.path.join(file_directory, phone), skip_header=1, filling_values=0))
        y_x_s.append(x)

    return len(set(len(x) for x in y_x_s))
def sotc_generator():
    file_directory = os.path.join(s.OUTPUT_DIR, "autovot_files_sotc")
    z_scores = np.load(os.path.join(s.OUTPUT_DIR, "zscores.npy"))
    y_x_s = []
    for phone in os.listdir(file_directory):
        try:
            phoneme, measurable, type_of_measure, phone_id= phone.split("-")
        except:
            continue
        if phoneme != PHONEME_TO_TRAIN:
            continue
        if measurable == "yes":
            y = 1
        else:
            y = 0
        x = np.nan_to_num(np.genfromtxt(os.path.join(file_directory, phone), skip_header=1, filling_values=0))
        x = (x-z_scores[0])/z_scores[1]
        y_x_s.append((x,y))
    x_dic = {}
    y_dic = {}

    y_x_s.sort(key=lambda x: len(x[0]))

    for (x, y) in y_x_s:
        if len(x) in x_dic:
            x_dic[len(x)].append(x)
            y_dic[len(x)].append(y)
        else:
            x_dic[len(x)] = [x]
            y_dic[len(x)] = [y]
    while True:
        for (_, x), (_, y) in zip(sorted(x_dic.items(), key=lambda x:x[0]), sorted(y_dic.items(), key=lambda x:x[0])):
            print(np.mean(np.stack(x, axis=0), axis=0))
            yield np.stack(x, axis=0), np.vstack(y)

data = {}
allophone_mapping = {}
if FEATURE_TYPE == "specgram":
    file_dir = os.path.join(s.OUTPUT_DIR, "filter_bank_files")
elif FEATURE_TYPE == "AutoVOT":
    file_dir = os.path.join(s.OUTPUT_DIR, "autovot_files")
else:
    raise KeyError("Invalid FEATURE_TYPE")

rand = random.Random(69)
if os.path.isfile(os.path.join(s.OUTPUT_DIR, "test_pickle.pickle")) and \
        os.path.isfile(os.path.join(s.OUTPUT_DIR, "train_pickle.pickle")):

    with open(os.path.join(s.OUTPUT_DIR, "test_pickle.pickle"), "rb") as test_pick, open(os.path.join(s.OUTPUT_DIR, "train_pickle.pickle"), "rb") as train_pick:
        train_data = pickle.load(train_pick)
        test_data = pickle.load(test_pick)

    for d in [test_data, train_data]:
        for l in d:
            new_list = []
            for i, x in enumerate(d[l]):
                if POSITION != "no_pos" and x["position"] != POSITION:
                    continue
                allophone = x["allophone"]
                if allophone not in allophone_mapping:
                    allophone_mapping[allophone] = len(allophone_mapping)
                new_list.append(x)
            d[l] = new_list
else:
    for phone in os.listdir(file_dir):
        if ".npy" in phone or "ax-h" in phone:
            continue 
        phoneme, allophone, pos, path, phone_id = phone.split("-")
        if phoneme in s.EXTRA_PHONES and rand.random() < 0.05:
            allophone = "other"
        else:
            if POSITION != "no_pos" and pos != POSITION:
                continue

            if phoneme == "t+":
                phoneme = "t"

            if phoneme != PHONEME_TO_TRAIN:
                continue 
        if FEATURE_TYPE == "specgram":
            x = np.load(os.path.join(file_dir, phone))
        elif FEATURE_TYPE == "AutoVOT":
            x = np.nan_to_num(np.genfromtxt(os.path.join(file_dir, phone), skip_header=1, filling_values=0))

        if allophone not in allophone_mapping:
            allophone_mapping[allophone] = len(allophone_mapping)
        length = x.shape[0]
        if length not in data:
            data[length] = []

        data[length].append({
            "allophone":allophone,
            "position":pos,
            "phoneme":phoneme,
            "path":path,
            "x_path":os.path.join(file_dir, phone),
            "speaker":path.split("_")[1][1:],
            "phone_id":phone_id.strip(".npy")
            })
    print("Data generated")
    print(allophone_mapping)
    size_of_lengths = sorted([(k, len(v)) for k, v in data.items()])
    train_data = {}
    test_data = {}
    for l in data:
        test_data[l] = []
        train_data[l] = []
        for x in data[l]:
            if x["speaker"] in TEST_SPEAKERS:
                test_data[l].append(x)
            else:
                train_data[l].append(x)
    print("Data sorted")

    with open(os.path.join(s.OUTPUT_DIR, "test_pickle.pickle"), "wb") as test_pick, open(os.path.join(s.OUTPUT_DIR, "train_pickle.pickle"), "wb") as train_pick:
        pickle.dump(train_data, train_pick)
        pickle.dump(test_data, test_pick)

def t_model():
    model = keras.Sequential()
    model.add(layers.Conv1D(64, 10, input_shape=(None, FEATURE_SIZE), activation="relu"))
    model.add(layers.Conv1D(64, 10, input_shape=(None, FEATURE_SIZE), activation="relu"))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(40, activation="relu", kernel_regularizer=regularizers.l2(0)))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(1, activation="sigmoid"))
    #model.add(layers.Dense(len(allophone_mapping), activation="softmax"))
    return model

def d_model():
    model = keras.Sequential()
    model.add(layers.Conv1D(64, 20, input_shape=(None, FEATURE_SIZE), activation="relu"))
    model.add(layers.Conv1D(64, 3, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(128, 3, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Conv1D(128, 3, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(40, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(len(allophone_mapping), activation="softmax"))
    return model


model = d_model()
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["categorical_accuracy"])
print("Model compiled")
train_size = sum([ceil(len(v)/BATCH_SIZE) for length, v in train_data.items() if length > MIN_LENGTH])
validation_size = sum([ceil(len(v)/BATCH_SIZE) for length, v in test_data.items() if length > MIN_LENGTH])
print("Sizes calculated")
y_s = []
for i, (x, y) in enumerate(data_generator(test_data)):
    if i >= validation_size:
        break
    y_s.append(y)
test_y = np.vstack(y_s)
classes = [k for k, v in sorted(list(allophone_mapping.items()), key=lambda x: x[1])]
model.fit_generator(data_generator(train_data), steps_per_epoch=train_size,\
        validation_data=data_generator(test_data), validation_steps=validation_size, epochs=EPOCHS)
pred = model.predict_generator(data_generator(test_data), validation_size)
evaluate_model(np.argmax(test_y, axis=1), np.argmax(pred, axis=1), classes)
#sotc_dir = os.path.join(s.OUTPUT_DIR, "autovot_files_sotc")
#y_s = []
#for phone in os.listdir(sotc_dir):
#    try:
#        phoneme, measurable, type_of_measure, phone_id= phone.split("-")
#    except:
#        continue
#    if phoneme != PHONEME_TO_TRAIN:
#        continue
#    if measurable == "yes":
#        y_s.append(1)
#    else:
#        y_s.append(0)
#test_y = np.vstack(y_s)
#steps = sotc_size()
#pred = model.predict_generator(sotc_generator(), steps=steps)
#evaluate_model(test_y, (pred > 0.25).astype(int), classes)
#evaluate_model(test_y, (pred > 0.5).astype(int), classes)
#evaluate_model(test_y, (pred > 0.75).astype(int), classes)
