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
from keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

BATCH_SIZE = 32
TEST_SPLIT = 0.8
MIN_LENGTH = 9
EPOCHS = 10
PHONEME_TO_TRAIN = "p"
FEATURE_SIZE=63
FEATURE_TYPE="AutoVOT"

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

def calculate_z_score(data, feature_size=FEATURE_SIZE):
    feats = np.empty((0, feature_size))
    for i, (length, list_of_phones) in enumerate(data.items()):
        feats = np.vstack((feats, np.vstack([x["X"] for x in list_of_phones])))
    output = np.array([np.mean(feats, axis=0), np.std(feats, axis=0)])
    np.save(os.path.join(s.OUTPUT_DIR, "zscores"), output)
    for length in data:
        for i, phone in enumerate(data[length]):
            data[length][i]["X"] = (phone["X"] - output[0])/output[1]
    return data

def data_generator(data, batch_size=BATCH_SIZE):
    length_order = list(data.keys())
    while True:
        for length in length_order:
            if length <= MIN_LENGTH:
                continue
            for position in range(0, len(data[length]), batch_size):
                batch_x = np.empty((0, length, FEATURE_SIZE))
                batch_y = np.empty((0, len(allophone_mapping)))
                for phone in data[length][position:position+batch_size]:
                    batch_x = np.append(batch_x, phone["X"][None], axis=0)
                    y = np.zeros((1, len(allophone_mapping)))
                    y[:, allophone_mapping[phone["allophone"]]] = 1
                    batch_y = np.append(batch_y, y, axis=0)
                yield batch_x, batch_y

data = {}
allophone_mapping = {}
if FEATURE_TYPE == "specgram":
    file_dir = os.path.join(s.OUTPUT_DIR, "filter_bank_files")
elif FEATURE_TYPE == "AutoVOT":
    file_dir = os.path.join(s.OUTPUT_DIR, "autovot_files")
else:
    raise KeyError("Invalid FEATURE_TYPE")

rand = random.Random(69)

for phone in os.listdir(file_dir):
    if "ax-h" in phone:
        continue 
    phoneme, allophone, pos, path, phone_id = phone.split("-")
    if phoneme in s.EXTRA_PHONES and rand.random() < 0.25:
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
        "X":x,
        "allophone":allophone,
        "position":pos,
        "phoneme":phoneme,
        "path":path,
        "phone_id":phone_id.strip(".npy")
        })
print("Data generated")
print(allophone_mapping)
size_of_lengths = sorted([(k, len(v)) for k, v in data.items()])
data = calculate_z_score(data)
print("Data Z_Scored")
train_data = {}
test_data = {}
for l in data:
    rand.shuffle(data[l])
    train_data[l] = data[l][:int(len(data[l])*TEST_SPLIT)]
    test_data[l] = data[l][int(len(data[l])*TEST_SPLIT):]



def t_model():
    model = keras.Sequential()
    model.add(layers.Conv1D(64, 3, input_shape=(None, FEATURE_SIZE), activation="relu"))
    model.add(layers.Conv1D(64, 3, activation="relu"))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(128, 3, activation="relu"))
    model.add(layers.Conv1D(128, 3, activation="relu"))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(len(allophone_mapping), activation="softmax"))
    return model

def d_model():
    model = keras.Sequential()
    model.add(layers.Conv1D(48, 3, input_shape=(None, FEATURE_SIZE), activation="relu"))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Conv1D(48, 3, activation="relu"))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(128, 3, activation="relu"))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Conv1D(128, 3, activation="relu"))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(40, activation="relu"))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(40, activation="relu"))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(len(allophone_mapping), activation="softmax"))
    return model


model = d_model()
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["categorical_accuracy"])
train_size = sum([ceil(len(v)/BATCH_SIZE) for length, v in train_data.items() if length > MIN_LENGTH])
validation_size = sum([ceil(len(v)/BATCH_SIZE) for length, v in test_data.items() if length > MIN_LENGTH])

y_s = []
for i, (x, y) in enumerate(data_generator(test_data)):
    if i >= validation_size:
        break
    y_s.append(y)
test_y = np.vstack(y_s)
classes = [k for k, v in sorted(list(allophone_mapping.items()), key=lambda x: x[1])]
model.fit_generator(data_generator(train_data), steps_per_epoch=train_size,\
        validation_data=data_generator(test_data), validation_steps=validation_size, epochs=EPOCHS)
pred = model.predict_generator(data_generator(test_data), steps=validation_size)

evaluate_model(np.argmax(test_y, axis=1), np.argmax(pred, axis=1), classes)
