import os
import sys
import itertools
from random import shuffle
from time import sleep
import numpy as np
import settings as s
import keras
from keras import layers
import matplotlib.pyplot as plt

MINIMUM_CONF = -1000000.0
OVERSAMPLING = True
TEST_SPLIT = 0.8

def evaluate_model(truth, predictions, classes):
    conf_mat = confusion_matrix(truth, predictions)
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix for /{stop}/")
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

def calculate_z_score(data, feature_size=40):
    feats = np.empty((0, feature_size))
    for i, (length, list_of_phones) in enumerate(data.items()):
        feats = np.vstack((feats, np.vstack([x["X"].reshape((length, feature_size)) for x in list_of_phones])))
    output = np.array([np.mean(feats, axis=0), np.std(feats, axis=0)])
    np.save(os.path.join(s.OUTPUT_DIR, "zscores"), output)
    for length in data:
        for i, phone in enumerate(data[length]):
            data[length][i]["X"] = (phone["X"] - output[0])/output[1]
    return data


data = {}
allophone_mapping = {}
for phone in os.listdir(os.path.join(s.OUTPUT_DIR, "filter_bank_files")):
    phoneme, allophone, pos, path, phone_id = phone.split("-")
    if phoneme != "t" and phoneme != "t+":
        continue 

    x = np.load(os.path.join(s.OUTPUT_DIR, "filter_bank_files", phone))
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

size_of_lengths = sorted([(k, len(v)) for k, v in data.items()])
print(size_of_lengths)
data = calculate_z_score(data)

def data_generator(batch_size=32):
    length_order = list(data.keys())
    while True:
        shuffle(length_order)
        for length in length_order:
            if length <= 9:
                continue
            for position in range(0, len(data[length]), batch_size):
                batch_x = np.empty((0, length, 40))
                batch_y = np.empty((0, len(allophone_mapping)))
                for phone in data[length][position:position+batch_size]:
                    batch_x = np.append(batch_x, phone["X"][None], axis=0)
                    y = np.zeros((1, len(allophone_mapping)))
                    y[:, allophone_mapping[phone["allophone"]]] = 1
                    batch_y = np.append(batch_y, y, axis=0)
                yield batch_x, batch_y

def model1():
    model = keras.Sequential()
    model.add(layers.Conv1D(64, 3, input_shape=(None, 40), activation="relu"))
    model.add(layers.Conv1D(64, 3, activation="relu"))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(128, 3, activation="relu"))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(len(allophone_mapping), activation="softmax"))
    return model

model = model1()
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["categorical_accuracy"])
model.fit_generator(data_generator(), steps_per_epoch=sum([y for x,y in size_of_lengths]), epochs=3)

