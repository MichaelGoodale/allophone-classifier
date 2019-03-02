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
FEATURE_SIZE=68
SLICE=200
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

def plot_history(history):
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("plots/{}-{}-{}epochs-accuracy".format(datetime.datetime.now(), PHONEME_TO_TRAIN, EPOCHS))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("plots/{}-{}-{}epochs-loss".format(datetime.datetime.now(), PHONEME_TO_TRAIN, EPOCHS))

def evaluate_model(truth, predictions, classes):
    conf_mat = confusion_matrix(truth, predictions)
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix for /{}/".format(PHONEME_TO_TRAIN))
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
    plt.savefig("plots/{}-{}-{}epochs-conf_mat".format(datetime.datetime.now(), PHONEME_TO_TRAIN, EPOCHS))
    print(conf_mat)

def data_generator(data, batch_size=BATCH_SIZE, do_shuffle=False, sample_weight=False):
    z_scores = np.load(os.path.join(s.OUTPUT_DIR, "zscores.npy"))
    rand = random.Random(32)
    length_order = list(data.keys())
    path_cache = {}
    number_of_classes = max(allophone_mapping.values())+1
    while True:
        if do_shuffle:
            rand.shuffle(length_order)
        for length in length_order:
            if length - SLICE < MIN_LENGTH:
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
                        phone_x = phone_x[:, :FEATURE_SIZE-3]
                        phone_x = np.insert(phone_x, FEATURE_SIZE-3, [[row["vot_conf"]], [row["vot_begin"]], [row["vot_end"]]], axis=1)
                        phone_x = phone_x[:-SLICE, :]
                        path_cache[phone["x_path"]] = phone_x
                    else:
                        phone_x = path_cache[phone["x_path"]]
                    batch_sample_weights.append(train_sample_weights[allophone_mapping[phone["allophone"]]])
                    batch_x.append(phone_x)
                    y = np.zeros((1, number_of_classes))
                    y[:, allophone_mapping[phone["allophone"]]] = 1
                    batch_y.append(y)
                batch_x = np.stack(batch_x, axis=0)
                batch_y = np.vstack(batch_y)
                batch_sample_weights = np.array(batch_sample_weights)
                if not sample_weight:
                    yield batch_x, batch_y
                else:
                    yield batch_x, batch_y, batch_sample_weights

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

LABELS, ALLOPHONE_CLASSES, VALID_PHONEMES = get_labels(PHONEME_TO_TRAIN)
previous_phoneme = "x"
count = 0
with open(os.path.join(s.OUTPUT_DIR, "timit_data.csv"), "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        if i % 50000 == 0:
            print("{}/224017".format(i))
        phoneme = row["phoneme"].lower()

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
            "vot_conf":(float(row["vot_conf"])-88.0977)/515.34,
            "vot_begin":(float(row["vot_begin"])-30.28)/43.5,
            "vot_end":(float(row["vot_end"])-68.447)/75.344,
            "phoneme":phoneme,
            "x_path":row["numpy_X"],
            "speaker":row["path"].split("/")[-2][1:],
            })
print(count)
print("Data generated")
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
print("Data sorted")

#print(f"Train numbers:{numbers_train}")
#print(f"Test numbers:{numbers_test}")


train_sample_weights = {v:0 for k, v in allophone_mapping.items()}
total = 0
for l in train_data:
    for x in train_data[l]:
        train_sample_weights[allophone_mapping[x["allophone"]]] += 1
        total += 1
for k, v in train_sample_weights.items():
    if v != 0:
        x = np.log(total/v)
    else:
        x = 1.0
    train_sample_weights[k] = x if x > 1.0 else 1.0

def new_model():
    model = keras.Sequential()
    model.add(layers.GaussianNoise(0.5, input_shape=(None, FEATURE_SIZE)))
    model.add(layers.Conv1D(256, 5, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.1))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(512, 5, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.1))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(1000, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(500, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(max(allophone_mapping.values())+1, activation="softmax"))
    return model

model = new_model()
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=True)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["categorical_accuracy"])
model.summary()
print("Model compiled")
train_size = sum([ceil(len(v)/BATCH_SIZE) for length, v in train_data.items() if length - SLICE > MIN_LENGTH])
validation_size = sum([ceil(len(v)/BATCH_SIZE) for length, v in test_data.items() if length - SLICE > MIN_LENGTH])
print("Sizes calculated")
y_s = []
for i, (x, y) in enumerate(data_generator(test_data)):
    if i >= validation_size:
        break
    y_s.append(y)
test_y = np.vstack(y_s)
history = model.fit_generator(data_generator(train_data, do_shuffle=True, sample_weight=False), steps_per_epoch=train_size,\
            validation_data=data_generator(test_data), validation_steps=validation_size, epochs=EPOCHS)
model.save("outputmodel.hd5")
plot_history(history)
pred = model.predict_generator(data_generator(test_data), validation_size)
evaluate_model(np.argmax(test_y, axis=1), np.argmax(pred, axis=1), LABELS)
