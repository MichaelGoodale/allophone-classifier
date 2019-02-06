import os
import sys
import pickle
import itertools
import pandas as pd
import numpy as np
import graphviz
import settings as s
from sklearn import svm, tree
from sklearn.utils import class_weight, shuffle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import keras
from keras import layers
import matplotlib.pyplot as plt

MINIMUM_CONF = -1000000.0
OVERSAMPLING = True
TEST_SPLIT = 0.8

def plot_history(history):
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def get_length(path):
    path = os.path.join(s.OUTPUT_DIR, "autovot_files", path)
    with open(path) as f:
        length = int(f.readline().split(" ")[0])
    return length

def get_values(path):
    path = os.path.join(s.OUTPUT_DIR, "autovot_files", path)
    matrix = np.nan_to_num(np.genfromtxt(path, skip_header=1, filling_values=0))
    matrix = np.concatenate([np.mean(x,axis=0) for x in np.array_split(matrix, 20)])
    return matrix

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

prediction_path = os.path.join(s.OUTPUT_DIR, "real_pred")
feature_slice = [f"feat-{x}" for x in range(63*20)]
z_scores = {}
if not os.path.isfile(os.path.join(s.OUTPUT_DIR, "dataframe.pkl")):
    data = pd.read_csv(prediction_path, sep=" ", names=["phoneme", "allophone", "word_pos", "label", "id", "conf", "begin", "end"])
    data["path"] = data['phoneme'].astype(str)+"-"+data['allophone']+"-"+data['word_pos']+"-"+data['label']+"-"+data['id' ].astype(str)
    data["VOT"] = data["end"] - data["begin"]

    temp = list(zip(*data["path"].map(get_values)))
    for i, c in enumerate(feature_slice):
        data[c] = temp[i]
        z_scores[c] = (data[c].mean(), data[c].std(ddof=0))
        data[c] =(data[c] - data[c].mean())/data[c].std(ddof=0)
    data["length"] = data["path"].apply(get_length)
    data.to_pickle(os.path.join(s.OUTPUT_DIR, "dataframe.pkl"))
    with open(os.path.join(s.OUTPUT_DIR, "z_scores.pkl"), 'wb') as f:
        pickle.dump(z_scores, f)
else:
    data = pd.read_pickle(os.path.join(s.OUTPUT_DIR, "dataframe.pkl"))
    with open(os.path.join(s.OUTPUT_DIR, "z_scores.pkl"), 'rb') as f:
        z_scores = pickle.load(f)

sotc_prediction_path = os.path.join(s.OUTPUT_DIR, "real_pred_sotc")
if not os.path.isfile(os.path.join(s.OUTPUT_DIR, "sotc_dataframe.pkl")):
    sotc_data = pd.read_csv(sotc_prediction_path, sep=" ", names=["phoneme", "measurable", "id", "conf", "begin", "end"])
    sotc_data["path"] = sotc_data['phoneme'].astype(str)+"-"+sotc_data['measurable'].astype(str)+"-"+sotc_data['id'].astype(str)
    sotc_data["VOT"] = sotc_data["end"] - sotc_data["begin"]
    temp = list(zip(*sotc_data["path"].map(get_values)))
    for i, c in enumerate(feature_slice):
        sotc_data[c] = temp[i]
        sotc_datac[c] = (sotc_data[c] - z_scores[c][0])/z_scores[c][1]
    sotc_data["length"] = sotc_data["path"].apply(get_length)
    sotc_data.to_pickle(os.path.join(s.OUTPUT_DIR, "sotc_dataframe.pkl"))
else:
    sotc_data = pd.read_pickle(os.path.join(s.OUTPUT_DIR, "sotc_dataframe.pkl"))



for col in ["conf", "VOT", "length"]:
    data.loc[:, col] = (data[col] - data[col].mean())/data[col].std(ddof=0)
    sotc_data.loc[:, col] = (sotc_data[col] - data[col].mean())/data[col].std(ddof=0)

#data.loc[:, "allophone"] = data["allophone"].replace([f"{stop}rl" for stop in s.STOPS]+s.STOPS, "yes")
#data.loc[:, "allophone"] = data["allophone"].replace(["dx", "q"]+[f"{stop}cl" for stop in s.STOPS], "no")
data.loc[:, "allophone"] = data["allophone"].replace([f"{stop}rl" for stop in s.STOPS]+s.STOPS+["dx", "q"]+[f"{stop}cl" for stop in s.STOPS], "yes")

for stop in s.STOPS:
    stop_data = data[(data["phoneme"] == stop)]
    #stop_data = data[(data["allophone"] == "yes") | (data["allophone"] == "no")]
    o_data = data[(data["phoneme"] == "vowel") | (data["phoneme"] == "nasal") | (data["phoneme"] == "fric")]
    o_data.loc[:, "allophone"] = "no"
    stop_data = stop_data.append(o_data[np.random.rand(len(o_data)) < len(stop_data)/2.5/len(o_data)])
    mask = np.random.rand(len(stop_data)) < 0.8

    y, classes = pd.factorize(stop_data.loc[:, "allophone"])
    X = stop_data[feature_slice]
    X = X.values.reshape((-1, 20, 63))
    X = np.append(X, np.repeat(stop_data[["conf", "VOT", "length"]].values.reshape(-1, 1, 3), 20, axis=1), axis=2)

    train_y = y[mask]
    test_y = y[~mask]
    train_X = X[mask]
    test_X = X[~mask]

    if stop == "t":
        n_data = data[(data["phoneme"] == "t+")]
        n_y = n_data.loc[:, "allophone"].apply(lambda x: np.where(classes==x)[0][0])
        n_X = n_data[feature_slice]
        n_X = n_X.values.reshape((-1, 20, 63))
        n_X = np.append(n_X, np.repeat(n_data[["conf", "VOT", "length"]].values.reshape(-1, 1, 3), 20, axis=1), axis=2)
        train_y = np.concatenate((train_y, n_y))
        train_X = np.vstack((train_X, n_X))

    train_X, train_y = shuffle(train_X, train_y, random_state=0)

    #Resample
    if len(classes) == 2:
        uniques, counts = np.unique(train_y, return_counts=True)
        print(counts)

    if len(classes) != 2:
        train_y = keras.utils.to_categorical(train_y)
    model = keras.Sequential()
    model.add(layers.Conv1D(64, 3, input_shape=(20, 66), activation="relu"))
    model.add(layers.Conv1D(64, 3, activation="relu"))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(128, 3, activation="relu"))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dropout(rate=0.5))
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

    if len(classes) == 2:
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])
    else:
        model.add(layers.Dense(len(classes), activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["categorical_accuracy"])

    history = model.fit(train_X, train_y, epochs=10, batch_size=64, validation_split=0.05)
    if len(classes) == 2:
        pred = model.predict(test_X)
    else:
        pred = np.argmax(model.predict(test_X), axis=1)

    sotc_stop_data = sotc_data[sotc_data["phoneme"] == stop]
    sotc_X = sotc_stop_data[feature_slice]
    sotc_X = sotc_X.values.reshape((-1, 20, 63))
    sotc_X = np.append(sotc_X, np.repeat(sotc_stop_data[["conf", "VOT", "length"]].values.reshape(-1, 1, 3), 20, axis=1), axis=2)
    sotc_y = sotc_stop_data.loc[:, "measurable"].apply(lambda x: np.where(classes==x)[0][0])

    pred = model.predict(sotc_X)
    for x in [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95]:
        print(x)
        pred_x = np.copy(pred)
        pred_x[pred_x > x] = 1
        pred_x[pred_x <= x] = 0
        print(confusion_matrix(sotc_y, pred_x))
    #plot_history(history)
    #evaluate_model(test_y, pred, classes)
