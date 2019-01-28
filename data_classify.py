import os
import sys
import itertools
import pandas as pd
import numpy as np
import graphviz
import settings as s
from sklearn import svm, tree
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import keras
from keras import layers
import matplotlib.pyplot as plt

MINIMUM_CONF = -1000000.0
OVERSAMPLING = True
TEST_SPLIT = 0.8

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

prediction_path = os.path.join(s.OUTPUT_DIR, "real_pred")
feature_slice = [f"feat-{x}" for x in range(63*20)]
if not os.path.isfile(os.path.join(s.OUTPUT_DIR, "dataframe.pkl")):
    data = pd.read_csv(prediction_path, sep=" ", names=["phoneme", "allophone", "word_pos", "label", "id", "conf", "begin", "end"])
    data["path"] = data['phoneme'].astype(str)+"-"+data['allophone']+"-"+data['word_pos']+"-"+data['label']+"-"+data['id' ].astype(str)
    data["VOT"] = data["end"] - data["begin"]

    temp = list(zip(*data["path"].map(get_values)))
    for i, c in enumerate(feature_slice):
        data[c] = temp[i]
        data[c] =(data[c] - data[c].mean())/data[c].std(ddof=0)
    data["length"] = data["path"].apply(get_length)
    data.to_pickle(os.path.join(s.OUTPUT_DIR, "dataframe.pkl"))
else:
    data = pd.read_pickle(os.path.join(s.OUTPUT_DIR, "dataframe.pkl"))

for stop in ["t", "d"]:
    stop_data = data[(data["phoneme"] == stop)]
    o_data = data[(data["phoneme"] == "vowel") | (data["phoneme"] == "nasal") | (data["phoneme"] == "fric")]
    o_data.loc[(o_data["phoneme"] == "vowel") | (o_data["phoneme"] == "nasal") | (o_data["phoneme"] == "fric"), "allophone"] = "x"
    stop_data = stop_data.append(o_data[np.random.rand(len(o_data)) < len(stop_data)/2.5/len(o_data)])
    mask = np.random.rand(len(stop_data)) < 0.8
    y, classes = pd.factorize(stop_data.loc[:, "allophone"])
    X = stop_data[["conf", "VOT", "length"]+feature_slice]

    train_y = y[mask]
    test_y = y[~mask]
    train_X = X[mask]
    test_X = X[~mask]

    if stop == "t":
        n_data = data[(data["phoneme"] == "t+")]
        n_y = n_data.loc[:, "allophone"].apply(lambda x: np.where(classes==x)[0][0])
        n_X = n_data[["conf", "VOT", "length"]+feature_slice]
        train_y = np.concatenate((train_y, n_y))
        train_X = np.vstack((train_X, n_X))
        print(train_y.shape)
        print(train_X.shape)

    #clf = tree.DecisionTreeClassifier()
    #clf.fit(train_X, train_y)

    class_weights = class_weight.compute_class_weight('balanced',
            np.unique(train_y),
            train_y)
    d_class_weights = dict(enumerate(class_weights))
    train_y = keras.utils.to_categorical(train_y)
    model = keras.Sequential()
    model.add(layers.Dense(100, input_shape=(3+63*20,), activation="sigmoid"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(25, activation="sigmoid"))
    model.add(layers.Dense(5, activation="sigmoid"))
    model.add(layers.Dense(len(classes), activation="softmax"))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(train_X, train_y, epochs=10, batch_size=32, class_weight=d_class_weights, validation_split=0.1)
    pred=np.argmax(model.predict(test_X), axis=1)

    #pred = clf.predict(test_X)
    conf_mat = confusion_matrix(test_y, pred)
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
