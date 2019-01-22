import os
import sys
import itertools
import pandas as pd
import numpy as np
import graphviz
import settings as s
from sklearn import svm
from sklearn import tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

MINIMUM_CONF = -1000000.0
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

for stop in s.STOPS:
    stop_data = data[(data["phoneme"] == stop) & (data["word_pos"] == "initial")]
    mask = np.random.rand(len(stop_data)) < 0.8
    y, classes = pd.factorize(stop_data.loc[:, "allophone"])
    X = stop_data[["conf", "VOT"]+feature_slice]

    train_y = y[mask]
    test_y = y[~mask]
    train_X = X[mask]
    test_X = X[~mask]

    #clf = tree.DecisionTreeClassifier(max_depth=5)
    clf = svm.SVC()
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
    conf_mat = confusion_matrix(test_y, pred)

    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix for /{stop}/")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
