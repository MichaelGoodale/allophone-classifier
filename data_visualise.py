import os
import sys
import pandas as  pd
import settings as s
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

MINIMUM_CONF = -1000000.0
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

plt.style.use('ggplot')
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

pca=PCA(n_components=2)
x=pca.fit_transform(data[feature_slice].values)
data["pca1"] = x[:, 0]
data["pca2"] = x[:, 1]
colours = ["blue", "red", "green", "orange", "yellow"]

for stop in s.STOPS:
    stop_data = data[(data["phoneme"] == stop) & (data["conf"] != MINIMUM_CONF)]
    f, axarr = plt.subplots(2,2)
    for i, allophone in enumerate(s.POSSIBLE_ALLOPHONES[stop]):
        allo = stop_data[stop_data["allophone"] == allophone]
        axarr[0, 0].scatter(allo["conf"], allo["begin"], c=colours[i], label=allophone, alpha=0.5, s=5)
        axarr[1, 0].scatter(allo["pca1"], allo["pca2"], c=colours[i], label=allophone, alpha=0.5, s=5)
        axarr[0, 1].scatter(allo["conf"], allo["length"], c=colours[i], label=allophone, alpha=0.5, s=5)
        axarr[1, 1].scatter(allo["conf"], allo["VOT"], c=colours[i], label=allophone, alpha=0.5, s=5)

    axarr[0,0].set_xlabel("conf")
    axarr[1,0].set_xlabel("pca1")
    axarr[0,1].set_xlabel("conf")
    axarr[1,1].set_xlabel("conf")
    axarr[0,0].set_ylabel("begin")
    axarr[1,0].set_ylabel("pca2")
    axarr[0,1].set_ylabel("length")
    axarr[1,1].set_ylabel("VOT")
    plt.legend()
    plt.show()

