import os
import sys
import pandas as  pd
import settings as s
import matplotlib.pyplot as plt

MINIMUM_CONF = -1000000.0

def get_values(path):
    with open(os.path.join(s.OUTPUT_DIR, "autovot_files", path)) as f:
        return int(f.readline().split(" ")[0])

plt.style.use('ggplot')
prediction_path = os.path.join(s.OUTPUT_DIR, "real_pred")
data = pd.read_csv(prediction_path, sep=" ", names=["phoneme", "allophone", "word_pos", "label", "id", "conf", "begin", "end"])
data["path"] = data['phoneme'].astype(str)+"-"+data['allophone']+"-"+data['word_pos']+"-"+data['label']+"-"+data['id' ].astype(str)
data["VOT"] = data["end"] - data["begin"]
data["length"] = data["path"].apply(get_values)
        

colours = ["blue", "red", "green", "orange", "yellow"]

for stop in s.STOPS:
    stop_data = data[(data["phoneme"] == stop) & (data["conf"] != MINIMUM_CONF)]
    f, axarr = plt.subplots(2,2)
    for i, allophone in enumerate(s.POSSIBLE_ALLOPHONES[stop]):
        allo = stop_data[stop_data["allophone"] == allophone]
        axarr[0, 0].scatter(allo["conf"], allo["begin"], c=colours[i], label=allophone)
        axarr[1, 0].scatter(allo["begin"], allo["end"], c=colours[i], label=allophone)
        axarr[0, 1].scatter(allo["conf"], allo["length"], c=colours[i], label=allophone)
        axarr[1, 1].scatter(allo["conf"], allo["VOT"], c=colours[i], label=allophone)

    axarr[0,0].set_xlabel("conf")
    axarr[1,0].set_xlabel("begin")
    axarr[0,1].set_xlabel("conf")
    axarr[1,1].set_xlabel("conf")
    axarr[0,0].set_ylabel("begin")
    axarr[1,0].set_ylabel("end")
    axarr[0,1].set_ylabel("length")
    axarr[1,1].set_ylabel("VOT")
    plt.legend()
    plt.show()

