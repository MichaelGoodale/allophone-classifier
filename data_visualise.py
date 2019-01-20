import os
import sys
import pandas as  pd
import settings as s
import matplotlib.pyplot as plt
import 

MINIMUM_CONF = -1000000.0

plt.style.use('ggplot')
prediction_path = os.path.join(s.OUTPUT_DIR, "real_pred")
data = pd.read_csv(prediction_path, sep=" ", names=["phoneme", "allophone", "word_pos", "label", "id", "conf", "begin", "end"])
colours = ["blue", "red", "green", "orange", "yellow"]

for stop in s.STOPS:
    stop_data = data[(data["phoneme"] == stop) & (data["conf"] != MINIMUM_CONF)]
    f, axarr = plt.subplots(3)
    for i, allophone in enumerate(s.POSSIBLE_ALLOPHONES[stop]):
        allo = stop_data[stop_data["allophone"] == allophone]
        axarr[0].scatter(allo["conf"], allo["begin"], c=colours[i], label=allophone)
        axarr[1].scatter(allo["begin"], allo["end"], c=colours[i], label=allophone)
        axarr[2].scatter(allo["conf"], allo["end"], c=colours[i], label=allophone)

    axarr[0].set_xlabel("conf")
    axarr[1].set_xlabel("begin")
    axarr[2].set_xlabel("conf")
    axarr[0].set_ylabel("begin")
    axarr[1].set_ylabel("end")
    axarr[2].set_ylabel("end")
    plt.legend()
    plt.show()

