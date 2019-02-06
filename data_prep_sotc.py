import os 
import subprocess
import time
import csv
import sys
import settings as s
from data_classes import Stop, same_place

#def get_path(filename):
#    for root, dirs, files in os.walk(s.SOTC_DIR):
#        if filename in files:
#            return os.path.join(root, filename)
#    print(f"Could not find {filename}")
#
#def resample(filename):
#    new_path = os.path.join("/tmp", os.path.basename(filename))
#    if not os.path.isfile(new_path):
#        subprocess.run(["sox", filename, "-c", "1", "-r", "16000", new_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#    return new_path
#
#END_FAKE_STOP = Stop(-1, -1, "x", "x")
#
inputlist = os.path.join(s.OUTPUT_DIR, "inputlist_sotc")
outputlist = os.path.join(s.OUTPUT_DIR, "outputlist_sotc")
predictions = os.path.join(s.OUTPUT_DIR, "predictions_sotc")
#
#with open(inputlist, "w") as input_f, open(outputlist, "w") as output_f:
#    with open('glasgow_stops_cleaned.csv', 'r') as f:
#        reader = csv.DictReader(f, delimiter=',')
#        for i, stop in enumerate(reader):
#            filename = stop['Filename'].split('_')[0]+".wav"
#            path = resample(get_path(filename))
#            end = float(stop["Phone_End"])
#            begin = float(stop["Phone_Start"])
#            duration = end - begin
#            phone = stop["Phone"]
#            word = stop["Word"]
#            measurable = stop["VOT_Measure"]
#            if duration < 0.03:
#                buf = 0.03 - duration
#                begin = max(0, begin-buf/2)
#                end = min(float(stop["Word_End"]), end+buf/2)
#                if (end-begin) < 0.03:
#                    continue
#            input_f.write(f"\"{path}\" {begin:3f} {end:3f} {begin:3f} {end:3f} [seconds]\n")
#            output_f.write(os.path.join(s.OUTPUT_DIR, "autovot_files", f"{phone}-{measurable}-{i}\n"))
#
#subprocess.run([os.path.join(s.PATH_TO_AUTOVOT, "VotFrontEnd2"), inputlist, outputlist, "null"])
#subprocess.run([os.path.join(s.PATH_TO_AUTOVOT, "VotDecode"), "-pos_only", "-output_predictions", predictions, outputlist, "null", s.CLASSIFIER])

with open(outputlist, "r") as stopnames_f, open(predictions, "r") as pred_f, open(os.path.join(s.OUTPUT_DIR, "real_pred_sotc"), "w") as f:
    for stop, pred in zip(stopnames_f, pred_f):
        f.write("{} {}".format(" ".join(stop.strip('\n').split("-")), pred))
