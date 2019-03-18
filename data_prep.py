import os 
import csv
import sys
import subprocess
import time
import datetime
import numpy as np
import settings as s
from data_classes import Phone, same_place, Sentence

glottal_features_memo = {}
PRAAT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "praat_barren")
OPENSAUCE_FEATURES = ["snackF0", "praatF0", "shrF0", "reaperF0", "snackFormants", "praatFormants", "SHR"]

def get_glottal_features(path):
    if path in glottal_features_memo:
        return glottal_features_memo[path]
    orig_dir = os.getcwd()

    os.chdir("../opensauce-python")
    subprocess.run(["python", "-m", "opensauce", path, "--praat-path", PRAAT_PATH, \
            "--no-textgrid", "--use-pyreaper", \
            "--measurements"] + OPENSAUCE_FEATURES + ["-o", "temp_out"])
    X = np.genfromtxt("temp_out", skip_header=1)[:, 1:]
    glottal_features_memo[path] = X
    os.chdir(orig_dir)
    return X

phones = []
for dialect_region in sorted(os.listdir(s.TIMIT_DIR)):
    for speaker in sorted(os.listdir(os.path.join(s.TIMIT_DIR, dialect_region))):
        sentences = set(x.split('.')[0] for x in os.listdir(os.path.join(s.TIMIT_DIR, dialect_region, speaker)))
        sentences = sorted(list(sentences))
        for sentence in sentences:
            sentence = Sentence(dialect_region, speaker, sentence)
            phones += sentence.phone_list

inputlist = os.path.join(s.OUTPUT_DIR, "inputlist")
outputlist = os.path.join(s.OUTPUT_DIR, "outputlist")
predictions = os.path.join(s.OUTPUT_DIR, "predictions")
numpy_output = os.path.join(s.OUTPUT_DIR, "timit_noisy_matrices")

phone_dictionaries = []
with open(inputlist, "w") as input_f, open(outputlist, "w") as output_f:
    for i, x in enumerate(phones):
        input_f.write("\"{}.WAV\" {:3f} {:3f} {:3f} {:3f} [seconds]\n".format(x.path,x.window_begin,x.window_end,x.window_begin,x.window_end))
        output_path = os.path.join(s.OUTPUT_DIR, "autovot_files", str(i))
        output_f.write(output_path+"\n")
        phone_dictionaries.append({"phone":x.phone,
            "phoneme":x.underlying_phoneme,
            "path":x.path,
            "word":x.word,
            "stress":x.stress,
            "boundary":x.boundary,
            "begin":x.begin,
            "end":x.end,
            "window_begin":x.window_begin,
            "window_end":x.window_end,
            "vot_file":output_path})
        if i % 1000 == 0:
            print("{} {}/{}".format(datetime.datetime.now(), i, len(phones)))

subprocess.run([os.path.join(s.PATH_TO_AUTOVOT, "VotFrontEnd2"), inputlist, outputlist, "null"])
subprocess.run([os.path.join(s.PATH_TO_AUTOVOT, "VotDecode"), "-pos_only", "-output_predictions", predictions, outputlist, "null", s.CLASSIFIER])
with open(predictions) as pred_f:
    for i, line in enumerate(pred_f):
        vot_conf, vot_begin, vot_end = line.split()
        phone_dictionaries[i]["vot_conf"] = vot_conf
        phone_dictionaries[i]["vot_begin"] = vot_begin
        phone_dictionaries[i]["vot_end"] = vot_end
        matrix = np.genfromtxt(phone_dictionaries[i]["vot_file"], skip_header=1)
        matrix_file = os.path.join(numpy_output, "{}.npy".format(i))
        np.save(matrix_file, matrix)
        phone_dictionaries[i]["numpy_X"] = matrix_file
        if i % 1000 == 0:
            print("{} {}/{}".format(datetime.datetime.now(), i, len(phones)))

with open(os.path.join(s.OUTPUT_DIR, "timit_wew_data.csv"), "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=phone_dictionaries[0].keys())
    writer.writeheader()
    writer.writerows(phone_dictionaries)
