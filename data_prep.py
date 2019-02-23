import os 
import csv
import sys
import subprocess
import time
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
            "--no-textgrid", "--use-pyreaper"\
            "--measurements"] + OPENSAUCE_FEATURES + ["-o", "temp_out"])
    X = np.genfromtxt("temp_out", skip_header=1)[:, 1:]
    glottal_features_memo[path] = X
    os.chdir(orig_dir)
    return X

phones = []
for dialect_region in os.listdir(s.TIMIT_DIR):
    for speaker in os.listdir(os.path.join(s.TIMIT_DIR, dialect_region)):
        sentences = set(x.split('.')[0] for x in os.listdir(os.path.join(s.TIMIT_DIR, dialect_region, speaker)))
        for sentence in sentences:
            sentence = Sentence(dialect_region, speaker, sentence)
            phones += sentence.phone_list

inputlist = os.path.join(s.OUTPUT_DIR, "inputlist")
outputlist = os.path.join(s.OUTPUT_DIR, "outputlist")
predictions = os.path.join(s.OUTPUT_DIR, "predictions")
numpy_output = os.path.join(s.OUTPUT_DIR, "timit_matrices")

phone_dictionaries = []
with open(inputlist, "w") as input_f, open(outputlist, "w") as output_f:
    for i, x in enumerate(phones):
        if i > 10000:
            break
        input_f.write(f"\"{x.path}.WAV\" {x.window_begin:3f} {x.window_end:3f} {x.window_begin:3f} {x.window_end:3f} [seconds]\n")
        output_path = os.path.join(s.OUTPUT_DIR, "autovot_files", f"{i}")
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

subprocess.run([os.path.join(s.PATH_TO_AUTOVOT, "VotFrontEnd2"), inputlist, outputlist, "null"])
subprocess.run([os.path.join(s.PATH_TO_AUTOVOT, "VotDecode"), "-pos_only", "-output_predictions", predictions, outputlist, "null", s.CLASSIFIER])
err = 0
with open(predictions) as pred_f:
    for i, line in enumerate(pred_f):
        vot_conf, vot_begin, vot_end = line.split()
        phone_dictionaries[i]["vot_conf"] = vot_conf
        phone_dictionaries[i]["vot_begin"] = vot_begin
        phone_dictionaries[i]["vot_end"] = vot_end
        matrix = np.genfromtxt(phone_dictionaries[i]["vot_file"], skip_header=1)
        matrix_file = os.path.join(numpy_output, f"{i}.npy")
        window_begin = int(phone_dictionaries[i]["window_begin"]*1000)
        window_end = int(phone_dictionaries[i]["window_end"]*1000)
        X = get_glottal_features(phone_dictionaries[i]["path"]+".WAV")[window_begin:window_end+1]
        if X.shape[0] > matrix.shape[0]:
            X = X[:matrix.shape[0]]
        if X.shape[0] != matrix.shape[0]:
            phone_dictionaries[i]["incl"] = 0
            phone_dictionaries[i]["numpy_X"] = "na"
            err += 1
            continue
        else:
            phone_dictionaries[i]["incl"] = 1
        matrix = np.hstack((matrix, X))
        np.save(matrix_file, matrix)
        phone_dictionaries[i]["numpy_X"] = matrix_file
print(f"{err} stops excluded")
with open(os.path.join(s.OUTPUT_DIR, "timit_data.csv"), "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=phone_dictionaries[0].keys())
    writer.writeheader()
    writer.writerows(phone_dictionaries)
