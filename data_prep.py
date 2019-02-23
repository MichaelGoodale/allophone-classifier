import os 
import csv
import sys
import subprocess
import time
import numpy as np
import settings as s
from data_classes import Phone, same_place, Sentence

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
        if i > 1000:
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

with open(predictions) as pred_f:
    for i, line in enumerate(pred_f):
        vot_conf, vot_begin, vot_end = line.split()
        phone_dictionaries[i]["vot_conf"] = vot_conf
        phone_dictionaries[i]["vot_begin"] = vot_begin
        phone_dictionaries[i]["vot_end"] = vot_end
        matrix = phone_dictionaries[i]["vot_file"]
