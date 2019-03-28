import os 
import csv
import sys
import subprocess
import time
import datetime
import numpy as np
import settings as s
from data_classes import Phone, same_place, Sentence


LABELS = ["prev_syl_stress", "next_syl_stress", "syl_pos"]
LABELS = ["phone_id",
         "phone",
         "begin",
         "end",
         "duration",
         "internal_boundary",
         "window_begin",
         "window_end",
         "speech_rate",
         "underlying_phoneme",
         "path",
         "word",
         "stress",
         "boundary",
         "next_phone",
         "next_underlying_phoneme",
         "next_word",
         "next_stress",
         "next_boundary",
         "prev_phone",
         "prev_underlying_phoneme",
         "prev_word",
         "prev_stress",
         "prev_boundary"]
phones = []
for dialect_region in sorted(os.listdir(s.TIMIT_DIR)):
    for speaker in sorted(os.listdir(os.path.join(s.TIMIT_DIR, dialect_region))):
        sentences = set(x.split('.')[0] for x in os.listdir(os.path.join(s.TIMIT_DIR, dialect_region, speaker)))
        sentences = sorted(list(sentences))
        for sentence in sentences:
            sentence = Sentence(dialect_region, speaker, sentence)
            phones += sentence.phone_list

with open(os.path.join(s.OUTPUT_DIR, "timit_phone_info.csv"), "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=LABELS)
    writer.writeheader()
    for i, phone in enumerate(filter(lambda x: x.underlying_phoneme.lower() in s.STOPS+["n_t", "t_y", "n_d", "d_y"] or x.phone == "q", phones)):
        row = {"phone_id": i,
               "phone":phone.phone,
               "begin":phone.begin,
               "end":phone.end,
               "duration":phone.duration,
               "internal_boundary":phone.internal_boundary,
               "underlying_phoneme": phone.underlying_phoneme,
               "window_begin":phone.window_begin,
               "window_end":phone.window_end,
               "speech_rate":phone.speech_rate,
               "path": phone.path,
               "word": phone.word,
               "stress": phone.stress,
               "boundary": phone.boundary}
        if phone.previous_phone != "START":
            row.update({"prev_phone":phone.previous_phone.phone,
               "prev_underlying_phoneme": phone.previous_phone.underlying_phoneme,
               "prev_word": phone.previous_phone.word,
               "prev_stress": phone.previous_phone.stress,
               "prev_boundary": phone.previous_phone.boundary})
        if phone.following_phone != "END":
            row.update({"next_phone":phone.following_phone.phone,
               "next_underlying_phoneme": phone.following_phone.underlying_phoneme,
               "next_word": phone.following_phone.word,
               "next_stress": phone.following_phone.stress,
               "next_boundary": phone.following_phone.boundary})
        writer.writerow(row)

