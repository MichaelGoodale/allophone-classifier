import os 
import subprocess
import time
import settings as s
from data_classes import Stop, same_place

END_FAKE_STOP = Stop(-1, -1, "x", "x")

def extract_wav_files(sentence_path):
    with open(sentence_path+".PHN", "r") as f:
        sentence_phones = []
        for line in f:
            (begin, end, phone) = line.strip('\n').split(' ')
            (begin, end) = (int(begin), int(end))
            sentence_phones.append(Stop(begin, end, phone, sentence_path))

    #Keep only stops, which aren't part of an affricate
    stop_phones = [phone for phone, next_phone in zip(sentence_phones, sentence_phones[1:] +[END_FAKE_STOP]) \
            if phone.phone in s.STOP_ALLOPHONES and next_phone.phone not in ["jh", "ch"]]


    data_dict = {stop:[] for stop in s.STOPS}
    combined_stop = False
    err = 0
    for stop, next_stop in zip(stop_phones, stop_phones[1:]+[END_FAKE_STOP]):
        if combined_stop:
            combined_stop = False
            continue
        try:
            if stop.end == next_stop.begin and same_place(stop.phone, next_stop.phone) \
                    and stop.word.begin == next_stop.word.begin and stop.word.end == next_stop.word.end:
                #These are two components of the same stop
                new_stop = Stop(stop.begin, next_stop.end, next_stop.phone+"rl", sentence_path)
                data_dict[new_stop.underlying_stop].append(new_stop)
                combined_stop = True
            else:
                #Just to make sure the associated word is findable
                stop.word
                data_dict[stop.underlying_stop].append(stop)
        except (KeyError, ValueError):
            #Either this is a epethenic glottal stop before a vowel, or I'm not sure what it is
            if stop.phone != "q":
                err += 1
            continue

    return data_dict, err


stop_dictionary = {stop:[] for stop in s.STOPS}
err = 0
for dialect_region in os.listdir(s.TIMIT_DIR):
    for speaker in os.listdir(os.path.join(s.TIMIT_DIR, dialect_region)):
        sentences = set(x.split('.')[0] for x in os.listdir(os.path.join(s.TIMIT_DIR, dialect_region, speaker)))
        for sentence in sentences:
            new_stop_dict, new_err = extract_wav_files(os.path.join(s.TIMIT_DIR, dialect_region, speaker, sentence))
            err += new_err
            for stop in s.STOPS:
                stop_dictionary[stop].extend(new_stop_dict[stop]) 
print("DATA INFO:")
print([f"{stop}:{len(stop_dictionary[stop])}" for stop in s.STOPS])
for stop in s.STOPS:
    counts = {}
    for allophone in s.POSSIBLE_ALLOPHONES[stop]:
        counts[allophone] = len([x for x in stop_dictionary[stop] if x.phone == allophone])
    print(f"{stop}:{counts}")

inputlist = os.path.join(s.OUTPUT_DIR, "inputlist")
outputlist = os.path.join(s.OUTPUT_DIR, "outputlist")
predictions = os.path.join(s.OUTPUT_DIR, "predictions")

with open(os.path.join(s.OUTPUT_DIR, f"inputlist"), "w") as input_f, open(os.path.join(s.OUTPUT_DIR, f"outputlist"), "w") as output_f:
    for stop in s.STOPS:
        for i, x in enumerate(stop_dictionary[stop]):
            file_info = "_".join(x.path.split('/')[-3:])
            if x.duration/16000 < 0.03:
                buf = 0.03 - (x.duration/16000)
                alt_begin = max(0, x.begin/16000-buf/2)
                alt_end = min(x.sentence_length, x.end/16000+buf/2)
                if (alt_end-alt_begin) < 0.03:
                    err += 1
                    continue
                input_f.write(f"\"{x.path}.WAV\" {alt_begin:3f} {alt_end:3f} {alt_begin:3f} {alt_end:3f} [seconds]\n")
            else:
                input_f.write(f"\"{x.path}.WAV\" {x.begin/16000:3f} {x.end/16000:3f} {x.begin/16000:3f} {x.end/16000:3f} [seconds]\n")

            output_file = os.path.join(s.OUTPUT_DIR, "autovot_files", f"{stop}-{x.phone}-{x.word_position}-{file_info}-{i}")
            output_f.write(f"{output_file}\n")

print(f"{err} stops excluded in total")
subprocess.run([os.path.join(s.PATH_TO_AUTOVOT, "VotFrontEnd2"), inputlist, outputlist, "null"])
subprocess.run([os.path.join(s.PATH_TO_AUTOVOT, "VotDecode"), "-pos_only", "-output_predictions", predictions, outputlist, "null", s.CLASSIFIER])

with open(outputlist, "r") as stopnames_f, open(predictions, "r") as pred_f, open(os.path.join(s.OUTPUT_DIR, "real_pred"), "w") as f:
    for stop, pred in zip(stopnames_f, pred_f):
        f.write("{} {}".format(" ".join(stop.strip('\n').split('/')[-1].split("-")), pred))
