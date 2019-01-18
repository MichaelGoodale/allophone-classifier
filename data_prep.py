import os 
import subprocess
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
    for stop, next_stop in zip(stop_phones, stop_phones[1:]+[END_FAKE_STOP]):
        if combined_stop:
            combined_stop = False
            continue
        elif stop.end == next_stop.begin and same_place(stop.phone, next_stop.phone):
            #These are two components of the same stop
            new_stop = Stop(stop.begin, next_stop.end, next_stop.phone+"rl", sentence_path)
            data_dict[new_stop.underlying_stop].append(new_stop)
            combined_stop = True
        else:
            try:
                data_dict[stop.underlying_stop].append(stop)
            except (KeyError, ValueError):
                #Either this is a epethenic glottal stop before a vowel, or I'm not sure what it is
                continue
    return data_dict


stop_dictionary = {stop:[] for stop in s.STOPS}
for dialect_region in os.listdir(s.TIMIT_DIR):
    for speaker in os.listdir(os.path.join(s.TIMIT_DIR, dialect_region)):
        sentences = set(x.split('.')[0] for x in os.listdir(os.path.join(s.TIMIT_DIR, dialect_region, speaker)))
        for sentence in sentences:
            new_stop_dict = extract_wav_files(os.path.join(s.TIMIT_DIR, dialect_region, speaker, sentence))
            for stop in s.STOPS:
                stop_dictionary[stop].extend(new_stop_dict[stop]) 

print("DATA INFO:")
print([f"{stop}:{len(stop_dictionary[stop])}" for stop in s.STOPS])
for stop in s.STOPS:
    counts = {}
    for allophone in s.POSSIBLE_ALLOPHONES[stop]:
        counts[allophone] = len([x for x in stop_dictionary[stop] if x.phone == allophone])
    print(f"{stop}:{counts}")

with open(os.path.join(s.OUTPUT_DIR, f"inputlist"), "w") as input_f, open(os.path.join(s.OUTPUT_DIR, f"outputlist"), "w") as output_f:
    for stop in s.STOPS:
        for i, x in enumerate(stop_dictionary[stop][:100]):
            if x.duration/16000 < 0.025:
                continue
            file_info = "_".join(x.path.split('/')[-3:])

            input_f.write(f"\"{x.path}.WAV\" {x.begin/16000:3f} {x.end/16000:3f} {x.begin/16000:3f} {x.end/16000:3f} [seconds]\n")
            output_file = os.path.join(s.OUTPUT_DIR, "autovot_files", f"{stop}-{x.phone}-{file_info}-{i}")
            output_f.write(f"{output_file}\n")
subprocess.run([os.path.join(s.PATH_TO_AUTOVOT, "VotFrontEnd2"), "-verbose", "DEBUG", \
        os.path.join(s.OUTPUT_DIR, "inputlist"), os.path.join(s.OUTPUT_DIR, "outputlist"), "null"])
subprocess.run([os.path.join(s.PATH_TO_AUTOVOT, "VotDecode"), "-pos_only", "-verbose", "DEBUG", \
        "-output_predictions", os.path.join(s.OUTPUT_DIR, "predictions"), os.path.join(s.OUTPUT_DIR, "outputlist"), "null", "classifier/classifier"])
with open(os.path.join(s.OUTPUT_DIR, "outputlist"), "r") as stopnames_f, open(os.path.join(s.OUTPUT_DIR, "predictions"), "r") as pred_f, open(os.path.join(s.OUTPUT_DIR, "real_pred"), "w") as f:
    for stop, pred in zip(stopnames_f, pred_f):
        f.write("{} {}".format(stop.strip('\n').split('/')[-1], pred))
