import os 
import time
CORPORA_DIRECTORY = "../corpora"
TIMIT_DIR = os.path.join(CORPORA_DIRECTORY, "spade-TIMIT", "textgrid-wav")
OUTPUT_DIR = "data"

STOP_ALLOPHONES = ["b", "d", "g", "p", "t", "k", "bcl", "dcl", "gcl", "pcl", "tcl", "kcl", "dx", "q"]
STOPS = ["b", "d", "g", "p", "t", "k"]
END_FAKE_STOP = (-1, -1, '0', "fakefilepath")
UNDERLYING_STOP = {"b": "b", "bcl": "b", "brl": "b",
        "d": "d", "dcl": "d", "drl": "d",
        "g": "g", "gcl": "g", "grl": "g",
        "p": "p", "pcl": "p", "prl": "p",
        "t": "t", "tcl": "t", "trl": "t",
        "k": "k", "kcl": "k", "krl": "k"}
POSSIBLE_ALLOPHONES = {"b": ["b", "bcl", "brl"],
        "d": ["d", "dcl", "drl", "dx"],
        "g": ["g", "gcl", "grl"],
        "p": ["p", "pcl", "prl"],
        "t": ["t", "tcl", "trl", "q", "dx"],
        "k": ["k", "kcl", "krl"]}


timit_dictionary = {}
with open('TIMITDIC.TXT', 'r') as f:
    for line in f:
        if line.startswith(';'):
            continue
        word, pronounciation = line.strip('\n').split('  ')
        timit_dictionary[word] = pronounciation.strip('/').split(' ')

def same_place(closure, release):
    '''Takes two strings representing different phonemes and checks 
       if they make a valid closure release combination, e.g. bcl and b, dcl and d'''
    return closure[0] == release

def combine_release_closure(closure, release):
    '''Takes two tuples representing closure and release stops+timing, and combines them'''
    if len(closure) == 3:
        return (closure[0], release[1], release[2]+"rl")
    return (closure[0], release[1], release[2]+"rl", release[3])


def determine_underlying_representation(stop):
    found_word = False
    with open(stop[3]+".WRD", "r") as f:
        for line in f:
            (begin, end, word) = line.strip('\n').split(' ')
            (begin, end) = (int(begin), int(end))
            if begin <= stop[0] <= stop[1] <= end:
                found_word = True
                break
    if not found_word:
        raise KeyError(f"Stop, {stop} is not in any of the associated words")
    word_begin = begin
    word_end = end
    dict_pron = timit_dictionary[word]
    actual_pron_raw = []
    with open(stop[3]+".PHN", "r") as f:
        for line in f:
            (begin, end, phone) = line.strip('\n').split(' ')
            (begin, end) = (int(begin), int(end))
            if word_begin <= begin <= end <= word_end:
                actual_pron_raw.append((begin, end, phone))

    combined_stop = False
    actual_pron = []
    for phone, next_phone in zip(actual_pron_raw, actual_pron_raw[1:]+[END_FAKE_STOP]):
        #Combine split stops/affricates
        if combined_stop:
            continue
        elif next_phone[2] in ["ch", "jh"] or same_place(phone, next_phone):
            combined_stop = True
            actual_pron.append(combine_release_closure(phone, next_phone))
        else:
            actual_pron.append(phone)

    if stop[2] == "q":
        if len(actual_pron) != len(dict_pron):
            raise ValueError(f"Stop, {stop} can not be easily mapped to a corresponding stop")
        return "t"
    elif stop[2] == "dx":
        dict_alveolar = [x for x in dict_pron if x in ["t", "d"]]
        actual_alveolar = [x for x in actual_pron if x[2] in ["t", "d", "q", "dx", "dcl"]]
        if len(actual_alveolar) != len(dict_alveolar):
            #420 stops excluded by this laziness
            raise ValueError(f"Stop, {stop} can not be easily mapped to a corresponding stop")
        stop_idx = [i for i, x in enumerate(actual_alveolar) if x[0] == stop[0] and x[1] == stop[1]][0]
        return dict_alveolar[stop_idx]
    else:
        raise ValueError(f"Stop {stop} does not work, this only works for [q, dx]")

def extract_wav_files(sentence_path):
    data_dict = {stop:[] for stop in STOPS}
    with open(sentence_path+".PHN", "r") as f:
        sentence_phones = []
        for line in f:
            (begin, end, phone) = line.strip('\n').split(' ')
            (begin, end) = (int(begin), int(end))
            sentence_phones.append((begin, end, phone, sentence_path))

    #Keep only stops, which aren't part of an affricate
    stop_phones = [phone for phone, next_phone in zip(sentence_phones, sentence_phones[1:] +[END_FAKE_STOP]) \
            if phone[2] in STOP_ALLOPHONES and next_phone[2] not in ["jh", "ch"]]

    combined_stop = False
    for phone, next_phone in zip(stop_phones, stop_phones[1:]+[END_FAKE_STOP]):
        if combined_stop:
            combined_stop = False
            continue
        elif phone[2] in ["q", "dx"]:
            #More complicated to figure out what it is underlying...
            try:
                data_dict[determine_underlying_representation(phone)].append(phone)
            except (KeyError, ValueError):
                #Either this is a epethenic glottal stop before a vowel, or I'm not sure what it is
                continue
        elif phone[1] == next_phone[0] and same_place(phone[2], next_phone[2]):
            #These are two components of the same stop
            data_dict[UNDERLYING_STOP[next_phone[2]]].append(combine_release_closure(phone, next_phone))
            combined_stop = True
        else:
            data_dict[UNDERLYING_STOP[phone[2]]].append(phone)
    return data_dict


stop_dictionary = {stop:[] for stop in STOPS}
for dialect_region in os.listdir(TIMIT_DIR):
    for speaker in os.listdir(os.path.join(TIMIT_DIR, dialect_region)):
        sentences = set(x.split('.')[0] for x in os.listdir(os.path.join(TIMIT_DIR, dialect_region, speaker)))
        for sentence in sentences:
            new_stop_dict = extract_wav_files(os.path.join(TIMIT_DIR, dialect_region, speaker, sentence))
            for stop in STOPS:
                stop_dictionary[stop].extend(new_stop_dict[stop]) 
print("DATA INFO:")
print([f"{stop}:{len(stop_dictionary[stop])}" for stop in STOPS])
for stop in STOPS:
    counts = {}
    for allophone in POSSIBLE_ALLOPHONES[stop]:
        counts[allophone] = len([x for x in stop_dictionary[stop] if x[2] == allophone])
    print(f"{stop}:{counts}")

for stop in STOPS:
    with open(os.path.join(OUTPUT_DIR, f"data_{stop}.csv"), "w") as f:
        f.write(f"begin,end,realisation,file\n")
        for x in stop_dictionary[stop]:
            f.write(f"{x[0]},{x[1]},{x[2]},{x[3]}\n")
