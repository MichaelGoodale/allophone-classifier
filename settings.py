import os 

CLASSIFIER = os.path.join("classifier", "classifier")

CORPORA_DIRECTORY = "../corpora"
TIMIT_DIR = os.path.join(CORPORA_DIRECTORY, "spade-TIMIT", "textgrid-wav")
OUTPUT_DIR = "data"

PATH_TO_AUTOVOT = "../autovot/autovot/bin"
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

