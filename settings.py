import os 

CLASSIFIER = os.path.join("classifier", "classifier")

CORPORA_DIRECTORY = "../corpora"
TIMIT_DIR = os.path.join(CORPORA_DIRECTORY, "spade-TIMIT-noisy", "spade-TIMIT-noisy", "audio_and_transcripts")
SOTC_DIR = os.path.join(CORPORA_DIRECTORY, "spade-SOTC", "audio_and_transcripts")
BUCKEYE_DIR = os.path.join(CORPORA_DIRECTORY, "spade-Buckeye", "textgrid-wav")
OUTPUT_DIR = "data"

PATH_TO_AUTOVOT = "../autovot/autovot/bin"

STOP_ALLOPHONES = ["b", "d", "g", "p", "t", "k", "bcl", "dcl", "gcl", "pcl", "tcl", "kcl", "dx", "q", "brl", "drl", "grl", "prl", "trl", "krl"]
STOPS = ["b", "d", "g", "p", "t", "k"]
FRICATIVES = ["s", "sh", "z", "zh", "f", "th", "v", "dh"]
NASALS = [ "m", "n", "ng", "em", "en", "eng", "nx"]
GLIDES = ["l", "r", "w", "y", "hh", "hv", "el"]
VOWELS = ["iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h" ]
PAUSE = ["pau"]
EXTRA_PHONES  = ["tx", "pau", "fric", "nasal", "glide", "vowel"]
OTHER_PHONES = FRICATIVES+NASALS+GLIDES+VOWELS+PAUSE
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

WINDOW_BEFORE = 50
WINDOW_AFTER = 50

INCLUDE_NON_T_Q = True

timit_dictionary = {}
with open('TIMITDIC.TXT', 'r') as f:
    for line in f:
        if line.startswith(';'):
            continue
        word, pronounciation = line.strip('\n').split('  ')
        word, _, _= word.partition("~")
        timit_dictionary[word] = pronounciation.strip('/').split(' ')
        if word.startswith("-"):
            timit_dictionary[word[1:]] = pronounciation.strip('/').split(' ')
        if word.endswith("-"):
            timit_dictionary[word[:-1]] = pronounciation.strip('/').split(' ')



