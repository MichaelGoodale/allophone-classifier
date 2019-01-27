import settings as s
import subprocess

def same_place(closure, release):
    '''Takes two strings representing different phonemes and checks 
       if they make a valid closure release combination, e.g. bcl and b, dcl and d'''
    return closure[0] == release

class Word:

    def __init__(self, stop):
        self.stop = stop

        found_word = False
        with open(self.stop.path+".WRD", "r") as f:
            for line in f:
                (begin, end, word) = line.strip('\n').split(' ')
                (begin, end) = (int(begin), int(end))
                if begin <= self.stop.begin <= self.stop.end <= end:
                    found_word = True
                    break
        if not found_word:
            raise KeyError(f"Stop, {self.stop} is not in any of the associated words")

        self.begin = begin
        self.end = end
        self.dictionary_pronunciation = s.timit_dictionary[word]
        self._actual_pronunciation = None

    @property
    def actual_pronunciation(self):
        if self._actual_pronunciation is not None:
            return self._actual_pronunciation

        actual_pron_raw = []
        with open(self.stop.path+".PHN", "r") as f:
            for line in f:
                (begin, end, phone) = line.strip('\n').split(' ')
                (begin, end) = (int(begin), int(end))
                if self.begin <= begin <= end <= self.end:
                    actual_pron_raw.append((begin, end, phone))

        combined_stop = False
        actual_pron = []
        for phone, next_phone in zip(actual_pron_raw, actual_pron_raw[1:]+[(-1, -1, "x")]):
            #Combine split stops/affricates
            if combined_stop:
                combined_stop = False
                continue
            elif next_phone[2] in ["ch", "jh"] or same_place(phone[2], next_phone[2]):
                combined_stop = True
                actual_pron.append((phone[0], next_phone[1], phone[2][0]+"rl"))
            else:
                actual_pron.append(phone)
        self._actual_pronunciation = actual_pron
        return self._actual_pronunciation



class Stop:

    def __init__(self, begin, end, phone, path):
        self.begin = int(begin)
        self.end = int(end)
        self.phone = phone
        self.path = path
        self._word = None
        self._word_position = None
        self._underlying_stop = None

    @property
    def duration(self):
        return self.end-self.begin

    @property
    def sentence_length(self):
        return float(subprocess.check_output(["soxi", "-D", self.path+".WAV"]))


    @property
    def word(self):
        if self._word is not None:
            return self._word
        self._word = Word(self)
        return self._word

    @property
    def word_position(self):
        if self._word_position is not None:
            return self._word_position
        for i, (begin, end, phone) in enumerate(self.word.actual_pronunciation):
            if begin == self.begin and end == self.end:
                break
        if i == 0:
            self._word_position = "initial"
        elif i == len(self.word.actual_pronunciation) - 1:
            self._word_position = "final"
        else:
            self._word_position = "medial"
        return self._word_position

    @property
    def underlying_stop(self):
        if self._underlying_stop is not None:
            return self._underlying_stop

        if self.phone in s.OTHER_PHONES:
            if self.phone in s.FRICATIVES:
                self._underlying_stop = "fric"
            elif self.phone in s.NASALS:
                self._underlying_stop = "nasal"
            elif self.phone in s.GLIDES:
                self._underlying_stop = "glide"
            elif self.phone in s.VOWELS:
                self._underlying_stop = "vowel"
            elif self.phone in s.PAUSE:
                self._underlying_stop = "pau"
        elif self.phone not in ["q", "dx"]:
            self._underlying_stop = s.UNDERLYING_STOP[self.phone]
        elif self.phone == "q":
            if len(self.word.actual_pronunciation) != len(self.word.dictionary_pronunciation):
                if s.INCLUDE_NON_T_Q:
                    self._underlying_stop = "t+"
                else:
                    raise ValueError(f"Stop, {self} can not be easily mapped to a corresponding stop")
            else:
                self._underlying_stop = "t"
        elif self.phone == "dx":
            dict_alveolar = [x for x in self.word.dictionary_pronunciation if x in ["t", "d"]]
            actual_alveolar = [x for x in self.word.actual_pronunciation if x[2] in ["t", "d", "q", "dx", "dcl", "drl", "tcl", "trl"]]
            if len(actual_alveolar) != len(dict_alveolar):
                raise ValueError(f"Stop, {self} can not be easily mapped to a corresponding stop")
            stop_idx = [i for i, x in enumerate(actual_alveolar) if x[0] == self.begin and x[1] == self.end][0]
            self._underlying_stop = dict_alveolar[stop_idx]

        return self._underlying_stop
