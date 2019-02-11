import subprocess
import numpy as np
import scipy.io.wavfile as wavefile
import settings as s

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
                    self._underlying_stop = "tx"
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

    def extract_filter_banks(self, margin=0, pre_emphasis=0.97, frame_size=0.005, frame_stride=0.003, NFFT=512, nfilt=40):
        '''Code adapted from https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html'''
        sample_rate, signal = wavefile.read(f"{self.path}.WAV")
        signal = signal[max(0, self.begin-int(margin*sample_rate)):self.end+int(margin*sample_rate)]
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        frames *= np.hamming(frame_length)

        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum


        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bins = np.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bins[m - 1])   # left
            f_m = int(bins[m])             # center
            f_m_plus = int(bins[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        return filter_banks


