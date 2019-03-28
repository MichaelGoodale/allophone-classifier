import os
import subprocess
import wave
import numpy as np
import scipy.io.wavfile as wavefile
import settings as s


phoneme_phone_mappings = {}

with open('wrdalign.timit', 'r') as f:
    for line in f:
        if line.startswith("# "):
            continue
        elif line.startswith("#"):
            _, dialect, speaker, sentence = line.strip("\n").split("/")
            phoneme_phone_mappings[(dialect, speaker, sentence)] = []
        else:
            if line.count('\t') == 5:
                phoneme, phone, distance, stress, boundary, word = line.strip("\n").split("\t")
            else:
                phoneme, phone, distance, stress, boundary = line.strip("\n").split("\t")
            if phone == "+":
                old_phoneme, old_phone, old_distance, old_stress, old_boundary, word = phoneme_phone_mappings[(dialect, speaker, sentence)][-1]
                phoneme = "{}_{}".format(old_phoneme, phoneme)
                phoneme_phone_mappings[(dialect, speaker, sentence)][-1] = (phoneme, old_phone, old_distance, old_stress, old_boundary, word)
            else:
                data = (phoneme, phone, distance, stress, boundary, word)
                phoneme_phone_mappings[(dialect, speaker, sentence)].append(data)

def same_place(closure, release):
    '''Takes two strings representing different phonemes and checks 
       if they make a valid closure release combination, e.g. bcl and b, dcl and d'''
    return closure[0] == release and closure.endswith("cl")

def get_associated_phones(dialect, speaker, sentence_id):
    dialect = dialect.lower()
    speaker = speaker.lower()
    sentence_id = sentence_id.lower()
    return phoneme_phone_mappings[(dialect, speaker, sentence_id)]

class Sentence:

    def __init__(self, dialect, speaker, sentence_id):
        self.dialect = dialect
        self.speaker = speaker
        self.sentence_id = sentence_id
        temp_phone_list = get_associated_phones(dialect, speaker, sentence_id)
        self.phone_list = []

        sentence_path = os.path.join(s.TIMIT_DIR, self.dialect, self.speaker, self.sentence_id)
        DEL_PHONE_OFFSET = int((25/1000)*16000)

        with open(sentence_path+".PHN", "r") as f:
            pron_list = []
            i = 0
            for line in f:
                (begin, end, phone) = line.strip('\n').split(' ')
                (begin, end) = (int(begin), int(end))
                if len(pron_list) == 1 and phone == "pau":
                    pron_list[0] = (pron_list[0][0], pron_list[0][1], end)
                    continue

                while i < len(temp_phone_list) and temp_phone_list[i][1] == '-':
                    if len(pron_list) > 0:
                        old_begin, old_end, _ = pron_list[-1]
                        pron_list.append((int(old_begin)+DEL_PHONE_OFFSET, int(old_end)+DEL_PHONE_OFFSET, "-"))
                    else:
                        pron_list.append((0, DEL_PHONE_OFFSET, "-"))
                    i += 1
                pron_list.append((begin, end, phone))
                i += 1
        if len(pron_list) != len(temp_phone_list):
            raise ValueError("Pron list does not match phone list in "+"{}/{}/{}".format(self.dialect, self.speaker, self.sentence))
        for (begin, end, phone), (phoneme, _, distance, stress, boundary, word) in zip(pron_list, temp_phone_list):
            if phoneme == "+":
                old_phone = self.phone_list[-1]
                if same_place(old_phone.phone, phone):
                    newphone = "{}rl".format(phone)
                else:
                    newphone = "{}_{}".format(old_phone.phone, phone)
                self.phone_list[-1] = Phone(old_phone.begin, end, newphone, old_phone.underlying_phoneme, word, \
                        old_phone.stress, old_phone.boundary, sentence_path, old_phone_end=old_phone.end, new_phone_begin=begin)
            else:
                self.phone_list.append(Phone(begin, end, phone, phoneme, word, stress, boundary, sentence_path))
            if len(self.phone_list) >= 2:
                self.phone_list[-1].set_previous_phone(self.phone_list[-2])
        syllables = [x.stress for x in self.phone_list if x.stress != "-"]
        syl_i = -1 
        speech_rate = len(syllables)/self.phone_list[0].sentence_length
        for phone in self.phone_list[0:-1]:
            if phone.boundary in ["1", "2"]:
                syl_i += 1
            if syl_i == -1:
                phone.stress = syllables[0]
            elif syl_i >= len(syllables):
                syl_i = len(syllables)-1
            else:
                phone.stress = syllables[syl_i]
            phone.set_speech_rate(speech_rate)

class Phone:
    lengths = {}
#
    def __init__(self, begin, end, phone, underlying_phoneme, word, stress, boundary, path, old_phone_end=None, new_phone_begin=None):
        self.begin = int(begin)
        self.end = int(end)
        self.phone = phone
        self.path = path
        self.word = word
        self.underlying_phoneme = underlying_phoneme
        self.stress = stress
        self.boundary = boundary
        self.previous_phone = "START"
        self.following_phone = "END"
        self.internal_boundary = 0
        if old_phone_end is not None and new_phone_begin is not None:
            self.internal_boundary = old_phone_end

    @property
    def duration(self):
        return self.end-self.begin

    @property
    def window_begin(self):
        '''Begin of audio window in seconds'''
        return max(0, self.begin/16000-(s.WINDOW_BEFORE/1000))

    @property
    def window_end(self):
        '''Begin of audio end in seconds'''
        return min(self.sentence_length-0.005, self.end/16000+(s.WINDOW_AFTER/1000))

    def set_speech_rate(self, speech_rate):
        self.speech_rate = speech_rate

    @property
    def sentence_length(self):
        if self.path in Phone.lengths:
            return Phone.lengths[self.path]
        with wave.open(self.path+".WAV", 'r') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        Phone.lengths[self.path] = duration
        return Phone.lengths[self.path]

    def set_following_phone(self, phone):
        self.following_phone = phone

    def set_previous_phone(self, phone):
        self.previous_phone = phone
        self.previous_phone.set_following_phone(self)

    def __repr__(self):
        return "/{}/,[{}],{},{}".format(self.underlying_phoneme, self.phone, self.begin, self.end)

    def extract_filter_banks(self, margin=0, pre_emphasis=0.97, frame_size=0.005, frame_stride=0.003, NFFT=512, nfilt=40):
        '''Code adapted from https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html'''
        sample_rate, signal = wavefile.read(self.path+".WAV")
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
