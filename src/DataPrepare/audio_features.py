import numpy as np
import scipy.io.wavfile as wav
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
import os
import math


def extract_melspec(audio_dir, files, destpath, fps):
    for f in files:
        file = os.path.join(audio_dir, f + '.wav')
        outfile = destpath + '/' + f + '.npy'

        print('{}\t->\t{}'.format(file, outfile))
        # fs1, X1 = wav.read(file)
        X, fs = librosa.load(file,sr=None)
        # print("X1" + str(X1))
        # print("X" + str(X))
        # X1 = X1.astype(float) / math.pow(2, 15) # ????
        # print("X1 pow" + str(X1))
        assert fs % fps == 0

        hop_len = int(fs / fps)

        n_fft = int(fs * 0.13)
        C = librosa.feature.melspectrogram(y=X, sr=fs, n_fft=2048, hop_length=hop_len, n_mels=27, fmin=0.0, fmax=8000)
        C = np.log(C)

        # plt.figure()
        # librosa.display.specshow(C,sr=fs,x_axis='time',y_axis='mel')
        # plt.title('Beat wavform')
        # plt.show()

        # print("fs: " + str(fs))
        # print("hop_len: " + str(hop_len))
        # print("n_fft: " + str(n_fft))
        # print(C.shape)
        # print(np.min(C), np.max(C))
        np.save(outfile, np.transpose(C))
        # dataC = np.load(outfile)
        # print('DataC {}'.format(dataC.shape))
