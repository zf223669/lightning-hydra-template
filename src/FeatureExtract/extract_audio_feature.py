import numpy as np
import scipy.io.wavfile as wav
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
import os
import math

import hydra
import pyrootutils
from omegaconf import DictConfig


def extract_melspec(audio_dir, files, destpath, fps):
    for f in files:
        file = os.path.join(audio_dir, os.path.splitext(f)[0] + '.wav')
        outfile = destpath + '/' + os.path.splitext(f)[0] + '.npy'

        print('{}\t->\t{}'.format(file, outfile))
        # fs1, X1 = wav.read(file)
        X, fs = librosa.load(file, sr=None)
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

        print("fs: " + str(fs))
        print("hop_len: " + str(hop_len))
        print("n_fft: " + str(n_fft))
        print(C.shape)
        print(np.min(C), np.max(C))
        np.save(outfile, np.transpose(C))
        # dataC = np.load(outfile)
        # print('DataC {}'.format(dataC.shape))


root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train_gesture_generation.yaml")
def main(cfg: DictConfig) -> float:
    audio_path = os.path.join(cfg.datamodule.data_dir, 'AudioData')
    feature_path = os.path.join(audio_path, 'Features')
    print(str(audio_path))
    print(feature_path)

    for path_root, dirs, files in os.walk(audio_path):
        print(files)
        extract_melspec(audio_dir=audio_path, files=files, destpath=feature_path, fps=20)
    return 0


if __name__ == "__main__":
    main()
