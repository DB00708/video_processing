import librosa
import numpy as np
from sklearn.preprocessing import minmax_scale


def extract_features(file_path, max_len=100):
    x, sr = librosa.load(file_path)

    zero_crossings = librosa.zero_crossings(x, pad=False)
    zero_crossing_rate = sum(zero_crossings)

    mfccs = librosa.feature.mfcc(y=x, sr=sr)
    normalized_mfccs = minmax_scale(mfccs, axis=1)

    if normalized_mfccs.shape[1] < max_len:
        normalized_mfccs = np.pad(normalized_mfccs, ((0, 0), (0, max_len - normalized_mfccs.shape[1])), mode='constant')
    elif normalized_mfccs.shape[1] > max_len:
        normalized_mfccs = normalized_mfccs[:, :max_len]

    return normalized_mfccs
