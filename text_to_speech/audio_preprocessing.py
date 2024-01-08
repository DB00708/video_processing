import librosa
import sklearn


def extract_features(file_path):
    x, sr = librosa.load(file_path)

    zero_crossings = librosa.zero_crossings(x, pad=False)
    zero_crossing_rate = sum(zero_crossings)

    mfccs = librosa.feature.mfcc(y=x, sr=sr)
    normalized_mfccs = sklearn.preprocessing.minmax_scale(mfccs, axis=1)

    return normalized_mfccs
