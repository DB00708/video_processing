import glob
import librosa
import numpy as np


def read_path_audio_and_target(is_train=True):
    paths, targets = [], []

    name = 'train' if is_train else 'valid'

    for cls in ['student', 'professor']:
        path = glob.glob(f'dataset_for_diarized_speakers/dataset/{name}/{cls}/*')

        if cls == 'student':
            target = [0 for _ in range(len(path))]
        else:
            target = [1 for _ in range(len(path))]

        paths.extend(path)
        targets.extend(target)

    return paths, targets


def reshape_sound_and_target(path_sound, targets, segment_size_t):
    segments_sound, segments_target, srs = [], [], []

    for path, target in zip(path_sound, targets):
        y, sr = librosa.load(path)
        segment_size = int(segment_size_t * sr)
        segments = [y[x:x + segment_size] for x in range(0, len(y) - segment_size + 1, segment_size)]
        target_segments = [target] * len(segments)

        segments_sound.extend(segments)
        segments_target.extend(target_segments)
        srs.extend([sr] * len(segments))

    return np.array(segments_sound), np.array(segments_target), np.array(srs)


def create_feature(segments, sr):
    features = [
        ('chroma_stft', librosa.feature.chroma_stft),
        ('rms', librosa.feature.rms),
        ('spectral_centroid', librosa.feature.spectral_centroid),
        ('spectral_bandwidth', librosa.feature.spectral_bandwidth),
        ('spectral_rolloff', librosa.feature.spectral_rolloff),
        ('zero_crossing_rate', librosa.feature.zero_crossing_rate),
        ('mfcc', librosa.feature.mfcc)
    ]

    features_segmentation = []

    for seg in segments:
        feature_segmentation = []
        try:
            for name, func in features:
                if name in ['rms', 'zero_crossing_rate']:
                    y0 = func(y=seg)
                    feature_segmentation.append(np.mean(y0))
                elif name == 'mfcc':
                    y0 = func(y=seg, sr=sr)
                    for i, m in enumerate(y0, 1):
                        feature_segmentation.append(np.mean(m))
                else:
                    y0 = func(y=seg, sr=sr)
                    feature_segmentation.append(np.mean(y0))
        except Exception as e:
            print(e)

        features_segmentation.append(feature_segmentation)

    features_segmentation = np.array(features_segmentation)
    return features_segmentation


def preprocessed_sounds():
    paths_train, targets_train = read_path_audio_and_target(is_train=True)
    paths_test, targets_test = read_path_audio_and_target(is_train=False)

    segment_size_t = 0.5
    segments_sound_train, segments_target_train, sr_train = reshape_sound_and_target(paths_train, targets_train,
                                                                                     segment_size_t)
    segments_sound_test, segments_target_test, sr_test = reshape_sound_and_target(paths_test, targets_test,
                                                                                  segment_size_t)

    feature_train = create_feature(segments_sound_train, sr_train[0])
    feature_test = create_feature(segments_sound_test, sr_test[0])

    return feature_train, feature_test, segments_target_train, segments_sound_test, segments_target_test, sr_test
