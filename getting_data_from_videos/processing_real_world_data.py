import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from getting_data_from_videos.extracting_audio import save_audio_file
from getting_data_from_videos.preprocessing_data import create_feature
from getting_data_from_videos.constants_for_diarized_speakers import SPEAKER_1_SOUND_FILENAME, SPEAKER_2_SOUND_FILENAME


def read_and_reshape_sound(audio_path, segment_size_t):
    y, sr = librosa.load(audio_path)
    segment_size = int(segment_size_t * sr)
    segments = [y[x:x + segment_size] for x in range(0, len(y) - segment_size + 1, segment_size)]
    return segments, sr


def process_real_world_audio(audio_path, model, segment_size_t=0.5, speaker_1_sound_filename=SPEAKER_1_SOUND_FILENAME, speaker_2_sound_filename=SPEAKER_2_SOUND_FILENAME):
    scaler = StandardScaler()

    segments, sr = read_and_reshape_sound(audio_path, segment_size_t)

    features = create_feature(segments, sr)
    scaler.fit(features)

    features_scaled = scaler.transform(features)

    predictions = model.predict(features_scaled)

    speaker_1_sound = np.concatenate([segments[i] for i in range(len(segments)) if predictions[i] == 0])
    speaker_2_sound = np.concatenate([segments[i] for i in range(len(segments)) if predictions[i] == 1])

    save_audio_file(speaker_1_sound, speaker_1_sound_filename, sr)
    save_audio_file(speaker_2_sound, speaker_2_sound_filename, sr)

    return speaker_1_sound, speaker_2_sound, sr
