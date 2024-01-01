import numpy as np
import soundfile as sf
from moviepy.video.io.VideoFileClip import VideoFileClip
from getting_data_from_videos.constants_for_diarized_speakers import SPEAKER_1_SOUND_FILENAME, SPEAKER_2_SOUND_FILENAME


def extract_audio(video_path, audio_file_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_file_path)


def save_audio_file(sound, filename, rate):
    sf.write(filename, sound, rate)


def segment_audio(predictions, segments, sr, speaker_1_sound_filename=SPEAKER_1_SOUND_FILENAME, speaker_2_sound_filename=SPEAKER_2_SOUND_FILENAME):
    student_sound = np.concatenate(segments[predictions == 0])
    professor_sound = np.concatenate(segments[predictions == 1])

    save_audio_file(student_sound, speaker_1_sound_filename, sr)
    save_audio_file(professor_sound, speaker_2_sound_filename, sr)
