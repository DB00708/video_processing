import os
from pytube import YouTube


def download_youtube_video(url, output_path='sample_videos'):
    output_file_path = ""
    try:
        yt = YouTube(url)
        video_stream = yt.streams.filter(file_extension="mp4").first()

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_file_path = os.path.join(output_path, f"{yt.title}.mp4")
        video_stream.download(output_path)

    except Exception as e:
        print(f"Error: {e}")

    return output_file_path
