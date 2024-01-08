import os
import pandas as pd


def processing_csv(file_path, audio_dir_path):
    try:
        df = pd.read_csv(file_path, sep='|', names=["File", "Text", "Text2"])
        df = df.drop(columns=["Text2"])
        df['audio_paths'] = list_files_in_directory(audio_dir_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred - {str(e)}")
        return None


def list_files_in_directory(directory):
    try:
        file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if
                      os.path.isfile(os.path.join(directory, file))]
        sorted_file_paths = sorted(file_paths)
        return sorted_file_paths
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory}")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred - {str(e)}")
        return None


def cleaned_dataset(csv_file_path, audio_dir_path ,save_cleaned_csv_file):
    metadata_df = processing_csv(csv_file_path, audio_dir_path)
    if save_cleaned_csv_file is not None and metadata_df is not None:
        metadata_df.to_csv(save_cleaned_csv_file, index=False)
        print(f"DataFrame saved to {save_cleaned_csv_file}")

    return metadata_df


if __name__ == "__main__":
    
    file_path = r'C:\Users\divya\PycharmProjects\video_processing\text_to_speech\text_to_speech_dataset\LJSpeech-1.1\metadata.csv'
    audio_dir_path = r'C:\Users\divya\PycharmProjects\video_processing\text_to_speech\text_to_speech_dataset\LJSpeech-1.1\wavs'
    save_cleaned_csv_file = r"C:\Users\divya\PycharmProjects\video_processing\text_to_speech\text_to_speech_dataset\cleaned.csv"
    metadata_df = cleaned_dataset(file_path, audio_dir_path, save_cleaned_csv_file)
