from collecting_data import configure_kaggle_credentials, download_dataset, unzip_dataset
from global_constants import KAGGLE_USERNAME, KAGGLE_API_KEY
from getting_data_from_videos.constants_for_diarized_speakers import DATASET_NAME, DATASET_FOLDER, DATASET_ZIP_FILE


def getting_data_from_kaggle_for_diarized_speakers():
    configure_kaggle_credentials(kaggle_username=KAGGLE_USERNAME, kaggle_api_key=KAGGLE_API_KEY)
    download_dataset(dataset_name=DATASET_NAME, destination_folder=DATASET_FOLDER)
    unzip_dataset(zip_file=DATASET_ZIP_FILE, source_folder=DATASET_FOLDER, destination_folder=DATASET_FOLDER)


if __name__ == "__main__":
    getting_data_from_kaggle_for_diarized_speakers()
