import os
from zipfile import ZipFile


def configure_kaggle_credentials(kaggle_username, kaggle_api_key):
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_api_key


def download_dataset(dataset_name, destination_folder):
    try:
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        os.system(f'kaggle datasets download -d {dataset_name} -p {destination_folder}')
    except Exception as e:
        print(f"Error downloading the dataset: {e}")


def unzip_dataset(zip_file, source_folder, destination_folder):
    try:
        with ZipFile(os.path.join(source_folder, zip_file), 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
    except Exception as e:
        print(f"Error unzipping the dataset: {e}")
