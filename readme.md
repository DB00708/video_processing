# Project Showcase

This repository includes three distinct projects. Below are brief descriptions and instructions for each:

## 1. Email Spam Filtering System

This project focuses on building an email spam filtering system using machine learning. The system is designed to classify emails into spam or non-spam (ham) categories.

### Usage
**Model Testing**: Run `testing_model_on_real_world_data.py` to test the trained model on real-world data located in the email_spam_filtering folder.

### Model Evaluation
The trained ensemble model achieved an **Accuracy of 97.67%** on the test dataset.

## 2. Speaker Detection in Videos

This project involves detecting the number of speakers in a video.

### Usage
**Model Testing**: Run `processing_real_world_data.py` to test the trained model on real-world data located in the getting_data_from_videos folder, it will store 2 files in the diarized_speakers folder.

### Model Evaluation
The trained ensemble model achieved an **Accuracy of 98%** on the test dataset.

**Limitations**
Please note that the current implementation is limited to detecting up to two speakers.

## 3. Face Detection in Videos

This project focuses on detecting faces in a video.

### Usage
**Model Testing**: Run `detecting_faces_in_the_video.py` to test the trained model on real-world data located in the getting_data_from_videos folder, it will store 2 files in the diarized_speakers folder.

## Requirements

- Python 3.x
- Required Python packages can be installed using: `pip install -r requirements.txt`

## Acknowledgments

- The SMS Spam Collection dataset is sourced from Kaggle and can be found [here](https://www.kaggle.com/uciml/sms-spam-collection-dataset).
- The Diarized Speakers dataset is sourced from kaggle and can be found [here](https://www.kaggle.com/datasets/wiradkp/mini-speech-diarization/data)
