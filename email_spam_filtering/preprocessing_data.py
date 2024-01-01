import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from constants_for_email_spam_filtering import TEXT_COLUMN, TEST_SIZE, RANDOM_STATE, LABEL_COLUMN, DATASET_FOLDER, DATASET_CSV_FILE


def load_dataset(dataset_folder=DATASET_FOLDER, dataset_csv_file=DATASET_CSV_FILE):
    return pd.read_csv(os.path.join(dataset_folder, dataset_csv_file), encoding='latin-1')


def preprocess_data(df, text_column, label_column):
    label_encoder = LabelEncoder()
    df[label_column] = label_encoder.fit_transform(df[label_column])
    train_data, test_data, train_labels, test_labels = train_test_split(
        df[text_column], df[label_column], test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return train_data, test_data, train_labels, test_labels


def oversample_data(train_features, train_labels):
    smote = SMOTE(random_state=RANDOM_STATE)
    train_features_resampled, train_labels_resampled = smote.fit_resample(train_features, train_labels)
    return train_features_resampled, train_labels_resampled


def extract_features(train_data, test_data):
    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(train_data)
    test_features = vectorizer.transform(test_data)

    return train_features, test_features, vectorizer


def preprocessing_the_dataset(text_column=TEXT_COLUMN, label_column=LABEL_COLUMN):
    df = load_dataset()

    train_data, test_data, train_labels, test_labels = preprocess_data(df, text_column, label_column)
    train_features, test_features, vectorizer = extract_features(train_data, test_data)

    train_features_resampled, train_labels_resampled = oversample_data(train_features, train_labels)
    return train_features_resampled, train_labels_resampled, test_features, test_labels, vectorizer
