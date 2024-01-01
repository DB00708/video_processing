import os
import joblib
from constants_for_email_spam_filtering import MODEL_FOLDER, MODEL_FILENAME, VECTORIZER_FILENAME


def getting_model_path(model_location=MODEL_FOLDER, filename=MODEL_FILENAME):
    model_path = os.path.join(model_location, filename)
    return model_path


def getting_vectorizer_path(model_location=MODEL_FOLDER, filename=VECTORIZER_FILENAME):
    vectorizer_path = os.path.join(model_location, filename)
    return vectorizer_path


def save_model_and_vectorizer(model, vectorizer, model_location=MODEL_FOLDER):
    os.makedirs(model_location, exist_ok=True)
    model_filename = getting_model_path()
    vectorizer_filename = getting_vectorizer_path()

    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)


def load_model_and_vectorizer():
    model_filename = getting_model_path()
    vectorizer_filename = getting_vectorizer_path()

    model = joblib.load(model_filename)
    vectorizer = joblib.load(vectorizer_filename)
    return model, vectorizer
