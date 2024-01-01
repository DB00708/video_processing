DATASET_NAME = 'uciml/sms-spam-collection-dataset'
DATASET_FOLDER = 'dataset_for_email_spam_filtering'
DATASET_ZIP_FILE = 'sms-spam-collection-dataset.zip'
DATASET_CSV_FILE = 'spam.csv'

LABEL_COLUMN = 'v1'
TEXT_COLUMN = 'v2'
RANDOM_STATE = 42
TEST_SIZE = 0.2

VOTING_SCHEME = 'hard'

MODEL_FOLDER = 'model_for_email_spam_filtering'
MODEL_FILENAME = 'ensemble_model.joblib'
VECTORIZER_FILENAME = 'tfidf_vectorizer.joblib'

CLASS_LABELS = ['not spam', 'spam']
