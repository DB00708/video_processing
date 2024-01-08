import pandas as pd
from sklearn.model_selection import train_test_split
from text_to_speech.constants import SAVE_CLEANED_CSV_PATH, TEST_SIZE, RANDOM_STATE


def split_dataset(metadata_df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    train_df, test_df = train_test_split(metadata_df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=test_size, random_state=random_state)
    return train_df, val_df, test_df


# if __name__ == "__main__":
#     saved_metadata_df = pd.read_csv(SAVE_CLEANED_CSV_PATH)
#     train_df, val_df, test_df = split_dataset(saved_metadata_df)
