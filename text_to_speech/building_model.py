import keras
import numpy as np
import pandas as pd
from text_to_speech.audio_preprocessing import extract_features
from text_to_speech.constants import SAVE_CLEANED_CSV_PATH
from text_to_speech.splitting_the_dataset import split_dataset

if __name__ == "__main__":

    saved_metadata_df = pd.read_csv(SAVE_CLEANED_CSV_PATH)
    train_df, val_df, test_df = split_dataset(saved_metadata_df)

    train_df['features'] = train_df['audio_paths'].apply(extract_features)
    val_df['features'] = val_df['audio_paths'].apply(extract_features)
    test_df['features'] = test_df['audio_paths'].apply(extract_features)

    X_train = train_df["text"]
    X_val = val_df["text"]
    X_test = test_df["text"]

    y_train = np.array(train_df['features'].tolist())
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_train_padded = keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=100, padding='post')

    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=100))
    model.add(keras.layers.LSTM(100, return_sequences=True))
    model.add(keras.layers.LSTM(100))
    model.add(keras.layers.Dense(y_train.shape[1], activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_split=0.1)

    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_val_padded = keras.preprocessing.sequence.pad_sequences(X_val_seq, maxlen=100, padding='post')

    y_val = np.array(val_df['features'].tolist())
    val_loss, val_mae = model.evaluate(X_val_padded, y_val)
    print(f"Validation Loss: {val_loss}, Validation MAE: {val_mae}")

    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_padded = keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=100, padding='post')
    y_test = np.array(test_df['features'].tolist())
    predictions = model.predict(X_test_padded)

    model.save("tts_model.h5")
