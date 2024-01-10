import keras
import numpy as np
import pandas as pd
from text_to_speech.audio_preprocessing import extract_features
from text_to_speech.constants import SAVE_CLEANED_CSV_PATH
from text_to_speech.splitting_the_dataset import split_dataset


def build_model(input_dim, output_dim):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(input_dim=input_dim, output_dim=50, input_length=100))
    model.add(keras.layers.LSTM(128, return_sequences=True, dropout=0.2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LSTM(128, return_sequences=True, dropout=0.2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(output_dim, activation='relu')))
    return model


if __name__ == "__main__":
    saved_metadata_df = pd.read_csv(SAVE_CLEANED_CSV_PATH)
    train_df, val_df, test_df = split_dataset(saved_metadata_df)

    train_df['features'] = train_df['audio_paths'].apply(extract_features)
    val_df['features'] = val_df['audio_paths'].apply(extract_features)
    test_df['features'] = test_df['audio_paths'].apply(extract_features)

    X_train = train_df["Text"]
    X_val = val_df["Text"]
    X_test = test_df["Text"]

    y_train = np.transpose(np.array(train_df['features'].tolist()), (0, 2, 1))
    y_val = np.transpose(np.array(val_df['features'].tolist()), (0, 2, 1))
    y_test = np.transpose(np.array(test_df['features'].tolist()), (0, 2, 1))

    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_train_padded = keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=100, padding='post')

    model = build_model(input_dim=len(tokenizer.word_index) + 1, output_dim=y_train.shape[2])
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train_padded, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_val_padded = keras.preprocessing.sequence.pad_sequences(X_val_seq, maxlen=100, padding='post')
    val_loss, val_accuracy = model.evaluate(X_val_padded, y_val)
    print(f"Validation Loss: {val_loss}, Validation ACCURACY: {val_accuracy}")

    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_padded = keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=100, padding='post')
    test_loss, test_accuracy = model.evaluate(X_test_padded, y_test)
    print(f"Test Loss: {test_loss}, Test ACCURACY: {test_accuracy}")

    model.save("tts_model.h5")
