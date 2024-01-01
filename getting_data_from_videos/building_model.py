import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from getting_data_from_videos.constants_for_diarized_speakers import ENSEMBLE_MODEL
from getting_data_from_videos.extracting_audio import segment_audio
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from getting_data_from_videos.preprocessing_data import preprocessed_sounds


def train_and_evaluate_model(model, feature_train, segments_target_train, feature_test, segments_target_test):
    scaler = StandardScaler()
    scaler.fit(feature_train)

    feature_train_scaler = scaler.transform(feature_train)
    feature_test_scaler = scaler.transform(feature_test)

    model.fit(feature_train_scaler, segments_target_train)
    pred = model.predict(feature_test_scaler)

    accuracy = accuracy_score(segments_target_test, pred)
    print(f"Accuracy: {accuracy:.2f}")


def ensemble_predict(models, features):
    predictions = [model.predict(features) for model in models]
    ensemble_pred = np.round(np.mean(predictions, axis=0))
    return ensemble_pred


if __name__ == "__main__":
    feature_train, feature_test, segments_target_train, segments_sound_test, segments_target_test, sr_test = preprocessed_sounds()

    scaler = StandardScaler()
    scaler.fit(feature_train)

    feature_train_scaler = scaler.transform(feature_train)
    feature_test_scaler = scaler.transform(feature_test)

    model_svc = SVC(C=0.1)
    model_rfc = RandomForestClassifier(random_state=42)

    ensemble_models = [('svc', model_svc), ('rfc', model_rfc)]
    ensemble = VotingClassifier(estimators=ensemble_models, voting='hard')

    train_and_evaluate_model(ensemble, feature_train, segments_target_train, feature_test, segments_target_test)

    joblib.dump(ensemble, ENSEMBLE_MODEL)
    loaded_ensemble = joblib.load(ENSEMBLE_MODEL)

    ensemble_pred_loaded = loaded_ensemble.predict(feature_test_scaler)
    ensemble_accuracy_loaded = accuracy_score(segments_target_test, ensemble_pred_loaded)

    segment_audio(ensemble_pred_loaded, segments_sound_test, sr_test[0])
