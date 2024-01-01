from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from email_spam_filtering.preprocessing_data import preprocessing_the_dataset
from constants_for_email_spam_filtering import RANDOM_STATE, VOTING_SCHEME
from email_spam_filtering.saving_and_loading_the_model import save_model_and_vectorizer


def evaluate_model(model, test_features, test_labels):
    predictions = model.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)

    return accuracy, report


def build_ensemble_models():
    return [
        RandomForestClassifier(random_state=RANDOM_STATE),
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        SVC(kernel='linear', random_state=RANDOM_STATE)
    ]


def build_voting_ensemble(models):
    return VotingClassifier(estimators=[(f"model_{i}", model) for i, model in enumerate(models)], voting=VOTING_SCHEME)


def model_to_classify_spam_or_ham():
    train_features_resampled, train_labels_resampled, test_features, test_labels, vectorizer = preprocessing_the_dataset()

    models = build_ensemble_models()
    ensemble_model = build_voting_ensemble(models)
    ensemble_model.fit(train_features_resampled, train_labels_resampled)

    # Uncomment to save the model and vectorizer
    # save_model_and_vectorizer(ensemble_model, vectorizer, model_location='model_for_email_spam_filtering')

    # Uncomment to see the accuracy of the model
    # accuracy, report = evaluate_model(ensemble_model, test_features, test_labels)
    # print(f"Accuracy: {accuracy:.4f}")
    # print("Classification Report:\n", report)

    return ensemble_model


# if __name__ == "__main__":
#     model = model_to_classify_spam_or_ham()
