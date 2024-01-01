from email_spam_filtering.constants_for_email_spam_filtering import CLASS_LABELS
from email_spam_filtering.saving_and_loading_the_model import load_model_and_vectorizer


def preprocess_the_text_of_email(email_text, vectorizer):
    text_features = vectorizer.transform([email_text])
    return text_features


def classify_spam_or_not_spam(email_text):
    ensemble_model, vectorizer = load_model_and_vectorizer()
    email_features = preprocess_the_text_of_email(email_text, vectorizer)
    numerical_prediction = ensemble_model.predict(email_features)

    predicted_label = CLASS_LABELS[numerical_prediction[0]]

    print(f"Predicted label for the given email: {predicted_label}")


if __name__ == "__main__":
    text = """congratulations, you have won 50$. go to http://bit.ly//123456 to claim now"""
    classify_spam_or_not_spam(text)
