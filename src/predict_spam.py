import joblib
import os

# Paths for saved model/vectorizer
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'spam_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl')

# Load trained model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Function to predict spam or ham
def predict_email(text):
    text_features = vectorizer.transform([text])
    prediction = model.predict(text_features)
    label = 'Spam' if prediction[0] == 1 else 'Ham'
    print(f"Email: {text}")
    print(f"Prediction: {label}")

# Example usage
sample_email = "Congratulations! You've won a $1000 Amazon gift card. Click here to claim."
predict_email(sample_email)

sample_email2 = "Hi, I have attached the project report for your review. Thanks."
predict_email(sample_email2)
