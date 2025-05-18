from flask import Flask, request, render_template
import pickle
import re
import string

# Load the trained model and vectorizer
with open("emotion_model_sklearn.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Text cleaning function (same as in your notebook)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    return text

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return render_template("index.html")  # You'll create this HTML file

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    input_text = request.form["text"]
    cleaned_text = clean_text(input_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    return render_template("index.html", input_text=input_text, predicted_emotion=prediction)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
