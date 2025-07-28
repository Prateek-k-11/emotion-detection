from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        text = request.form["text"]
        transformed = vectorizer.transform([text])
        prediction = model.predict(transformed)[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
