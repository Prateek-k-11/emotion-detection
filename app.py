from flask import Flask, render_template, request
import joblib

model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        text = request.form["text"]
        transformed = vectorizer.transform([text])
        prediction = model.predict(transformed)[0]
    return '''
        <form method="post">
            Enter text: <input type="text" name="text">
            <input type="submit">
        </form>
        <h3>{}</h3>
    '''.format(prediction if prediction else "")

if __name__ == "__main__":
    app.run(debug=True)
