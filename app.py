from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        int(request.form['area']),
        int(request.form['bedrooms']),
        int(request.form['bathrooms']),
        int(request.form['stories']),
        encoders['mainroad'].transform([request.form['mainroad']])[0],
        encoders['guestroom'].transform([request.form['guestroom']])[0],
        encoders['basement'].transform([request.form['basement']])[0],
        encoders['hotwaterheating'].transform([request.form['hotwaterheating']])[0],
        encoders['airconditioning'].transform([request.form['airconditioning']])[0],
        int(request.form['parking']),
        encoders['prefarea'].transform([request.form['prefarea']])[0]
    ]

    prediction = model.predict([features])[0]
    prediction_text = f"Predicted House Price: ₹{int(prediction):,}"

    return render_template('index.html', prediction=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
