from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Load dataset for demo endpoints
df = pd.read_csv("data/creditcard.csv")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_features = []

        # Time
        input_features.append(float(request.form["Time"]))

        # V1 to V28
        for i in range(1, 29):
            input_features.append(float(request.form[f"V{i}"]))

        # Amount
        input_features.append(float(request.form["Amount"]))

        features = np.array(input_features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        if prediction == 1:
            result = f"⚠️ Fraudulent Transaction Detected!"
        else:
            result = f"✅ Normal Transaction"

        return render_template(
            "index.html",
            prediction_text=result,
            probability_text=f"Fraud Risk: {probability:.2%}"
        )

    except Exception as e:
        return f"Error occurred: {e}"


# --------------------------
# Clean Demo Test Endpoints
# --------------------------

def format_features(sample_row):
    formatted = ""
    for col in sample_row.index:
        if col != "Class":
            formatted += f"<b>{col}</b>: {sample_row[col]:.4f}<br>"
    return formatted


@app.route("/test_normal")
def test_normal():
    sample = df[df["Class"] == 0].iloc[0]
    features = sample.drop("Class").values
    scaled = scaler.transform([features])

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    formatted_features = format_features(sample)

    return f"""
    <h2>Normal Transaction Test</h2>
    <p>{formatted_features}</p>
    <h3>Prediction: {prediction}</h3>
    <h3>Fraud Risk: {probability:.2%}</h3>
    """


@app.route("/test_fraud")
def test_fraud():
    sample = df[df["Class"] == 1].iloc[0]
    features = sample.drop("Class").values
    scaled = scaler.transform([features])

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    formatted_features = format_features(sample)

    return f"""
    <h2>Fraud Transaction Test</h2>
    <p>{formatted_features}</p>
    <h3>Prediction: {prediction}</h3>
    <h3>Fraud Risk: {probability:.2%}</h3>
    """


if __name__ == "__main__":
    app.run(debug=True)