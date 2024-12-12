from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import random

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Transaction amount from form
        amount = float(request.form["amount"])
        
        # Create a feature array with 30 features, defaulting all except 'amount'
        input_data = [0] * 30  # Replace 0 with default values as per your dataset
        input_data[0] = amount  # Assuming 'Amount' is the first feature
        
        # Make prediction
        prediction = model.predict(np.array(input_data).reshape(1, -1))[0]
        result = "Fraudulent" if prediction == 1 else "Legitimate"
        return render_template("index.html", prediction=result)
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")



@app.route("/dashboard")
def dashboard():
    live_data = [
        {
            "transaction_id": i,
            "amount": round(random.uniform(1, 1000), 2),
            "status": random.choice(["Legitimate", "Fraudulent"]),
        }
        for i in range(1, 11)
    ]
    return jsonify(live_data)

if __name__ == "__main__":
    app.run(debug=False)
