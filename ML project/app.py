from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("diab.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        try:
            pregnancies = int(request.form["pregnancies"])
            glucose = int(request.form["glucose"])
            insulin = int(request.form["insulin"])
            bmi = float(request.form["bmi"])
            age = int(request.form["age"])

            # Prepare input data for prediction
            input_data = np.array([[pregnancies, glucose, insulin, bmi, age]])

            # Make prediction
            prediction_result = model.predict(input_data)

            # Display the result
            prediction = "Diabetic" if prediction_result[0] == 1 else "Non-diabetic"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)