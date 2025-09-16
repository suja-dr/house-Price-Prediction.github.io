from flask import Flask, render_template, request
import numpy as np
import pickle
import traceback

app = Flask(__name__)

# Load trained model
try:
    with open(r'C:\Users\praveen\Downloads\house price prediction\model\housepriceprediction.pkl', 'rb') as file:
        model1 = pickle.load(file)
except Exception as e:
    print(f" Error loading model: {e}")
    model1 = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        d1 = request.form['Property_Name']
        d2 = int(request.form['Location'])
        d3 = int(request.form['Region'])
        d4 = float(request.form['Property_Age'])
        d5 = int(request.form['Availability'])
        d6 = int(request.form['Area_Type'])
        d7_raw = float(request.form['Area_SqFt'])     
        d8_raw = float(request.form['Rate_SqFt'])      
        d9 = float(request.form['Floor_No'])
        d10 = float(request.form['Bedroom'])
        d11 = float(request.form['Bathroom'])
        d12 = int(request.form['Parking'])

        if d7_raw <= 0 or d8_raw <= 0:
            raise ValueError("Area and Rate must be greater than 0.")

        d7 = d7_raw / 1000
        d8 = d8_raw / 5000

        input_features = np.array([[d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12]])

        prediction = model1.predict(input_features)
        predicted_price = prediction.item()
        print(f" Raw prediction from model: {predicted_price}")

        if predicted_price <= 0:
            fallback_price = d7_raw * d8_raw / 100000  
            fallback_price = round(fallback_price, 2)
            message = f"⚠️ Model returned an invalid price for {d1}. Showing estimated fallback price."
            return render_template("result.html", prediction_label=d1, prediction_value=fallback_price, warning=message)
        else:
            predicted_price = round(predicted_price, 2)
            return render_template("result.html", prediction_label=d1, prediction_value=predicted_price)

    except Exception as e:
        print(" Exception occurred:")
        traceback.print_exc()
        return f"<h3 style='color:red'> Error occurred: {e}</h3>"

if __name__ == "__main__":
    app.run(debug=True)
