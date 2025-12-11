from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model and scaler
with open("model/flight_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Encoding maps
airline_map = {
    "Vistara": 1,
    "Air_India": 2,
    "Indigo": 3,
    "GO_FIRST": 4,
    "AirAsia": 5,
    "SpiceJet": 6
}

city_map = {
    "Delhi": 1, "Mumbai": 2, "Bangalore": 3,
    "Kolkata": 4, "Hyderabad": 5, "Chennai": 6
}

time_map = {
    "Morning": 1, "Early_Morning": 2, "Evening": 3,
    "Night": 4, "Afternoon": 5, "Late_Night": 6
}

stops_map = {
    "one": 1, "zero": 2, "two_or_more": 3
}

class_map = {
    "Economy": 1, "Business": 2
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "airline": airline_map[request.form["airline"]],
            "source_city": city_map[request.form["source_city"]],
            "departure_time": time_map[request.form["departure_time"]],
            "stops": stops_map[request.form["stops"]],
            "arrival_time": time_map[request.form["arrival_time"]],
            "destination_city": city_map[request.form["destination_city"]],
            "flight_class": class_map[request.form["flight_class"]],
            "duration": float(request.form["duration"]),
            "days_left": int(request.form["days_left"])
        }

        df = pd.DataFrame([data])
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)[0]

        return render_template("index.html", result=round(prediction, 2))

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)