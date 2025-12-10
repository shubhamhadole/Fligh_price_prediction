import pickle
import pandas as pd
from database.mongo_connection import get_collection
from config.config import MODEL_PATH


# Load model


with open("flight_price_model.pkl", "rb") as f:
    model = pickle.load(f)


# Mappings
airline_map = {
    'Vistara': 1, 'Air India': 2, 'Indigo': 3,
    'GO FIRST': 4, 'AirAsia': 5, 'SpiceJet': 6
}

city_map = {
    'Delhi': 1, 'Mumbai': 2, 'Bangalore': 3,
    'Kolkata': 4, 'Hyderabad': 5, 'Chennai': 6
}

departure_time_map = {
    'Morning': 1, 'Early Morning': 2, 'Evening': 3,
    'Night': 4, 'Afternoon': 5, 'Late Night': 6
}

arrival_time_map = {
    'Night': 1, 'Evening': 2, 'Morning': 3,
    'Afternoon': 4, 'Early Morning': 5, 'Late Night': 6
}

stops_map = {
    'Zero': 0, 'One': 1, 'Two or More': 2
}

flight_class_map = {
    'Economy': 1, 'Business': 2
}

feature_order = [
    'airline', 'source_city', 'departure_time', 'stops',
    'arrival_time', 'destination_city', 'flight_class',
    'duration', 'days_left'
]

# ---------- Prediction function ----------
def predict_flight_price(user_data: dict):

    user_data['airline'] = airline_map[user_data['airline']]
    user_data['source_city'] = city_map[user_data['source_city']]
    user_data['destination_city'] = city_map[user_data['destination_city']]
    user_data['departure_time'] = departure_time_map[user_data['departure_time']]
    user_data['arrival_time'] = arrival_time_map[user_data['arrival_time']]
    user_data['stops'] = stops_map[user_data['stops']]
    user_data['flight_class'] = flight_class_map[user_data['flight_class']]

    df = pd.DataFrame([user_data])
    df = df[feature_order]

    prediction = model.predict(df)
    return round(float(prediction[0]), 2)


# ---------- MongoDB logging ----------
def save_user_to_db(user_data, predicted_price):
    collection = get_collection()

    doc = {
        "user_input": user_data,
        "predicted_price": predicted_price
    }

    collection.insert_one(doc)
