import streamlit as st
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import requests
import folium
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "surge_model.txt"

model = lgb.Booster(model_file=str(MODEL_PATH))

encoders = joblib.load(ROOT / "models" / "encoders.pkl")
le_surge = joblib.load(ROOT / "models" / "surge_label_encoder.pkl")
feature_names = joblib.load(ROOT / "models" / "feature_names.pkl")

le_zone = encoders["zone"]
le_weather = encoders["weather"]

API_KEY = st.secrets["OPENWEATHER_API_KEY"]

feature_cols = [
    "passenger_count",
    "hour",
    "day",
    "zone",
    "weather",
    "temperature",
    "traffic"
]

def get_future_weather(lat, lon, target_dt):
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(url)
        data = response.json()
    except Exception:
        return 25.0, "Clear"

    if response.status_code != 200:
        return 25.0, "Clear"

    if "list" not in data:
        return 25.0, "Clear"

    best_match = None
    min_diff = float("inf")

    for entry in data["list"]:
        entry_dt = pd.to_datetime(entry["dt_txt"])
        diff = abs((entry_dt - target_dt).total_seconds())
        if diff < min_diff:
            min_diff = diff
            best_match = entry

    if best_match:
        temp = best_match["main"]["temp"]
        weather_desc = best_match["weather"][0]["main"]
        return temp, weather_desc
    else:
        return 25.0, "Clear"

def estimate_future_traffic(dt, zone=None):
    hour = dt.hour
    weekday = dt.weekday()
    
    high_traffic_zones = ["MG Road", "Indiranagar", "Koramangala", "Whitefield", "Marathahalli"]
    medium_traffic_zones = ["HSR Layout", "Bellandur", "Electronic City", "BTM Layout"]
    low_traffic_zones = ["Jayanagar", "Basavanagudi", "Banashankari", "Hebbal", "Rajajinagar", "Yelahanka"]
    
    zone_multiplier = 1.0
    if zone:
        if zone in high_traffic_zones:
            zone_multiplier = 1.2
        elif zone in low_traffic_zones:
            zone_multiplier = 0.8

    if weekday < 5:
        if 8 <= hour <= 11:
            base_traffic = 5
        elif 17 <= hour <= 20:
            base_traffic = 5
        elif 12 <= hour <= 16:
            base_traffic = 3
        else:
            base_traffic = 2
    else:
        if 10 <= hour <= 13:
            base_traffic = 4
        elif 18 <= hour <= 21:
            base_traffic = 5
        else:
            base_traffic = 2
    
    return min(5, base_traffic * zone_multiplier)

def build_input_df(prediction_dt, zone, temperature, weather, traffic, passenger_count):
    row = {
        "passenger_count": int(passenger_count),
        "hour": int(prediction_dt.hour),
        "day": int(prediction_dt.weekday()),
        "zone": int(le_zone.transform([zone])[0]),
        "weather": int(le_weather.transform([weather])[0]),
        "temperature": float(temperature),
        "traffic": float(traffic)
    }
    df = pd.DataFrame([row])
    df = df[feature_names]
    return df

def predict_from_model(model, X):
    proba = model.predict(X)
    proba = np.asarray(proba)
    
    if proba.ndim == 1:
        proba = proba.reshape(1, -1)
    
    if proba.shape[1] != 3:
        proba_transposed = proba.T
        if proba_transposed.shape[1] == 3:
            proba = proba_transposed
    
    pred = np.argmax(proba, axis=1)
    return pred, proba

bengaluru_zones = [
    {"zone": "Koramangala", "lat": 12.9352, "lon": 77.6245},
    {"zone": "HSR Layout", "lat": 12.9121, "lon": 77.6401},
    {"zone": "Indiranagar", "lat": 12.9716, "lon": 77.6412},
    {"zone": "MG Road", "lat": 12.9750, "lon": 77.6050},
    {"zone": "Whitefield", "lat": 12.9698, "lon": 77.7499},
    {"zone": "Bellandur", "lat": 12.9279, "lon": 77.6762},
    {"zone": "Electronic City", "lat": 12.8406, "lon": 77.6770},
    {"zone": "Marathahalli", "lat": 12.9550, "lon": 77.7011},
    {"zone": "Jayanagar", "lat": 12.9250, "lon": 77.5938},
    {"zone": "Basavanagudi", "lat": 12.9416, "lon": 77.5679},
    {"zone": "Banashankari", "lat": 12.9150, "lon": 77.5736},
    {"zone": "Hebbal", "lat": 13.0358, "lon": 77.5970},
    {"zone": "Rajajinagar", "lat": 12.9910, "lon": 77.5533},
    {"zone": "BTM Layout", "lat": 12.9166, "lon": 77.6101},
    {"zone": "Yelahanka", "lat": 13.1007, "lon": 77.5963}
]

st.set_page_config(page_title="SurgeSense", layout="wide")

if "selected_date" not in st.session_state:
    st.session_state.selected_date = pd.Timestamp.now().date()

if "selected_time" not in st.session_state:
    st.session_state.selected_time = pd.Timestamp.now().time()

tabs = st.tabs(["Home", "Map & Zones", "Explainability"])

with tabs[0]:
    st.header("Predict Future Surge")

    col_date, col_time = st.columns(2)

    with col_date:
        today = pd.Timestamp.now().date()
        max_date = today + pd.Timedelta(days=5)

        selected_date = st.date_input(
            "Select Date", 
            value=st.session_state.selected_date,
            min_value=today,
            max_value=max_date,
            help="Prediction is limited to 5 days ahead due to weather forecast accuracy."
        )
        st.session_state.selected_date = selected_date

    with col_time:
        selected_time = st.time_input("Select Time", st.session_state.selected_time)
        st.session_state.selected_time = selected_time

    prediction_dt = pd.to_datetime(f"{selected_date} {selected_time}")

    zone = st.selectbox("Pickup Zone", le_zone.classes_.tolist())
    passenger_count = st.number_input("Passenger Count", 1, 6, 1)
    base_fare = st.number_input("Base Fare (₹)", 50, 1000, 150)

    if st.button("Predict Surge"):
        zone_info = next(z for z in bengaluru_zones if z["zone"] == zone)
        temp, weather_desc = get_future_weather(zone_info["lat"], zone_info["lon"], prediction_dt)
        traffic = estimate_future_traffic(prediction_dt, zone)

        if weather_desc not in le_weather.classes_:
            weather_desc = "Clear"

        df_input = build_input_df(prediction_dt, zone, temp, weather_desc, traffic, passenger_count)

        pred, proba = predict_from_model(model, df_input)
        pred = int(pred[0])
        surge_label = le_surge.inverse_transform([pred])[0]

        multipliers = {"Low": 1.0, "Medium": 1.3, "High": 1.8}
        mult = multipliers[surge_label]

        st.metric("Predicted Surge", surge_label)
        p = proba[0]
        st.write(f"Probabilities — Low: {p[0]:.2f}, Medium: {p[1]:.2f}, High: {p[2]:.2f}")
        st.write(f"Estimated Fare: ₹{base_fare * mult:.0f} (×{mult})")

        with st.expander("Check Price for 1 Hour Later"):
            future_dt = prediction_dt + pd.Timedelta(hours=1)
            
            f_traffic = estimate_future_traffic(future_dt, zone)
            f_temp, f_weather = get_future_weather(zone_info["lat"], zone_info["lon"], future_dt)
            
            if f_weather not in le_weather.classes_:
                f_weather = "Clear"
            
            df_future = build_input_df(future_dt, zone, f_temp, f_weather, f_traffic, passenger_count)
            
            f_pred, _ = predict_from_model(model, df_future)
            f_label = le_surge.inverse_transform([int(f_pred[0])])[0]
            
            st.write(f"Prediction for **{future_dt.strftime('%H:%M')}**: {f_label}")
            
            if surge_label == "High" and f_label != "High":
                st.success(f"Money Saver: If you wait until {future_dt.strftime('%H:%M')}, the surge drops to {f_label}!")
            elif surge_label == f_label:
                st.info("The price stays the same. No need to wait.")
            else:
                st.warning("Price is going UP later. Book now!")

with tabs[1]:
    st.header("Live Surge Map of Bengaluru")
    
    prediction_dt = pd.to_datetime(f"{st.session_state.selected_date} {st.session_state.selected_time}")
    
    st.write(f"Showing surge prediction for: {prediction_dt.strftime('%Y-%m-%d %H:%M')}")
    st.write(f"Hour: {prediction_dt.hour}, Day of week: {prediction_dt.weekday()}, Base traffic estimate: {estimate_future_traffic(prediction_dt)}")
    
    zone_df = pd.DataFrame(bengaluru_zones)

    preds = []
    labels = []
    multipliers = []
    debug_info = []

    for _, row in zone_df.iterrows():
        zone_temp, zone_weather = get_future_weather(row["lat"], row["lon"], prediction_dt)
        zone_traffic = estimate_future_traffic(prediction_dt, row["zone"])

        if zone_weather not in le_weather.classes_:
            zone_weather = "Clear"

        df_input = build_input_df(
            prediction_dt,
            row["zone"],
            zone_temp,
            zone_weather,
            zone_traffic,
            1
        )

        pred, proba = predict_from_model(model, df_input)
        pred = int(pred[0])
        
        surge_label = le_surge.inverse_transform([pred])[0]
        
        multipliers_map = {"Low": 1.0, "Medium": 1.3, "High": 1.8}
        multiplier = multipliers_map[surge_label]

        preds.append(pred)
        labels.append(surge_label)
        multipliers.append(round(multiplier, 2))
        
        debug_info.append({
            "zone": row["zone"],
            "temp": round(zone_temp, 1),
            "weather": zone_weather,
            "traffic": zone_traffic,
            "prob_low": round(proba[0][0], 3),
            "prob_med": round(proba[0][1], 3),
            "prob_high": round(proba[0][2], 3)
        })

    zone_df["pred"] = preds
    zone_df["surge_label"] = labels
    zone_df["multiplier"] = multipliers
    zone_df["color"] = zone_df["surge_label"].map({"Low": "green", "Medium": "orange", "High": "red"})

    st.subheader("Surge Predictions by Zone")
    st.dataframe(zone_df[["zone", "surge_label", "multiplier"]])

    m = folium.Map(location=[12.97, 77.59], zoom_start=12)

    for _, r in zone_df.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=12,
            color=r["color"],
            fill=True,
            fill_color=r["color"],
            fill_opacity=0.8,
            popup=f"{r['zone']} — {r['surge_label']} — x{r['multiplier']}"
        ).add_to(m)

    st_folium(m, width=800, height=550)

with tabs[2]:
    st.header("Explainability (SHAP)")
    import shap
    import matplotlib.pyplot as plt

    data_path = ROOT / "data" / "processed" / "merged_data.csv"
    full = pd.read_csv(data_path)
    
    full["zone"] = le_zone.transform(full["zone"])
    full["weather"] = le_weather.transform(full["weather"])
    
    X = full[feature_names].sample(500, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        target_index = 2 if len(shap_values) > 2 else -1
        shap_vals_target = shap_values[target_index]
    else:
        shap_vals_target = shap_values

    st.subheader("What drives High Surge?")
    
    if shap_vals_target.shape[1] != X.shape[1]:
        st.warning(f"Mismatch detected. Plotting without feature values.")
        shap.summary_plot(shap_vals_target, show=False)
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        shap.summary_plot(shap_vals_target, X, show=False)
        st.pyplot(plt.gcf())
        plt.clf()