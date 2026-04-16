#  SurgeSense

SurgeSense is a machine learning-based web app that predicts cab surge pricing and helps users decide when and where to book a ride. Instead of getting surprised by high prices, users can plan ahead and save money.

---

## Features

- Predicts surge level (Low / Medium / High)
- Shows estimated fare using surge multipliers
- Suggests better times to book rides
- Recommends cheaper nearby pickup zones
- Displays a zone-wise surge map of Bengaluru
- Includes model explainability using SHAP

---

## Tech Stack

- LightGBM, scikit-learn  
- pandas, numpy  
- Streamlit  
- folium, matplotlib  
- OpenWeatherMap API  
- joblib  

---

## How it works

- User selects date, time, and pickup zone  
- Weather data is fetched using API  
- Traffic is estimated based on time and zone  
- Inputs are encoded and passed to the model  
- Model predicts surge level and probabilities  
- App suggests whether to book now or wait  

---

## Setup

Clone the repository and install dependencies:

git clone https://github.com/your-username/surgesense.git  
cd surgesense  
pip install -r requirements.txt  

---

## Before Running

Make sure the following files exist inside the `models/` folder:

- surge_model.txt  
- encoders.pkl  
- feature_names.pkl  
- surge_label_encoder.pkl  

If these are missing, run the notebooks to generate them.

---

## Run the App

streamlit run app/streamlit_app.py  

---

## API Key

Create a file: `.streamlit/secrets.toml`

OPENWEATHER_API_KEY = "your_api_key_here"

---


## Running from Scratch

If you don’t have trained model files:

1. Run notebook 1 (data preparation)  
2. Run notebook 2 (model training)  
3. Then start the Streamlit app  

---

## Note

This project uses simulated data and estimated traffic values. It models realistic surge behavior but does not reflect actual Uber/Ola pricing.

---

## Contributing

Contributions are welcome.

- Check open issues  
- Look for "good-first-issue" labels  
- Improve documentation, UI, or features  
- Raise issues if you find bugs  

---

## Future Scope

- Real-time traffic integration  
- Event-based surge prediction  
- Multi-city support  

---

## License

For learning and academic use.
