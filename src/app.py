import os
import pandas as pd
from flask import Flask, request, jsonify
import pickle
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

print(f"DEBUG: App's Current Working Directory: {os.getcwd()}")

MODEL_PATH = 'models/arima_sales_model.pkl'
# --- CHANGE THIS ---
DATASET_URL = 'https://raw.githubusercontent.com/4GeeksAcademy/alternative-time-series-project/main/sales.csv'
# --- END CHANGE ---

model_fit = None
last_known_date = None

def load_resources():
    global model_fit, last_known_date

    # Load the trained ARIMA model
    try:
        with open(MODEL_PATH, 'rb') as pkl:
            model_fit = pickle.load(pkl)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. "
              f"Please ensure it exists and Step 5 in your notebook was run. "
              f"Looked in: {os.path.abspath(MODEL_PATH)}")
        model_fit = None
    except Exception as e:
        print(f"Error loading model: {e}")
        model_fit = None

    # Load the sales data directly from the URL
    try:
        # --- CHANGE THIS ---
        df = pd.read_csv(DATASET_URL)
        print(f"Dataset loaded successfully from URL: {DATASET_URL}")
        # --- END CHANGE ---

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        last_known_date = df.index[-1]
        print(f"Last known date from sales data: {last_known_date}")
    except KeyError:
        print("Error: 'date' or 'sales' column not found in the online dataset. Check its structure.")
        last_known_date = None
    except Exception as e:
        print(f"Error loading sales data from URL: {e}")
        last_known_date = None


with app.app_context():
    load_resources()

@app.route('/')
def home():
    return "Sales Forecasting API. Use /predict?days=N to get future predictions."

@app.route('/predict', methods=['GET'])
def predict_sales():
    if model_fit is None or last_known_date is None:
        return jsonify({"error": "Model or data not loaded. Check server logs for details."}), 500

    try:
        days_to_forecast = int(request.args.get('days', 7))
        if days_to_forecast <= 0:
            return jsonify({"error": "Number of days to forecast must be a positive integer."}), 400

        forecast_result = model_fit.forecast(steps=days_to_forecast)

        predictions = []
        for i, val in enumerate(forecast_result):
            forecast_date = last_known_date + pd.Timedelta(days=i + 1)
            predictions.append({
                "date": forecast_date.strftime('%Y-%m-%d'),
                "sales_prediction": round(val, 2)
            })

        return jsonify({"forecast": predictions}), 200

    except ValueError:
        return jsonify({"error": "Invalid value for 'days'. Please provide an integer."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)