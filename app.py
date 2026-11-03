import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. PAGE SETUP
# -------------------------------------------------------------
st.set_page_config(page_title="Stock Prediction Dashboard", page_icon="📈", layout="wide")
st.title("📈 Final PyTorch Stock Prediction Dashboard")
st.write("Forecast future stock prices using your trained Deep Learning models and feature scalers.")

# -------------------------------------------------------------
# 2. DEFINE MODEL ARCHITECTURES (Matches 5 Input Features)
# -------------------------------------------------------------
class StockLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=1):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class StockANN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1):
        super(StockANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x[:, -1, :] # Take the last timestep if sequence provided
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# -------------------------------------------------------------
# 3. LOAD ARTIFACTS SAFELY
# -------------------------------------------------------------
@st.cache_resource
def load_artifacts(model_choice):
    scaler_path = "scaler.pkl"
    model_file = "lstm_best.pt" if model_choice == "LSTM (lstm_best.pt)" else "ann_best.pt"
    
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    if model_choice == "LSTM (lstm_best.pt)":
        model = StockLSTM(input_dim=5)
    else:
        model = StockANN(input_dim=5)
        
    if os.path.exists(model_file):
        try:
            state_dict = torch.load(model_file, map_location=torch.device('cpu'))
            if isinstance(state_dict, dict) or 'OrderedDict' in str(type(state_dict)):
                model.load_state_dict(state_dict)
            else:
                model = state_dict
            model.eval()
        except Exception as e:
            st.warning(f"Could not load state dict directly for {model_file}, trying full load: {e}")
            try:
                model = torch.load(model_file, map_location=torch.device('cpu'))
                model.eval()
            except Exception as ex:
                st.error(f"Error loading model: {ex}")
                
    return model, scaler

# -------------------------------------------------------------
# 4. SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.header("⚙️ Dashboard Controls")

# Model Choice
model_selection = st.sidebar.selectbox(
    "Select Model Architecture", 
    ["LSTM (lstm_best.pt)", "ANN (ann_best.pt)"]
)

# Load selected model and scaler
model, scaler = load_artifacts(model_selection)

# Ticker Choice (Popular Options + Custom)
ticker_option = st.sidebar.selectbox(
    "Select Stock Ticker", 
    ["AAPL (Apple)", "TSLA (Tesla)", "MSFT (Microsoft)", "GOOGL (Google)", "TCS.NS (TCS)", "Custom Input"]
)

if ticker_option == "Custom Input":
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
else:
    ticker = ticker_option.split(" ")[0]

forecast_days = st.sidebar.slider("Days Ahead to Forecast", min_value=1, max_value=14, value=5)
seq_length = 60

# -------------------------------------------------------------
# 5. PREDICTION & VISUALIZATION LOGIC
# -------------------------------------------------------------
if st.sidebar.button("Run Model Prediction"):
    with st.spinner(f"Running inference for {ticker} using {model_selection}..."):
        try:
            df = yf.Ticker(ticker).history(period="6mo")
            
            if df.empty or len(df) < seq_length:
                st.error(f"Not enough historical data found for '{ticker}'. Choose another ticker or increase timeframe.")
            else:
                feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                available_cols = [col for col in feature_columns if col in df.columns]
                
                input_data = df[available_cols].values
                
                if scaler is None or model is None:
                    st.error("Missing `scaler.pkl` or model weight file in repository root.")
                else:
                    # Scale 5 features
                    scaled_data = scaler.transform(input_data)
                    
                    last_sequence = scaled_data[-seq_length:]
                    current_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)
                    
                    future_predictions_scaled = []
                    with torch.no_grad():
                        for _ in range(forecast_days):
                            pred = model(current_seq)
                            future_predictions_scaled.append(pred.item())
                            
                            # Feedback loop for multi-step forecasting
                            next_row = current_seq[0, -1, :].clone().numpy()
                            next_row[3] = pred.item() # Assuming Close is index 3
                            
                            new_pred_tensor = torch.tensor(next_row, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                            current_seq = torch.cat((current_seq[:, 1:, :], new_pred_tensor), dim=1)
                    
                    # Inverse transform predictions
                    dummy_matrix = np.zeros((len(future_predictions_scaled), 5))
                    dummy_matrix[:, 3] = future_predictions_scaled
                    inv_transformed = scaler.inverse_transform(dummy_matrix)
                    future_prices = inv_transformed[:, 3]
                    
                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')
                    
                    # Metrics Display
                    col1, col2, col3 = st.columns(3)
                    latest_price = df['Close'].iloc[-1]
                    predicted_final = future_prices[-1]
                    price_diff = predicted_final - latest_price
                    pct_diff = (price_diff / latest_price) * 100
                    
                    col1.metric("Current Close Price", f"${latest_price:.2f}")
                    col2.metric(f"Predicted Price ({forecast_days} Days)", f"${predicted_final:.2f}", delta=f"{price_diff:+.2f} ({pct_diff:+.2f}%)")
                    col3.metric("Active Model", model_selection)
                    
                    # Plotting Chart
                    st.subheader(f"📊 {ticker} Historical & Forecast Trend ({model_selection})")
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(df.index[-60:], df['Close'].iloc[-60:], label="Historical Close", color="#1f77b4", linewidth=2)
                    
                    forecast_x = [last_date] + list(future_dates)
                    forecast_y = [latest_price] + list(future_prices)
                    ax.plot(forecast_x, forecast_y, label="Model Forecast", color="#ff7f0e", linestyle="--", marker="o", linewidth=2)
                    
                    ax.grid(True, linestyle="--", alpha=0.5)
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Data Table Expander
                    with st.expander("View Numerical Forecast Table"):
                        forecast_df = pd.DataFrame({
                            "Date": future_dates.strftime('%Y-%m-%d'), 
                            "Predicted Close Price": future_prices
                        })
                        st.dataframe(forecast_df)
                        
        except Exception as e:
            st.error(f"Prediction error: {e}")
else:
    st.info("Select your model and ticker from the sidebar, then click **Run Model Prediction**.")
