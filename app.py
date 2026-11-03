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
st.title("📈 Live Stock Price Prediction Dashboard")
st.write("Using pre-trained PyTorch LSTM model (`lstm_best.pt`) and `scaler.pkl`.")

# -------------------------------------------------------------
# 2. DEFINE MODEL ARCHITECTURE (Matches 5 Input Features)
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

# -------------------------------------------------------------
# 3. LOAD ARTIFACTS SAFELY
# -------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler_path = "scaler.pkl"
    model_path = "lstm_best.pt"
    
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    model = StockLSTM(input_dim=5)
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            if isinstance(state_dict, dict) or 'OrderedDict' in str(type(state_dict)):
                model.load_state_dict(state_dict)
            else:
                model = state_dict # In case full model was saved
            model.eval()
        except Exception as e:
            st.error(f"Model load warning: {e}")
            
    return model, scaler

model, scaler = load_artifacts()

# -------------------------------------------------------------
# 4. SIDEBAR SETTINGS
# -------------------------------------------------------------
st.sidebar.header("Prediction Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="TCS.NS").upper()
forecast_days = st.sidebar.slider("Days Ahead to Forecast", min_value=1, max_value=14, value=5)
seq_length = 60

# -------------------------------------------------------------
# 5. PREDICTION EXECUTION
# -------------------------------------------------------------
if st.sidebar.button("Run Model Prediction"):
    with st.spinner(f"Fetching 5-feature data and running inference for {ticker}..."):
        try:
            df = yf.Ticker(ticker).history(period="6mo")
            
            if df.empty or len(df) < seq_length:
                st.error(f"Not enough data for '{ticker}'. Need at least {seq_length} days.")
            else:
                # Ensure we capture the 5 typical columns used in training: Open, High, Low, Close, Volume
                # (Adjust column names if your notebook used specific indicators, typically these 5)
                feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                # Check if all columns exist, otherwise fallback or map
                available_cols = [col for col in feature_columns if col in df.columns]
                
                if len(available_cols) < 5:
                    st.warning(f"Found {len(available_cols)} columns, but scaler expects 5. Using available numerical data.")
                
                input_data = df[available_cols].values
                
                if scaler is None or model is None:
                    st.error("Missing `scaler.pkl` or `lstm_best.pt` in repository.")
                else:
                    # Scale the 5 features
                    scaled_data = scaler.transform(input_data)
                    
                    last_sequence = scaled_data[-seq_length:]
                    current_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)
                    
                    future_predictions_scaled = []
                    with torch.no_grad():
                        for _ in range(forecast_days):
                            pred = model(current_seq)
                            future_predictions_scaled.append(pred.item())
                            
                            # Create a dummy row for the next step prediction feedback loop
                            # We replicate the last known feature row and replace the Close price (or target index) with prediction
                            next_row = current_seq[0, -1, :].clone().numpy()
                            next_row[3] = pred.item() # Assuming Close is index 3
                            
                            new_pred_tensor = torch.tensor(next_row, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                            current_seq = torch.cat((current_seq[:, 1:, :], new_pred_tensor), dim=1)
                    
                    # Inverse transform predictions back to price scale
                    # Since scaler expects 5 columns, we pad the prediction column back to shape (N, 5) for inverse transform
                    dummy_matrix = np.zeros((len(future_predictions_scaled), 5))
                    dummy_matrix[:, 3] = future_predictions_scaled # Assuming Close is index 3
                    inv_transformed = scaler.inverse_transform(dummy_matrix)
                    future_prices = inv_transformed[:, 3]
                    
                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')
                    
                    # Display Metrics
                    col1, col2, col3 = st.columns(3)
                    latest_price = df['Close'].iloc[-1]
                    predicted_final = future_prices[-1]
                    price_diff = predicted_final - latest_price
                    pct_diff = (price_diff / latest_price) * 100
                    
                    col1.metric("Current Close Price", f"${latest_price:.2f}")
                    col2.metric(f"Predicted Price ({forecast_days} Days)", f"${predicted_final:.2f}", delta=f"{price_diff:+.2f} ({pct_diff:+.2f}%)")
                    col3.metric("Status", "Inference Success ✅")
                    
                    # Plot Chart
                    st.subheader(f"📊 {ticker} Forecast Chart")
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(df.index[-60:], df['Close'].iloc[-60:], label="Historical Close", color="#1f77b4", linewidth=2)
                    
                    forecast_x = [last_date] + list(future_dates)
                    forecast_y = [latest_price] + list(future_prices)
                    ax.plot(forecast_x, forecast_y, label="LSTM Forecast", color="#ff7f0e", linestyle="--", marker="o", linewidth=2)
                    
                    ax.grid(True, linestyle="--", alpha=0.5)
                    ax.legend()
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f5"Prediction error: {e}")
else:
    st.info("Click **Run Model Prediction** on the sidebar to start forecasting.")
