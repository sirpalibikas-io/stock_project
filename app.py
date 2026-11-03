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
# 1. PAGE SETUP & TITLE
# -------------------------------------------------------------
st.set_page_config(page_title="Stock Prediction Dashboard", page_icon="📈", layout="wide")
st.title("📈 Live Stock Price Prediction Dashboard")
st.write("Using pre-trained PyTorch LSTM model (`lstm_best.pt`) and `scaler.pkl` to forecast future prices.")

# -------------------------------------------------------------
# 2. DEFINE MODEL ARCHITECTURE
# (Must match the structure used during training in stock.ipynb)
# -------------------------------------------------------------
class StockLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# -------------------------------------------------------------
# 3. LOAD SCALER AND MODEL WEIGHTS
# -------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler_path = "scaler.pkl"
    model_path = "lstm_best.pt"
    
    scaler = None
    model = None
    
    # Load Scaler
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.error(f"Error loading scaler.pkl: {e}")
            
    # Load PyTorch Weights
    if os.path.exists(model_path):
        try:
            model = StockLSTM()
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Load weights (handles both full model save and state_dict)
            if isinstance(state_dict, StockLSTM):
                model = state_dict
            else:
                model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            # Fallback if state dict structure differs slightly
            try:
                model = torch.load(model_path, map_location=torch.device('cpu'))
                model.eval()
            except Exception as ex:
                st.error(f"Error loading model weights: {ex}")
                
    return model, scaler

model, scaler = load_artifacts()

# -------------------------------------------------------------
# 4. SIDEBAR INPUTS
# -------------------------------------------------------------
st.sidebar.header("Prediction Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="TCS.NS").upper()
forecast_days = st.sidebar.slider("Days Ahead to Forecast", min_value=1, max_value=14, value=5)
seq_length = 60  # Default lookback window length

# -------------------------------------------------------------
# 5. PREDICTION LOGIC & DASHBOARD
# -------------------------------------------------------------
if st.sidebar.button("Run Model Prediction"):
    with st.spinner(f"Fetching data and running inference for {ticker}..."):
        try:
            # Fetch 6 months of data for lookback sequence
            df = yf.Ticker(ticker).history(period="6mo")
            
            if df.empty or len(df) < seq_length:
                st.error(f"Not enough data for ticker '{ticker}'. At least {seq_length} trading days needed.")
            else:
                close_prices = df[['Close']].values
                
                # Check if artifacts exist
                if scaler is None or model is None:
                    st.warning("⚠️ `scaler.pkl` or `lstm_best.pt` file not found in repository root. Showing historical data only.")
                else:
                    # Step A: Scale the closing prices
                    scaled_data = scaler.transform(close_prices)
                    
                    # Step B: Prepare the lookback sequence (last N days)
                    last_sequence = scaled_data[-seq_length:]
                    current_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)  # Shape: (1, seq_length, 1)
                    
                    # Step C: Generate iterative future predictions
                    future_predictions_scaled = []
                    
                    with torch.no_grad():
                        for _ in range(forecast_days):
                            pred = model(current_seq) # Output shape: (1, 1)
                            future_predictions_scaled.append(pred.item())
                            
                            # Update lookback sequence with new prediction
                            new_pred_tensor = pred.unsqueeze(1) # Shape: (1, 1, 1)
                            current_seq = torch.cat((current_seq[:, 1:, :], new_pred_tensor), dim=1)
                    
                    # Step D: Inverse transform scaled predictions back to actual dollar amounts
                    future_predictions_scaled = np.array(future_predictions_scaled).reshape(-1, 1)
                    future_predictions = scaler.inverse_transform(future_predictions_scaled).flatten()
                    
                    # Step E: Build Future Dates index
                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')
                    
                    # Metrics Display
                    col1, col2, col3 = st.columns(3)
                    latest_price = close_prices[-1][0]
                    predicted_final = future_predictions[-1]
                    price_diff = predicted_final - latest_price
                    pct_diff = (price_diff / latest_price) * 100
                    
                    col1.metric("Current Close Price", f"${latest_price:.2f}")
                    col2.metric(f"Predicted Price ({forecast_days} Days)", f"${predicted_final:.2f}", delta=f"{price_diff:+.2f} ({pct_diff:+.2f}%)")
                    col3.metric("Model Status", "Inference Active ✅")
                    
                    # Plotting Historical vs Predicted Chart
                    st.subheader(f"📊 Historical Price + {forecast_days}-Day Forecast for {ticker}")
                    
                    fig, ax = plt.subplots(figsize=(12, 5))
                    
                    # Plot recent 60 days historical price
                    ax.plot(df.index[-60:], df['Close'].iloc[-60:], label="Historical Close Price", color="#1f77b4", linewidth=2)
                    
                    # Plot predictions starting from last historical point to forecast dates
                    forecast_x = [last_date] + list(future_dates)
                    forecast_y = [latest_price] + list(future_predictions)
                    ax.plot(forecast_x, forecast_y, label="PyTorch LSTM Forecast", color="#ff7f0e", linestyle="--", marker="o", linewidth=2)
                    
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price ($)")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    # Show Prediction Data Table
                    with st.expander("View Forecast Values Table"):
                        forecast_df = pd.DataFrame({"Date": future_dates.strftime('%Y-%m-%d'), "Predicted Close Price": future_predictions})
                        st.dataframe(forecast_df)
                        
        except Exception as e:
            st.error(f"Error during prediction execution: {e}")
else:
    st.info("Select options on the sidebar and click **Run Model Prediction** to start model inference.")
