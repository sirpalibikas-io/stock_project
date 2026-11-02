from fastapi import FastAPI, HTTPException
import pydantic
import torch
import torch.nn as nn
import joblib
import numpy as np
import os

# Initialize FastAPI
app = FastAPI(
    title="TCS Stock Prediction API",
    description="Serving a pre-trained PyTorch LSTM model.",
    version="1.0.0"
)

# --- 1. DEFINE LSTM ARCHITECTURE (Must match training architecture) ---
class StockLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- 2. LOAD PRE-TRAINED ARTIFACTS ON STARTUP ---
MODEL_PATH = "lstm_best.pt"
SCALER_PATH = "scaler.pkl"

model = None
scaler = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print("Scaler loaded successfully!")
        
        if os.path.exists(MODEL_PATH):
            model = StockLSTM(input_dim=5, hidden_dim=64, num_layers=2)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            model.eval()
            print("Pre-trained LSTM model loaded successfully!")
    except Exception as e:
        print(f"Error loading saved files: {e}")

# --- 3. INPUT DATA FORMAT ---
class PredictionRequest(pydantic.BaseModel):
    # Expects a 60-day window with 5 features [Open, High, Low, Close, Volume]
    features: list[list[float]]

@app.get("/")
def home():
    return {"status": "Online 🟢", "model": "LSTM (Pre-trained)", "target": "TCS.NS"}

# --- 4. PREDICTION ROUTE ---
@app.post("/predict")
def predict_stock(data: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model weights not found on server.")
    
    try:
        input_array = np.array(data.features, dtype=np.float32)
        
        if input_array.ndim != 2 or input_array.shape[1] != 5:
            raise HTTPException(status_code=400, detail="Invalid shape. Expected 60 rows and 5 columns.")
        
        # Add batch dimension -> Shape: (1, 60, 5)
        input_tensor = torch.tensor(input_array).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(input_tensor).item()
            
        return {
            "predicted_next_close": prediction,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))