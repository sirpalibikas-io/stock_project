from fastapi import FastAPI, HTTPException
import yfinance as yf
import pandas as pd

app = FastAPI(title="Stock Charting API", version="1.0")

@app.get("/")
def read_root():
    return {"message": "Stock Charting FastAPI backend is running successfully."}

@app.get("/stock/{ticker}")
def get_stock_data(ticker: str, period: str = "1mo"):
    """
    Fetch historical stock data for a given ticker symbol.
    Example: /stock/AAPL?period=1mo
    """
    try:
        stock = yf.Ticker(ticker.upper())
        df = stock.history(period=period)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker '{ticker}'")
        
        # Format dataframe for JSON response
        df.reset_index(inplace=True)
        # Convert timestamps to string format for compatibility
        df['Date'] = df['Date'].astype(str)
        
        records = df.to_dict(orient="records")
        return {
            "ticker": ticker.upper(),
            "period": period,
            "data": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
