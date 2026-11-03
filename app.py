import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit Page Configuration
st.set_page_config(
    page_title="Stock Charting & Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Charting & Prediction Application")
st.write("Analyze historical stock trends and view forecasts directly from your dashboard.")

# Sidebar Controls for User Input
st.sidebar.header("Configuration")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()
period_option = st.sidebar.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)

# Main Content Area
if st.sidebar.button("Fetch Stock Data"):
    with st.spinner(f"Fetching data for {ticker_symbol}..."):
        try:
            # Fetch data using yfinance
            stock = yf.Ticker(ticker_symbol)
            df = stock.history(period=period_option)
            
            if df.empty:
                st.error(f"No historical data found for ticker '{ticker_symbol}'. Please check the symbol and try again.")
            else:
                st.success(f"Successfully loaded data for {ticker_symbol}!")
                
                # Display Metrics
                latest_close = df['Close'].iloc[-1]
                prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_close
                price_change = latest_close - prev_close
                percent_change = (price_change / prev_close) * 100
                
                col1, col2 = st.columns(2)
                col1.metric(label="Latest Close Price", value=f"${latest_close:.2f}", delta=f"{price_change:.2f} ({percent_change:.2f}%)")
                col2.metric(label="Total Records Loaded", value=len(df))
                
                # Plotting Historical Trend Chart
                st.subheader(f"Historical Trend for {ticker_symbol}")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df.index, df['Close'], label="Close Price", color="#1f77b4", linewidth=2)
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.set_title(f"{ticker_symbol} Stock Price ({period_option})")
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend()
                
                st.pyplot(fig)
                
                # Show Raw Data Table in an Expander
                with st.expander("View Raw Data Table"):
                    st.dataframe(df)
                    
        except Exception as e:
                st.error(f"An error occurred while fetching data: {e}")
else:
    st.info("Enter a stock ticker in the sidebar and click **Fetch Stock Data** to begin.")
