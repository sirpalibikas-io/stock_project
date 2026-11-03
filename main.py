from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd

app = FastAPI(title="Stock Prediction Dashboard", version="2.0")

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stock Prediction & Charting Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-900 text-white font-sans min-h-screen p-6">
        <div class="max-w-4xl mx-auto">
            <header class="mb-8 text-center">
                <h1 class="text-3xl font-bold text-blue-400 mb-2">📈 Stock Prediction Dashboard</h1>
                <p class="text-gray-400">Real-time charting and historical trend analysis powered by FastAPI & Render</p>
            </header>

            <div class="bg-gray-800 p-6 rounded-xl shadow-lg mb-6 flex flex-wrap gap-4 items-center justify-between">
                <div class="flex items-center space-x-3">
                    <label for="tickerInput" class="font-semibold text-gray-300">Ticker:</label>
                    <input type="text" id="tickerInput" value="AAPL" class="bg-gray-700 text-white px-3 py-2 rounded uppercase font-bold w-28 focus:outline-none focus:ring-2 focus:ring-blue-500">
                </div>
                <div class="flex items-center space-x-3">
                    <label for="periodSelect" class="font-semibold text-gray-300">Period:</label>
                    <select id="periodSelect" class="bg-gray-700 text-white px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-blue-500">
                        <option value="1mo">1 Month</option>
                        <option value="3mo" selected>3 Months</option>
                        <option value="6mo">6 Months</option>
                        <option value="1y">1 Year</option>
                    </select>
                </div>
                <button onclick="fetchStockData()" class="bg-blue-600 hover:bg-blue-500 text-white px-5 py-2 rounded font-semibold transition">Fetch Data</button>
            </div>

            <div class="bg-gray-800 p-6 rounded-xl shadow-lg">
                <div id="metrics" class="grid grid-cols-2 gap-4 mb-6 text-center">
                    <div class="bg-gray-700 p-4 rounded-lg">
                        <p class="text-gray-400 text-sm">Latest Close Price</p>
                        <p id="latestPrice" class="text-2xl font-bold text-green-400">-</p>
                    </div>
                    <div class="bg-gray-700 p-4 rounded-lg">
                        <p class="text-gray-400 text-sm">Status</p>
                        <p id="statusText" class="text-xl font-semibold text-blue-300">Ready</p>
                    </div>
                </div>
                <div class="relative h-80">
                    <canvas id="stockChart"></canvas>
                </div>
            </div>
        </div>

        <script>
            let myChart = null;

            async function fetchStockData() {
                const ticker = document.getElementById('tickerInput').value.trim().toUpperCase();
                const period = document.getElementById('periodSelect').value;
                document.getElementById('statusText').innerText = "Loading...";

                try {
                    const response = await window.fetch(`/stock/${ticker}?period=${period}`);
                    if (!response.ok) throw new Error("Ticker not found");
                    const result = await response.json();

                    const data = result.data;
                    const labels = data.map(item => item.Date.split('T')[0]);
                    const prices = data.map(item => item.Close);

                    const latest = prices[prices.length - 1];
                    document.getElementById('latestPrice').innerText = `$${latest.toFixed(2)}`;
                    document.getElementById('statusText').innerText = "Online";

                    renderChart(labels, prices, ticker);
                } catch (error) {
                    alert("Error fetching stock data: " + error.message);
                    document.getElementById('statusText').innerText = "Error";
                }
            }

            function renderChart(labels, data, ticker) {
                const ctx = document.getElementById('stockChart').getContext('2d');
                if (myChart) myChart.destroy();

                myChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: `${ticker} Close Price`,
                            data: data,
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { grid: { color: 'rgba(255, 255, 255, 0.1)' } },
                            y: { grid: { color: 'rgba(255, 255, 255, 0.1)' } }
                        }
                    }
                });
            }

            window.onload = fetchStockData;
        </script>
    </body>
    </html>
    """

@app.get("/stock/{ticker}")
def get_stock_data(ticker: str, period: str = "3mo"):
    try:
        stock = yf.Ticker(ticker.upper())
        df = stock.history(period=period)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found")
        df.reset_index(inplace=True)
        df['Date'] = df['Date'].astype(str)
        return {"ticker": ticker.upper(), "data": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
