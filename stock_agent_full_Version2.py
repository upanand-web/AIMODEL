from fastapi import FastAPI, Request
import requests
import os

app = FastAPI()
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_API_KEY')

@app.post("/stock-info")
async def stock_info(request: Request):
    data = await request.json()
    ticker = data.get("ticker")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={API_KEY}"
    try:
        resp = requests.get(url)
        stock_data = resp.json().get("Global Quote", {})
        if not stock_data or not stock_data.get("05. price"):
            return {"error": f"No data found. Try another ticker or check symbol on Alpha Vantage."}
        return {
            "ticker": ticker,
            "price": stock_data.get("05. price"),
            "open": stock_data.get("02. open"),
            "high": stock_data.get("03. high"),
            "low": stock_data.get("04. low"),
            "volume": stock_data.get("06. volume"),
        }
    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}

@app.post("/market-predict")
async def market_predict(request: Request):
    data = await request.json()
    ticker = data.get("ticker")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={API_KEY}"
    try:
        resp = requests.get(url)
        ts = resp.json().get("Time Series (Daily)", {})
        days = sorted(ts.keys())
        if len(days) < 2:
            return {"error": "Not enough data"}
        yesterday = float(ts[days[-2]]["4. close"])
        today = float(ts[days[-1]]["4. close"])
        prediction = "up" if today > yesterday else "down"
        return {
            "ticker": ticker,
            "yesterday_close": yesterday,
            "today_close": today,
            "prediction": prediction
        }
    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}

@app.get("/hello")
async def hello():
    return {"message": "Your API is working!"}
