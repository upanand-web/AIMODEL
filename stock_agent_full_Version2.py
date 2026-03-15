from fastapi import FastAPI, Request
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

app = FastAPI()

@app.post("/stock-info")
async def stock_info(request: Request):
    data = await request.json()
    ticker = data.get("ticker")
    try:
        info = yf.Ticker(ticker).info
        price = info.get("currentPrice")
        market_cap = info.get("marketCap")
        sector = info.get("sector")
        company = info.get("shortName")
        if price is None:
            return {"error": "Price not found for ticker. Check symbol or Yahoo API limits."}
        return {
            "ticker": ticker,
            "price": price,
            "market_cap": market_cap,
            "sector": sector,
            "company": company,
        }
    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}

@app.post("/predict-price")
async def predict_price(request: Request):
    data = await request.json()
    ticker = data["ticker"]
    period = data.get("period", "1y")
    df = yf.download(ticker, period=period)
    closes = df['Close'].values

    look_back = 10
    X, y = [], []
    for i in range(len(closes) - look_back):
        X.append(closes[i:i+look_back])
        y.append(closes[i+look_back])
    X = np.array(X)
    y = np.array(y)

    if len(X) < 1:
        return {"error": "Not enough data!"}

    model = LinearRegression()
    model.fit(X, y)

    last_seq = closes[-look_back:].reshape(1, -1)
    pred = model.predict(last_seq)[0]
    return {
        "ticker": ticker,
        "last_close": float(closes[-1]),
        "predicted_next_close": float(pred)
    }

@app.get("/hello")
async def hello():
    return {"message": "Your API is working!"}
