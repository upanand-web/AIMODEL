from fastapi import FastAPI, Request
import yfinance as yf
import numpy as np
import tensorflow as tf

app = FastAPI()

@app.post("/stock-info")
async def stock_info(request: Request):
    data = await request.json()
    ticker = data["ticker"]
    info = yf.Ticker(ticker).info
    price = info.get("currentPrice")
    market_cap = info.get("marketCap")
    sector = info.get("sector")
    return {
        "ticker": ticker,
        "price": price,
        "market_cap": market_cap,
        "sector": sector,
        "company": info.get("shortName"),
    }

def create_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

@app.post("/predict-price")
async def predict_price(request: Request):
    data = await request.json()
    ticker = data["ticker"]
    period = data.get("period", "5y")
    df = yf.download(ticker, period=period)
    closes = df['Close'].values

    look_back = 10
    X, y = [], []
    for i in range(len(closes) - look_back):
        X.append(closes[i:i+look_back])
        y.append(closes[i+look_back])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = create_lstm_model((look_back, 1))
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

    last_seq = closes[-look_back:]
    last_seq = last_seq.reshape((1, look_back, 1))
    pred = model.predict(last_seq)[0][0]
    return {
        "ticker": ticker,
        "last_close": float(closes[-1]),
        "predicted_next_close": float(pred)
    }