import yfinance as yf
from sqlalchemy import create_engine, text
from datetime import datetime
from config import TICKERS, DATABASE_URL

def fetch_and_store_prices():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        for ticker in TICKERS:
            try:
                data = yf.Ticker(ticker).history(period="1d", interval="5m")
                if data.empty:
                    print(f"No data for {ticker}")
                    continue
                latest = data.iloc[-1]
                timestamp = data.index[-1].strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(text("""
                    INSERT INTO stock_prices (ticker, timestamp, open, high, low, close, volume)
                    VALUES (:ticker, :timestamp, :open, :high, :low, :close, :volume)
                """), {
                    "ticker": ticker, "timestamp": timestamp,
                    "open": float(latest['Open']), "high": float(latest['High']),
                    "low": float(latest['Low']), "close": float(latest['Close']),
                    "volume": int(latest['Volume'])
                })
                print(f"{ticker}: {timestamp} close={latest['Close']:.2f}")
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
        conn.commit()

if __name__ == "__main__":
    fetch_and_store_prices()
