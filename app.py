import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import joblib
import plotly.graph_objects as go
from datetime import datetime
import os

st.set_page_config(page_title="Real-time stock sentiment predictor", layout="wide")

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ["DATABASE_URL"]

MODEL_PATH = "stock_sentiment_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data(ttl=300)
def load_data():
    engine = create_engine(DATABASE_URL)
    prices = pd.read_sql("SELECT * FROM stock_prices", engine)
    news = pd.read_sql("SELECT * FROM news_sentiment", engine)

    prices['timestamp'] = pd.to_datetime(prices['timestamp'])
    news['published_at'] = pd.to_datetime(news['published_at'], utc=True).dt.tz_localize(None)
    prices['date'] = prices['timestamp'].dt.date
    news['date'] = news['published_at'].dt.date

    prices = prices.sort_values(['ticker', 'date'])
    daily_prices = prices.groupby(['ticker', 'date']).agg(
        day_open=('open', 'first'), day_close=('close', 'last'),
        day_high=('high', 'max'), day_low=('low', 'min'), day_volume=('volume', 'sum')
    ).reset_index()
    daily_prices['daily_return'] = (daily_prices['day_close'] - daily_prices['day_open']) / daily_prices['day_open']
    daily_prices['ma_7'] = daily_prices.groupby('ticker')['day_close'].transform(lambda x: x.rolling(7).mean())
    daily_prices['ma_14'] = daily_prices.groupby('ticker')['day_close'].transform(lambda x: x.rolling(14).mean())
    daily_prices['volatility_7'] = daily_prices.groupby('ticker')['daily_return'].transform(lambda x: x.rolling(7).std())

    daily_sentiment = news.groupby(['ticker', 'date']).agg(
        avg_sentiment=('sentiment_score', 'mean'),
        num_articles=('sentiment_score', 'count'),
        positive_ratio=('sentiment_label', lambda x: (x == 'positive').mean())
    ).reset_index()

    model_df = pd.merge(daily_prices, daily_sentiment, on=['ticker', 'date'], how='left')
    model_df['avg_sentiment'] = model_df['avg_sentiment'].fillna(0)
    model_df['num_articles'] = model_df['num_articles'].fillna(0)
    model_df['positive_ratio'] = model_df['positive_ratio'].fillna(0)

    return model_df, prices, news

model = load_model()
model_df, prices, news = load_data()

st.title("Real-time stock sentiment predictor")
st.caption("Combines live news sentiment with price technical indicators to predict next-day direction")

tickers = sorted(model_df['ticker'].unique())
selected_ticker = st.sidebar.selectbox("Select ticker", tickers)

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(f"Data last loaded: {datetime.now().strftime('%H:%M:%S')}")
st.sidebar.info("Data updates automatically every 30 minutes via GitHub Actions.")

ticker_df = model_df[model_df['ticker'] == selected_ticker].sort_values('date')
ticker_news = news[news['ticker'] == selected_ticker].sort_values('published_at', ascending=False)
latest = ticker_df.iloc[-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest close", f"${latest['day_close']:.2f}")
col2.metric("Daily return", f"{latest['daily_return']*100:.2f}%")
col3.metric("Avg sentiment", f"{latest['avg_sentiment']:.3f}")
col4.metric("Articles today", f"{int(latest['num_articles'])}")

st.subheader(f"{selected_ticker} price with moving averages")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=ticker_df['date'], y=ticker_df['day_close'], name="Close"))
fig_price.add_trace(go.Scatter(x=ticker_df['date'], y=ticker_df['ma_7'], name="7-day MA"))
fig_price.add_trace(go.Scatter(x=ticker_df['date'], y=ticker_df['ma_14'], name="14-day MA"))
fig_price.update_layout(height=400, xaxis_title="Date", yaxis_title="Price ($)")
st.plotly_chart(fig_price, use_container_width=True)

st.subheader(f"{selected_ticker} sentiment vs. price trend")
fig_sent = go.Figure()
fig_sent.add_trace(go.Bar(x=ticker_df['date'], y=ticker_df['avg_sentiment'], name="Avg sentiment", yaxis="y2", opacity=0.4))
fig_sent.add_trace(go.Scatter(x=ticker_df['date'], y=ticker_df['day_close'], name="Close price"))
fig_sent.update_layout(
    height=400,
    yaxis=dict(title="Price ($)"),
    yaxis2=dict(title="Sentiment", overlaying="y", side="right"),
)
st.plotly_chart(fig_sent, use_container_width=True)

st.subheader("Tomorrow's prediction")
feature_cols = ['avg_sentiment', 'num_articles', 'positive_ratio',
                 'day_volume', 'daily_return', 'ma_7', 'ma_14', 'volatility_7']

input_row = latest[feature_cols].to_frame().T
for t in tickers:
    input_row[f"ticker_{t}"] = 1 if t == selected_ticker else 0
input_row = input_row.reindex(columns=model.feature_names_in_, fill_value=0)

pred = model.predict(input_row)[0]
proba = model.predict_proba(input_row)[0]

if pred == 1:
    st.success(f"Model predicts price UP tomorrow ({proba[1]*100:.1f}% confidence)")
else:
    st.error(f"Model predicts price DOWN tomorrow ({proba[0]*100:.1f}% confidence)")

st.caption("Trained on a small early dataset — this is a pipeline demo, not financial advice.")

st.subheader("Recent headlines")
st.dataframe(ticker_news[['published_at', 'headline', 'sentiment_label', 'source']].head(10), use_container_width=True)
