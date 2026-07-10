import requests
from sqlalchemy import create_engine, text
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import NEWSAPI_KEY, TICKERS, DATABASE_URL

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_label(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"

def fetch_and_store_news():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        for ticker in TICKERS:
            url = "https://newsapi.org/v2/everything"
            params = {"q": ticker, "language": "en", "sortBy": "publishedAt",
                       "pageSize": 10, "apiKey": NEWSAPI_KEY}
            try:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                articles = resp.json().get("articles", [])
                for a in articles:
                    headline = a.get("title", "")
                    if not headline:
                        continue
                    score = analyzer.polarity_scores(headline)["compound"]
                    label = get_sentiment_label(score)
                    conn.execute(text("""
                        INSERT INTO news_sentiment
                        (ticker, headline, source, published_at, sentiment_score, sentiment_label, fetched_at)
                        VALUES (:ticker, :headline, :source, :published_at, :score, :label, :fetched_at)
                    """), {
                        "ticker": ticker, "headline": headline,
                        "source": a.get("source", {}).get("name"),
                        "published_at": a.get("publishedAt"),
                        "score": score, "label": label,
                        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                print(f"{ticker}: stored {len(articles)} headlines")
            except Exception as e:
                print(f"Error fetching news for {ticker}: {e}")
        conn.commit()

if __name__ == "__main__":
    fetch_and_store_news()
