from sqlalchemy import create_engine, text
from config import DATABASE_URL

def create_tables():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            id SERIAL PRIMARY KEY,
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume BIGINT
        )
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id SERIAL PRIMARY KEY,
            ticker TEXT NOT NULL,
            headline TEXT NOT NULL,
            source TEXT,
            published_at TEXT,
            sentiment_score REAL,
            sentiment_label TEXT,
            fetched_at TEXT
        )
        """))
        conn.commit()
    print("Postgres tables created.")

if __name__ == "__main__":
    create_tables()
