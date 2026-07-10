import schedule
import time
from ingest_prices import fetch_and_store_prices
from ingest_news import fetch_and_store_news

schedule.every(15).minutes.do(fetch_and_store_prices)
schedule.every(60).minutes.do(fetch_and_store_news)

print("Scheduler started. Press Ctrl+C to stop.")
fetch_and_store_prices()
fetch_and_store_news()

while True:
    schedule.run_pending()
    time.sleep(30)
