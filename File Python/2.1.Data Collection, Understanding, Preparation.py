import os
import pandas as pd
import numpy as np
import yfinance as yf

TICKERS = ["META", "RDDT"]
start_date = "2024-01-01"
end_date = "2025-07-01"
data_dir = "data"


def download_price_data(ticker):
    ticker_dir = os.path.join(data_dir, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    price_data = yf.download(ticker, start=start_date, end=end_date)
    price_path = os.path.join(ticker_dir, f"DATA_price.csv")
    price_data.to_csv(price_path)
    print(f"Đã lưu {price_path}")
    # Sửa dòng header thứ 3
    with open(price_path, "r") as f:
        lines = f.readlines()
    if len(lines) >= 3:
        lines[2] = "Date,Close,High,Low,Open,Volume\n"
        with open(price_path, "w") as f:
            f.writelines(lines)
        print(f"Đã cập nhật dòng header thứ 3 trong {price_path}")
    else:
        print(f"File {price_path} không đủ dòng để sửa header.")

def download_earnings_data(ticker):
    ticker_dir = os.path.join(data_dir, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    t = yf.Ticker(ticker)
    earnings_df = t.earnings_dates.reset_index()
    earnings_df["Earnings Date"] = pd.to_datetime(earnings_df["Earnings Date"])
    earnings_df = earnings_df.set_index("Earnings Date")
    earnings_df = earnings_df.sort_index()
    earnings_df = earnings_df[start_date:end_date]
    earnings_path = os.path.join(ticker_dir, f"DATA_earnings.csv")
    earnings_df.to_csv(earnings_path)
    print(f"Đã lưu dữ liệu earnings vào file: {earnings_path}")

def generate_fake_sentiment(ticker):
    ticker_dir = os.path.join(data_dir, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    t = yf.Ticker(ticker)
    earnings_df = t.earnings_dates.reset_index()
    earnings_df["Earnings Date"] = pd.to_datetime(earnings_df["Earnings Date"]).dt.tz_localize(None)
    earnings_df = earnings_df[
        (earnings_df["Earnings Date"] >= pd.to_datetime(start_date)) &
        (earnings_df["Earnings Date"] <= pd.to_datetime(end_date))
    ]
    earnings_df['compound'] = np.random.uniform(-1, 1, size=len(earnings_df))
    sentiment_df = earnings_df[['Earnings Date', 'compound']]
    sentiment_path = os.path.join(ticker_dir, f"DATA_sentiment.csv")
    sentiment_df.to_csv(sentiment_path, index=False)
    print(f"Đã tạo file sentiment giả định: {sentiment_path}")

def main():
    for ticker in TICKERS:
        print(f"\n==== Tải dữ liệu cho {ticker} ====")
        download_price_data(ticker)
        download_earnings_data(ticker)
        generate_fake_sentiment(ticker)

if __name__ == "__main__":
    main() 