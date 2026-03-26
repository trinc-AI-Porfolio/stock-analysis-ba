import sys
import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

data_dir = "data"
TICKERS = ["META", "RDDT"]  

# ====== Mô tả dữ liệu cho từng file ======
def describe_file_structure(ticker):
    print(f"\n====== {ticker} ======")
    base_path = os.path.join(data_dir, ticker)
    filenames = ["DATA_price.csv", "DATA_earnings.csv", "DATA_sentiment.csv"]
    for fname in filenames:
        fpath = os.path.join(base_path, fname)
        print(f"\n--- File: {fname} ---")
        if not os.path.exists(fpath):
            print("File not found.")
            continue
        skip = 2 if "price" in fname else 0
        try:
            df = pd.read_csv(fpath, skiprows=skip)
            print(f"Số dòng: {df.shape[0]} | Số cột: {df.shape[1]}")
            print("\nCấu trúc dữ liệu:")
            print(df.dtypes)
            print("\nThiếu dữ liệu mỗi cột:")
            print(df.isnull().sum())
            print("\nMô tả thống kê:")
            print(df.describe(include='all'))
        except Exception as e:
            print(f"Lỗi đọc file: {e}")

# ====== Xử lý nâng cao cho từng mã ======
def advanced_preprocessing(ticker):
    print(f"\n--- Xử lý dữ liệu cho: {ticker} ---")
    base_path = os.path.join(data_dir, ticker)
    fpath = os.path.join(base_path, "DATA_price.csv")
    if not os.path.exists(fpath):
        print("Không tìm thấy file price.")
        return
    df = pd.read_csv(fpath, skiprows=2)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.set_index("Date", inplace=True)
    # 1. Missing value: interpolation + KNN imputation
    df.interpolate(method='linear', inplace=True)
    imputer = KNNImputer(n_neighbors=3)
    df[df.columns] = imputer.fit_transform(df)
    # 2. Outlier removal: Z-score + IQR (2 bước)
    z_thresh = 3
    z_scores = np.abs((df - df.mean()) / df.std())
    df = df[(z_scores < z_thresh).all(axis=1)]
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    # 3. Feature engineering
    if 'Close' in df.columns and 'High' in df.columns and 'Low' in df.columns:
        df['rolling_mean_5'] = df['Close'].rolling(window=5).mean()
        df['rolling_std_5'] = df['Close'].rolling(window=5).std()
        df['volatility'] = df['High'] - df['Low']
        df['price_change_pct'] = df['Close'].pct_change() * 100
        # Thêm các cột biến động đặc biệt
        df['biendong'] = (df['Close'] - df['Close'].shift(1)).abs()
        df['tangtoida'] = (df['High'] - df['Open']).clip(lower=0)
        df['giamtoida'] = (df['Open'] - df['Low']).clip(lower=0)
        # Thêm cột target: 1 nếu tăng mạnh, -1 nếu giảm mạnh, 0 nếu bình thường
        df['target'] = df['price_change_pct'].apply(lambda x: 1 if x > 2 else (-1 if x < -2 else 0))
    df.dropna(inplace=True)
    # 4. Normalize (chỉ scale các feature đầu vào, không scale target)
    features_to_scale = [col for col in df.columns if col != 'target']
    scaler = MinMaxScaler()
    scaled_df = df.copy()
    scaled_df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    # target giữ nguyên
    # Save lại
    out_path = os.path.join(base_path, f"{ticker}_processed.csv")
    scaled_df.to_csv(out_path)
    print(f"✅ Đã lưu: {out_path}")


# ====== Menu tổng hợp ======
def main():
    for tk in TICKERS:
        describe_file_structure(tk)
    for tk in TICKERS:
        advanced_preprocessing(tk)
    print("\nĐã hoàn thành mô tả và xử lý nâng cao cho tất cả các mã!")

if __name__ == "__main__":
    main() 