import pandas as pd
import sqlite3
import os

TICKERS = ["META", "RDDT"]
DATA_FOLDER = "data"

for ticker in TICKERS:
    print(f"\n==== Tạo database cho {ticker} ====")
    price_path = os.path.join(DATA_FOLDER, ticker, "DATA_price.csv")
    db_name = f"{ticker.lower()}_analysis.db"

    if not os.path.exists(price_path):
        print(f"Không tìm thấy file {price_path}, bỏ qua.")
        continue
    df_price = pd.read_csv(price_path, skiprows=1)
    # Lấy dòng đầu tiên làm header thực sự
    df_price.columns = df_price.iloc[0]
    df_price = df_price[1:]
    df_price = df_price.reset_index(drop=True)
    # Đổi tên cột ngày về 'Date' nếu cần
    if 'Date' not in df_price.columns:
        for col in df_price.columns:
            if 'date' in col.lower():
                df_price = df_price.rename(columns={col: 'Date'})
                break

    conn = sqlite3.connect(db_name)
    df_price.to_sql('price', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Đã tạo/làm mới bảng price trong {db_name}")
    print(df_price.head()) 