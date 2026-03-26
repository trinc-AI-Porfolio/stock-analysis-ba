import sqlite3
import os
import pandas as pd

# Mục tiêu bước này:
# Sử dụng SQL để phân tích mối quan hệ giữa các chỉ số chứng khoán của 2 mã (META, RDDT),
# tập trung vào độ biến động nhằm phục vụ phân tích xu hướng tăng giảm.
# Thực hiện các truy vấn dữ liệu để trả lời các câu hỏi cụ thể liên quan đến mức độ biến động.

TICKERS = ["META", "RDDT"]
data_folder = "data"

for ticker in TICKERS:
    print(f"\n==== Phân tích cho {ticker} ====")
    db_name = f'{ticker.lower()}_analysis.db'
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 1. Tính độ biến động giá trung bình (volatility = High - Low)
    print("\n[1] Độ biến động giá trung bình từng quý:")
    query_vol = '''
        SELECT 
            strftime('%Y-%m', Date) AS month,
            AVG(High - Low) AS avg_volatility
        FROM price
        GROUP BY month
        ORDER BY month
    '''
    for row in cursor.execute(query_vol):
        print(f"Tháng {row[0]}: Độ biến động TB = {row[1]:.2f}")

    # 2. Ngày có biến động mạnh nhất
    print("\n[2] Ngày có biến động mạnh nhất:")
    query_max = '''
        SELECT Date, (High - Low) AS volatility
        FROM price
        ORDER BY volatility DESC
        LIMIT 1
    '''
    row = cursor.execute(query_max).fetchone()
    print(f"Ngày {row[0]}: Độ biến động = {row[1]:.2f}")

    # 3. Số ngày có biến động trên 5% so với giá đóng cửa hôm trước
    print("\n[3] Số ngày biến động trên 5% so với giá đóng cửa hôm trước:")
    query_pct = '''
        SELECT COUNT(*) FROM (
            SELECT Date, 
                (Close - LAG(Close) OVER (ORDER BY Date)) * 100.0 / LAG(Close) OVER (ORDER BY Date) AS pct_change
            FROM price
        )
        WHERE ABS(pct_change) > 5
    '''
    row = cursor.execute(query_pct).fetchone()
    print(f"Số ngày biến động > 5%: {row[0]}")

    conn.close()

# 4. So sánh độ biến động trung bình giữa hai mã
print("\n==== So sánh độ biến động trung bình giữa hai mã ====")
results = []
for ticker in TICKERS:
    db_name = f'{ticker.lower()}_analysis.db'
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    query = 'SELECT AVG(High - Low) FROM price'
    avg_vol = cursor.execute(query).fetchone()[0]
    results.append((ticker, avg_vol))
    conn.close()
for ticker, avg_vol in results:
    print(f"{ticker}: Độ biến động TB = {avg_vol:.2f}")
if results[0][1] > results[1][1]:
    print(f"{results[0][0]} có độ biến động trung bình cao hơn {results[1][0]}")
else:
    print(f"{results[1][0]} có độ biến động trung bình cao hơn {results[0][0]}")
