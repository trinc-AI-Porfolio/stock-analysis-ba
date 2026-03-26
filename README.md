# Phân Tích Cổ Phiếu — BA Final Project

Phân tích và dự báo biến động cổ phiếu **META** (Meta Platforms) và **RDDT** (Reddit) sử dụng Python, SQL và Power BI.

## Tổng quan

Dự án BA cuối kỳ — Nhóm 5. Phân tích toàn diện dữ liệu lịch sử giá cổ phiếu, xây dựng mô hình dự báo và trực quan hóa kết quả.

## Nội dung

| Bước | Mô tả |
|---|---|
| `2.1 / 2.2` | Thu thập & làm sạch dữ liệu |
| `3.1 / 3.2` | Phân tích với SQL |
| `4` | Phân tích với Python (pandas, numpy) |
| `5` | Trực quan hóa (seaborn, matplotlib) |
| `6` | Hồi quy & dự báo (Random Forest, Linear, SVR, KNN, Decision Tree, VAR) |

## Công nghệ sử dụng

- **Python**: pandas, numpy, scikit-learn, statsmodels, seaborn, matplotlib
- **SQL**: SQLite
- **Power BI**: Dashboard tương tác
- **Models**: Linear Regression, Random Forest, Decision Tree, SVR, KNN, VAR

## Kết quả

- Phân tích biến động (`biendong`), tăng mạnh (`tangtoida`), giảm mạnh (`giamtoida`) theo thời gian
- Ma trận tương quan giữa các chỉ số
- So sánh biến động META vs RDDT (10 ngày gần nhất)
- Dự báo biến động tháng tới bằng VAR + KNN/Linear Regression

## Cấu trúc thư mục

```
├── File Python/     # Script Python cho từng bước
├── File ipynb/      # Jupyter Notebook
├── data/            # Dữ liệu gốc (META, RDDT)
├── chart/           # Biểu đồ kết quả
├── power pi/        # Power BI file (.pbix)
└── *.docx / *.pdf   # Báo cáo nhóm
```
