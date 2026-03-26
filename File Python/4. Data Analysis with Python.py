import pandas as pd

# Danh sách file processed và tên mã
files = [
    ("META", "data/META/META_processed.csv"),
    ("RDDT", "data/RDDT/RDDT_processed.csv")
]

for ticker, file_path in files:
    print(f"\n===== PHÂN TÍCH CHO {ticker} =====")
    df = pd.read_csv(file_path, index_col=0)

    # In 5 dòng đầu của data
    print("\n5 dòng đầu của dataset:")
    print(df.head())

    # Thông tin cơ bản về dataset
    print("\nThông tin cơ bản về dataset:")
    print(df.info())

    # Thống kê mô tả cơ bản
    print("\nThống kê mô tả cơ bản:")
    print(df.describe())

    # Kiểm tra missing values
    print("\nSố lượng missing values trong mỗi cột:")
    print(df.isnull().sum())

    # Nhóm dữ liệu theo cột 'target' và tính trung bình
    if 'target' in df.columns:
        grouped_by_target = df.groupby('target').mean(numeric_only=True)
        print("\nTrung bình các chỉ số theo mức độ biến động (target):")
        print(grouped_by_target)
    else:
        print("\nKhông tìm thấy cột 'target' để nhóm dữ liệu.")

    # Hàm tìm outliers sử dụng phương pháp IQR
    def find_outliers(column):
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return column[(column < lower_bound) | (column > upper_bound)]

    # Tìm outliers trong các cột quan trọng
    for col in ['biendong', 'tangtoida', 'giamtoida']:
        if col in df.columns:
            outliers = find_outliers(df[col])
            print(f"\nCác giá trị ngoại lệ trong cột '{col}':")
            print(outliers)
        else:
            print(f"\nKhông tìm thấy cột '{col}' trong dữ liệu.")

    # Tính ma trận tương quan
    correlation_matrix = df.corr(numeric_only=True)
    print("\nMa trận tương quan giữa các chỉ số:")
    print(correlation_matrix)

    # Lọc ra các tương quan mạnh với cột 'target'
    if 'target' in correlation_matrix.columns:
        target_correlations = correlation_matrix['target'].sort_values(ascending=False)
        print("\nTương quan giữa các chỉ số với cột 'target':")
        print(target_correlations)
    else:
        print("\nKhông tìm thấy cột 'target' trong ma trận tương quan.")

    # Phân bố biến động tối đa theo nhóm
    group_cols = ['biendong', 'tangtoida', 'giamtoida']
    for col in group_cols:
        if 'target' in df.columns and col in df.columns:
            distribution = df.groupby('target')[col].describe()
            print(f"\nPhân bố {col} theo nhóm:")
            print(distribution)
        else:
            print(f"\nKhông đủ cột để phân tích phân bố {col} theo nhóm.")
