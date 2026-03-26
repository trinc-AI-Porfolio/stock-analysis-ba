
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

TICKERS = ["META", "RDDT"]
DATA_FOLDER = "data"

for ticker in TICKERS:
    print(f"\n==== Regression dự báo price_change_pct cho {ticker} ====")
    fpath = os.path.join(DATA_FOLDER, ticker, f"{ticker}_processed.csv")
    df = pd.read_csv(fpath, parse_dates=['Date'], index_col=0)
    # Chọn feature tốt nhất
    features = []
    for col in ['rolling_mean_5', 'rolling_std_5', 'volatility', 'Volume']:
        if col in df.columns:
            features.append(col)
    if 'price_change_pct' not in df.columns or len(features) == 0:
        print(f"Không đủ feature cho {ticker}, bỏ qua.")
        continue
    X = df[features]
    y = df['price_change_pct']
    # Chia train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Chuẩn hóa
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Các mô hình
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'SVR': SVR(),
        'KNN': KNeighborsRegressor()
    }
    results = {}
    for name, model in models.items():
        if name in ['SVR', 'KNN']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results[name] = rmse
        print(f"{name}: RMSE = {rmse:.3f}")
    best_model = min(results, key=results.get)
    print(f"\n==> Mô hình tốt nhất cho {ticker}: {best_model} (RMSE = {results[best_model]:.3f})")

    # Lấy độ quan trọng của các đặc trưng (feature importance)
    print(f"-- Độ quan trọng của các đặc trưng cho {ticker} --")
    importance = None
    model_for_importance = None
    if 'Random Forest' in models:
        model_for_importance = models['Random Forest']
        model_for_importance.fit(X, y)
        importance = model_for_importance.feature_importances_
    elif hasattr(model, 'feature_importances_'):
        model_for_importance = model
        importance = model.feature_importances_
    if importance is not None:
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        print(importance_df)
        # Vẽ biểu đồ độ quan trọng với seaborn, palette viridis
        import matplotlib.pyplot as plt
        import seaborn as sns
        chart_dir = os.path.join('chart', '6_feature_importance', ticker)
        os.makedirs(chart_dir, exist_ok=True)
        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=importance_df,
            x="Importance",
            y="Feature",
            hue="Feature",
            palette="viridis",
            legend=False
        )
        plt.title(f"Độ quan trọng của các đặc trưng trong mô hình Random Forest - {ticker}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, 'feature_importance.png'))
        plt.show()
        plt.close()
    else:
        print("Không thể tính độ quan trọng đặc trưng với mô hình này.")

# ================= DỰ BÁO BIẾN ĐỘNG TRUNG BÌNH THÁNG TỚI (NHIỀU FEATURE) ================
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

TICKERS = ["META", "RDDT"]
DATA_FOLDER = "data"

print("\n==== DỰ BÁO BIẾN ĐỘNG TRUNG BÌNH THÁNG TỚI (NHIỀU FEATURE) ====")
for ticker in TICKERS:
    print(f"\n--- {ticker} ---")
    processed_path = os.path.join(DATA_FOLDER, ticker, f"{ticker}_processed.csv")
    if not os.path.exists(processed_path):
        print(f"Không tìm thấy file {processed_path}, bỏ qua.")
        continue
    df = pd.read_csv(processed_path)
    if 'Date' not in df.columns:
        print(f"Không có cột Date trong {processed_path}, bỏ qua.")
        continue
    df['Date'] = pd.to_datetime(df['Date'])
    df['month'] = df['Date'].dt.to_period('M')
    features = ['biendong', 'volatility', 'Volume', 'rolling_mean_5', 'rolling_std_5', 'price_change_pct']
    features = [f for f in features if f in df.columns]
    for extra in ['sentiment', 'earning']:
        if extra in df.columns:
            features.append(extra)
    df_month = df.groupby('month')[features].mean().reset_index()
    if len(df_month) < 13:
        print("Không đủ dữ liệu để dự báo 12 tháng.")
        continue
    X_list = []
    y_list = []
    for i in range(len(df_month) - 12):
        row_feats = []
        for f in features:
            row_feats.extend(df_month[f].iloc[i:i+12].values)
        X_list.append(row_feats)
        y_list.append(df_month['biendong'].iloc[i+12])
    X = pd.DataFrame(X_list)
    y = pd.Series(y_list)
    # Chọn mô hình theo ticker
    if ticker == 'META':
        model = KNeighborsRegressor()
    else:
        model = LinearRegression()
    model.fit(X, y)
    last_feats = []
    for f in features:
        last_feats.extend(df_month[f].iloc[-12:].values)
    last_feats_df = pd.DataFrame([last_feats], columns=X.columns)
    pred_next_month = model.predict(last_feats_df)[0]
    print("Biến động trung bình 12 tháng gần nhất:")
    print(df_month[['month', 'biendong']].tail(12).round(4).to_string(index=False))
    print(f"\nDự báo biến động trung bình tháng tới: {pred_next_month:.2f}")
    last_month = df_month['biendong'].iloc[-1]
    pct_change = 100 * (pred_next_month - last_month) / last_month if last_month != 0 else 0
    print(f"Dự báo tháng tới sẽ {'TĂNG' if pct_change > 0 else 'GIẢM'} {abs(pct_change):.2f}% so với tháng này.")

