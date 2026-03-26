import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analysis Dashboard — META & RDDT",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f17; }
    .metric-card {
        background: linear-gradient(135deg, #1c1c2e, #16213e);
        border: 1px solid rgba(139,92,246,0.25);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .stMetric { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(ticker):
    path = f"data/{ticker}/{ticker}_processed.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data
def load_price(ticker):
    path = f"data/{ticker}/DATA_price.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")
    return df

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-market.png", width=64)
    st.title("📊 Stock Analysis")
    st.caption("Final Project BA — Nhóm 5")
    st.divider()
    
    ticker = st.radio("Chọn cổ phiếu:", ["META", "RDDT"], horizontal=True)
    st.divider()
    
    page = st.selectbox("📋 Trang phân tích", [
        "🏠 Tổng Quan",
        "📈 Xu Hướng Giá",
        "🔥 Phân Tích Biến Động",
        "🧠 Mô Hình Dự Báo",
        "⚖️ So Sánh META vs RDDT",
    ])

# ── Load ───────────────────────────────────────────────────────────────────
try:
    df = load_data(ticker)
    price_df = load_price(ticker)
except Exception as e:
    st.error(f"Không thể tải dữ liệu: {e}")
    st.stop()

COLOR = "#8b5cf6" if ticker == "META" else "#06b6d4"

# ═══════════════════════════════════════════════════════
# PAGE 1: TỔNG QUAN
# ═══════════════════════════════════════════════════════
if page == "🏠 Tổng Quan":
    st.title(f"🏠 Tổng Quan — {ticker}")
    st.caption("Thống kê mô tả dữ liệu lịch sử giá cổ phiếu")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    latest = price_df.iloc[-1]
    prev   = price_df.iloc[-2]
    close_col = "Close" if "Close" in price_df.columns else price_df.columns[-1]

    with col1:
        close_now  = float(latest[close_col]) if close_col in latest else 0
        close_prev = float(prev[close_col])   if close_col in prev   else 0
        delta = close_now - close_prev
        st.metric("💰 Giá Đóng Cửa (Mới Nhất)", f"${close_now:.2f}", f"{delta:+.2f}")
    with col2:
        if "biendong" in df.columns:
            avg_vol = df["biendong"].mean()
            st.metric("📊 Biến Động TB", f"{avg_vol:.2f}")
    with col3:
        if "target" in df.columns:
            pct_up = (df["target"] == 1).mean() * 100
            st.metric("📈 % Phiên Tăng Mạnh", f"{pct_up:.1f}%")
    with col4:
        st.metric("📋 Số Phiên GD", f"{len(df):,}")

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📋 Thống Kê Mô Tả")
        cols_show = [c for c in ["biendong","tangtoida","giamtoida","price_change_pct","volatility"] if c in df.columns]
        st.dataframe(df[cols_show].describe().round(4), use_container_width=True)

    with col_right:
        st.subheader("🔢 Phân Bố Nhóm Biến Động")
        if "target" in df.columns:
            label_map = {-1: "Giảm Mạnh (-1)", 0: "Bình Thường (0)", 1: "Tăng Mạnh (1)"}
            vc = df["target"].value_counts().reset_index()
            vc.columns = ["target", "count"]
            vc["label"] = vc["target"].map(label_map)
            fig = px.pie(vc, values="count", names="label",
                         color_discrete_sequence=["#ef4444","#8b5cf6","#10b981"],
                         hole=0.4)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="white", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    st.subheader("🌡️ Ma Trận Tương Quan")
    corr_cols = [c for c in ["biendong","tangtoida","giamtoida","price_change_pct","volatility","target"] if c in df.columns]
    if corr_cols:
        corr = df[corr_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        aspect="auto", zmin=-1, zmax=1)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# PAGE 2: XU HƯỚNG GIÁ  
# ═══════════════════════════════════════════════════════
elif page == "📈 Xu Hướng Giá":
    st.title(f"📈 Xu Hướng Giá — {ticker}")

    close_col = "Close" if "Close" in price_df.columns else price_df.columns[-1]
    n_days = st.slider("Số ngày hiển thị:", 30, len(price_df), min(252, len(price_df)))
    df_plot = price_df.tail(n_days)

    # Candlestick if OHLC available
    if all(c in price_df.columns for c in ["Open","High","Low","Close"]):
        fig = go.Figure(data=[go.Candlestick(
            x=df_plot["Date"], open=df_plot["Open"],
            high=df_plot["High"], low=df_plot["Low"], close=df_plot["Close"],
            increasing_line_color="#10b981", decreasing_line_color="#ef4444",
        )])
        fig.update_layout(title=f"Candlestick Chart — {ticker}",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,23,0.8)",
                          font_color="white", xaxis_rangeslider_visible=False,
                          height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.line(df_plot, x="Date", y=close_col, title=f"Giá Đóng Cửa {ticker}")
        fig.update_traces(line_color=COLOR)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,23,0.8)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    # Volume
    if "Volume" in price_df.columns:
        fig2 = px.bar(df_plot, x="Date", y="Volume", title="Khối Lượng Giao Dịch",
                      color_discrete_sequence=[COLOR])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,23,0.8)", font_color="white")
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════
# PAGE 3: PHÂN TÍCH BIẾN ĐỘNG
# ═══════════════════════════════════════════════════════
elif page == "🔥 Phân Tích Biến Động":
    st.title(f"🔥 Phân Tích Biến Động — {ticker}")

    n_days = st.slider("Số ngày:", 10, min(len(df), 200), 60)
    df_plot = df.tail(n_days).copy()
    df_plot.index = pd.to_datetime(df_plot.index)

    col1, col2 = st.columns(2)

    with col1:
        # Line: biendong, tangtoida, giamtoida
        cols_v = [c for c in ["biendong","tangtoida","giamtoida"] if c in df_plot.columns]
        if cols_v:
            fig = go.Figure()
            colors_line = [COLOR, "#f59e0b", "#ef4444"]
            for i, col in enumerate(cols_v):
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col], name=col,
                                         line=dict(color=colors_line[i], width=2)))
            fig.update_layout(title="Xu Hướng Biến Động Theo Thời Gian",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,23,0.8)",
                              font_color="white", legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Boxplot biendong by target
        if "biendong" in df.columns and "target" in df.columns:
            label_map = {-1: "Giảm Mạnh", 0: "Bình Thường", 1: "Tăng Mạnh"}
            df_box = df[["biendong","target"]].copy()
            df_box["Nhóm"] = df_box["target"].map(label_map)
            fig = px.box(df_box, x="Nhóm", y="biendong", color="Nhóm",
                         color_discrete_map={"Giảm Mạnh":"#ef4444","Bình Thường":"#8b5cf6","Tăng Mạnh":"#10b981"},
                         title="Phân Bố Biến Động theo Nhóm Target")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,23,0.8)",
                              font_color="white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Rolling std
    if "price_change_pct" in df_plot.columns:
        df_plot["rolling_vol"] = df_plot["price_change_pct"].rolling(10).std()
        fig = px.area(df_plot, x=df_plot.index, y="rolling_vol",
                      title="Volatility Rolling 10 ngày (Std of price_change_pct)",
                      color_discrete_sequence=[COLOR])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,23,0.8)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# PAGE 4: MÔ HÌNH DỰ BÁO
# ═══════════════════════════════════════════════════════
elif page == "🧠 Mô Hình Dự Báo":
    st.title(f"🧠 Mô Hình Hồi Quy — {ticker}")
    st.caption("So sánh 5 mô hình dự báo `price_change_pct`")

    features = [c for c in ["rolling_mean_5","rolling_std_5","volatility","Volume"] if c in df.columns]
    if "price_change_pct" not in df.columns or len(features) == 0:
        st.warning("Không đủ features để chạy mô hình.")
        st.stop()

    df_model = df[features + ["price_change_pct"]].dropna()
    X = df_model[features]
    y = df_model["price_change_pct"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=50),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
    }

    with st.spinner("🔄 Đang train mô hình..."):
        results = {}
        preds   = {}
        for name, model in models.items():
            if name in ["SVR","KNN"]:
                model.fit(X_train_s, y_train); pred = model.predict(X_test_s)
            else:
                model.fit(X_train, y_train);   pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            results[name] = rmse
            preds[name]   = pred

    best = min(results, key=results.get)
    st.success(f"🏆 Mô hình tốt nhất: **{best}** (RMSE = {results[best]:.4f})")

    col1, col2 = st.columns(2)
    with col1:
        rmse_df = pd.DataFrame(list(results.items()), columns=["Model","RMSE"]).sort_values("RMSE")
        fig = px.bar(rmse_df, x="Model", y="RMSE", text="RMSE",
                     color="RMSE", color_continuous_scale="Viridis_r",
                     title="RMSE các Mô Hình")
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,23,0.8)",
                          font_color="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Feature importance from Random Forest
        rf = models["Random Forest"]
        rf.fit(X, y)
        imp_df = pd.DataFrame({"Feature": features, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=True)
        fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Plasma",
                     title="Feature Importance (Random Forest)")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,23,0.8)",
                          font_color="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Actual vs Predicted
    st.subheader(f"Actual vs Predicted — {best}")
    pred_best = preds[best]
    fig = go.Figure()
    idx = range(min(100, len(y_test)))
    fig.add_trace(go.Scatter(x=list(idx), y=list(y_test.values[:100]), name="Actual", line=dict(color="#10b981")))
    fig.add_trace(go.Scatter(x=list(idx), y=list(pred_best[:100]), name="Predicted", line=dict(color=COLOR, dash="dash")))
    fig.update_layout(title=f"Dự báo vs Thực tế ({best}) — 100 điểm cuối",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,23,0.8)", font_color="white")
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# PAGE 5: SO SÁNH META vs RDDT
# ═══════════════════════════════════════════════════════
elif page == "⚖️ So Sánh META vs RDDT":
    st.title("⚖️ So Sánh META vs RDDT")

    try:
        df_meta = load_data("META").reset_index().rename(columns={"index":"Date"})
        df_rddt = load_data("RDDT").reset_index().rename(columns={"index":"Date"})
        df_meta["Date"] = pd.to_datetime(df_meta["Date"])
        df_rddt["Date"] = pd.to_datetime(df_rddt["Date"])
        merged = pd.merge(df_meta, df_rddt, on="Date", suffixes=("_META","_RDDT"))
    except Exception as e:
        st.error(f"Không thể merge dữ liệu: {e}")
        st.stop()

    n_days = st.slider("Số ngày hiển thị:", 10, min(len(merged), 200), 60)
    m = merged.tail(n_days)

    col1, col2 = st.columns(2)
    with col1:
        if "biendong_META" in m.columns and "biendong_RDDT" in m.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=m["Date"], y=m["biendong_META"], name="META", line=dict(color="#8b5cf6", width=2)))
            fig.add_trace(go.Scatter(x=m["Date"], y=m["biendong_RDDT"], name="RDDT", line=dict(color="#06b6d4", width=2)))
            fig.update_layout(title="So Sánh Biến Động META vs RDDT",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,23,0.8)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "biendong_META" in m.columns and "biendong_RDDT" in m.columns:
            fig = px.scatter(m, x="biendong_META", y="biendong_RDDT",
                             trendline="ols", title="Scatter: META vs RDDT Biến Động",
                             color_discrete_sequence=["#f59e0b"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,23,0.8)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)

    # Rolling correlation
    if "biendong_META" in m.columns and "biendong_RDDT" in m.columns:
        window = st.slider("Rolling window (ngày):", 3, 20, 5)
        m = m.copy()
        m["rolling_corr"] = m["biendong_META"].rolling(window).corr(m["biendong_RDDT"])
        fig = px.line(m, x="Date", y="rolling_corr",
                      title=f"Rolling Correlation META vs RDDT (window={window}d)",
                      color_discrete_sequence=["#10b981"])
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,23,0.8)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    # Target comparison
    if "target_META" in merged.columns and "target_RDDT" in merged.columns:
        col1, col2 = st.columns(2)
        for col_key, col_name in [("col1","META"), ("col2","RDDT")]:
            with eval(col_key):
                lm = {-1:"Giảm Mạnh",0:"Bình Thường",1:"Tăng Mạnh"}
                vc = merged[f"target_{col_name}"].map(lm).value_counts().reset_index()
                vc.columns = ["Nhóm","count"]
                fig = px.pie(vc, values="count", names="Nhóm",
                             title=f"Phân Bố Target — {col_name}",
                             color_discrete_sequence=["#ef4444","#8b5cf6","#10b981"],
                             hole=0.4)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("📊 Stock Analysis Dashboard | BA Final Project — Nhóm 5 | Data: META & RDDT")
