# =========================
# IMPORT
# =========================
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Dashboard Distribusi Pupuk",
    page_icon="🌾",
    layout="wide"
)

# =========================
# TITLE
# =========================
st.title("🌾 Dashboard Analisis & Forecast Distribusi Pupuk")
st.markdown(
    "Dashboard ini menyajikan Exploratory Data Analysis (EDA), "
    "analisis tren historis, evaluasi model forecasting, "
    "serta status distribusi pupuk per kecamatan."
)

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("Rekap Penyaluran Pupuk.xlsx")
tahun = [2020, 2021, 2022, 2023, 2024, 2025]

KEBUTUHAN_PER_HA = 250
KONVERSI_TON_KE_KG = 1000

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    kecamatan = st.selectbox(
        "Pilih Kecamatan",
        df["Kecamatan"].unique()
    )
    st.markdown("---")
    st.caption("Sumber Data: Rekap Penyaluran Pupuk")

# =========================
# PREPARE DATA
# =========================
data = df[df["Kecamatan"] == kecamatan]
y = data[[f"Realisasi {t}" for t in tahun]].values.flatten()

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Tren & Forecast",
    "🔄 Evaluasi Model",
    "📊 Ranking Stabilitas",
    "📊 EDA Distribusi"
])

# ======================================================
# TAB 1 — TREN & FORECAST
# ======================================================
with tab1:
    st.subheader(f"📈 Tren & Forecast – {kecamatan}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=tahun,
        y=y,
        mode="lines+markers+text",
        text=[f"{v:.0f}" for v in y],
        textposition="top center",
        name="Data Historis"
    ))

    # Prophet
    df_long = pd.DataFrame({
        "ds": pd.to_datetime([f"{t}-01-01" for t in tahun]),
        "y": y
    })

    model_p = Prophet(yearly_seasonality=False)
    model_p.fit(df_long)

    future = model_p.make_future_dataframe(periods=3, freq="YS")
    forecast = model_p.predict(future)
    future_p = forecast[forecast["ds"].dt.year > 2025]

    fig.add_trace(go.Scatter(
        x=future_p["ds"].dt.year,
        y=future_p["yhat"],
        mode="lines+markers",
        line=dict(dash="dash"),
        name="Forecast Prophet"
    ))

    # Linear Regression
    X = np.array(tahun).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X, y)
    pred_lr = lr.predict(np.array([2026, 2027, 2028]).reshape(-1, 1))

    fig.add_trace(go.Scatter(
        x=[2026, 2027, 2028],
        y=pred_lr,
        mode="lines+markers",
        line=dict(dash="dot"),
        name="Forecast Linear Regression"
    ))

    fig.update_layout(
        xaxis_title="Tahun",
        yaxis_title="Realisasi (Ton)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Status Distribusi
    st.subheader("🟢 Status Distribusi Tahun 2025")

    luas_2025 = data["Luas Tanam 2024"].values[0]
    realisasi_2025 = data["Realisasi 2025"].values[0] * KONVERSI_TON_KE_KG
    kebutuhan_2025 = luas_2025 * KEBUTUHAN_PER_HA

    if realisasi_2025 > kebutuhan_2025 * 1.05:
        status = "🟢 Berlebih"
    elif realisasi_2025 < kebutuhan_2025 * 0.95:
        status = "🔴 Kurang"
    else:
        status = "🟡 Cukup"

    col1, col2, col3 = st.columns(3)
    col1.metric("Kebutuhan (kg)", f"{kebutuhan_2025:,.0f}")
    col2.metric("Realisasi (kg)", f"{realisasi_2025:,.0f}")
    col3.metric("Status", status)

# ======================================================
# TAB 2 — EVALUASI MODEL
# ======================================================
with tab2:
    st.subheader(f"🔄 Evaluasi Akurasi Model – {kecamatan}")

    y_true = y

    # Prophet
    model_p = Prophet(yearly_seasonality=False)
    model_p.fit(df_long)
    pred_p = model_p.predict(df_long)["yhat"]

    mape_p = np.mean(np.abs((y_true - pred_p) / y_true)) * 100
    akurasi_p = 100 - mape_p

    # Linear Regression
    lr.fit(X, y_true)
    pred_lr_hist = lr.predict(X)
    mape_lr = np.mean(np.abs((y_true - pred_lr_hist) / y_true)) * 100
    akurasi_lr = 100 - mape_lr

    col1, col2, col3 = st.columns(3)
    col1.metric("Akurasi Prophet", f"{akurasi_p:.2f} %")
    col2.metric("Akurasi Linear Reg.", f"{akurasi_lr:.2f} %")
    col3.metric("Model Terbaik", "Prophet" if mape_p < mape_lr else "Linear Regression")

# ======================================================
# TAB 3 — RANKING STABILITAS
# ======================================================
with tab3:
    st.subheader("📊 Top 5 Kecamatan Paling Stabil")

    hasil = []

    for kec in df["Kecamatan"].unique():
        d = df[df["Kecamatan"] == kec]
        y_k = d[[f"Realisasi {t}" for t in tahun]].values.flatten()

        df_k = pd.DataFrame({
            "ds": pd.to_datetime([f"{t}-01-01" for t in tahun]),
            "y": y_k
        })

        model = Prophet(yearly_seasonality=False)
        model.fit(df_k)
        pred = model.predict(df_k)["yhat"]

        mape = np.mean(np.abs((y_k - pred) / y_k)) * 100

        hasil.append({
            "Kecamatan": kec,
            "MAPE (%)": round(mape, 2),
            "Akurasi (%)": round(100 - mape, 2)
        })

    df_rank = pd.DataFrame(hasil).sort_values("MAPE (%)").head(5)
    st.table(df_rank)

# ======================================================
# TAB 4 — EDA DISTRIBUSI (LENGKAP)
# ======================================================
with tab4:
    st.subheader(f"📊 Exploratory Data Analysis – {kecamatan}")

    total = y.sum()
    rata2 = y.mean()
    std = y.std()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Realisasi", f"{total:,.0f}")
    col2.metric("Rata-rata Tahunan", f"{rata2:,.0f}")
    col3.metric("Fluktuasi (Std Dev)", f"{std:.2f}")

    st.markdown("---")

    # Shock Analysis
    delta = np.diff(y)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{tahun[i]}–{tahun[i+1]}" for i in range(len(delta))],
        y=delta
    ))

    fig.update_layout(
        title="Perubahan Ekstrem Antar Tahun",
        xaxis_title="Periode",
        yaxis_title="Δ Realisasi (Ton)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "📝 Analisis perubahan ekstrem dilakukan untuk mengidentifikasi "
        "kenaikan atau penurunan realisasi pupuk yang signifikan antar tahun."
    )

    st.markdown("---")

    # Kontribusi Kecamatan
    total_all = df[[f"Realisasi {t}" for t in tahun]].sum().sum()
    kontribusi = (total / total_all) * 100

    st.metric("Kontribusi terhadap Total Kabupaten (%)", f"{kontribusi:.2f} %")

    st.markdown(
        "📝  Analisis kontribusi dilakukan untuk mengetahui peran relatif "
        "setiap kecamatan terhadap total realisasi tahunan."
    )

    st.markdown("---")

    # Karakter Distribusi
    rata_std = df[[f"Realisasi {t}" for t in tahun]].std(axis=1).mean()

    karakter = "Fluktuatif" if std > rata_std else "Relatif Stabil"

    st.success(f"📌 Karakter Distribusi Kecamatan: **{karakter}**")

    st.markdown("---")
    st.subheader("📉 Analisis Stabilitas Distribusi (Coefficient of Variation)")

    # =========================
    # COEFFICIENT OF VARIATION
    # =========================
    cv = std / rata2

    col1, col2 = st.columns(2)
    col1.metric("Std Dev", f"{std:.2f}")
    col2.metric("Coefficient of Variation (CV)", f"{cv:.2f}")

    # =========================
    # INTERPRETASI & PENANGANAN
    # =========================
    if cv > 0.4:
        karakter_cv = "Tidak Stabil"
        st.error("📌 Karakter Distribusi: **Tidak Stabil**")
        st.write(
            "➡️ **Penanganan:** "
            "Perlu pengendalian distribusi yang lebih ketat, evaluasi kesesuaian "
            "antara kebutuhan dan realisasi, serta monitoring berkala setiap musim tanam."
        )

        st.markdown(
            "📝 **Narasi Analisis:** "
            "**Standard Deviation (Std Dev)** dihitung dari variasi realisasi pupuk "
            "antar tahun selama periode 2020–2025, yang mencerminkan besarnya fluktuasi "
            "distribusi pupuk di kecamatan ini. "
            "Sementara itu, **Coefficient of Variation (CV)** merupakan perbandingan "
            "antara Std Dev dengan rata-rata realisasi tahunan, sehingga menunjukkan "
            "tingkat fluktuasi relatif terhadap skala distribusinya. "
            "Nilai CV yang melebihi **0,4** menandakan bahwa variasi distribusi "
            "sangat besar dibandingkan rata-ratanya, sehingga distribusi pupuk "
            "dikategorikan **tidak stabil** dan memerlukan perhatian serta pengendalian khusus."
        )

    elif cv > 0.2:
        karakter_cv = "Cukup Stabil"
        st.warning("📌 Karakter Distribusi: **Cukup Stabil**")
        st.write(
            "➡️ **Penanganan:** "
            "Optimalkan perencanaan tahunan, perbaiki sinkronisasi data kebutuhan "
            "dengan realisasi, serta lakukan evaluasi di tahun dengan perubahan signifikan."
        )

        st.markdown(
            "📝 **Analisis:** "
            "Perhitungan **Std Dev** dan **CV** didasarkan pada data realisasi pupuk "
            "tahunan periode 2020–2025. "
            "Std Dev menunjukkan adanya fluktuasi distribusi antar tahun, "
            "sedangkan nilai CV berada pada rentang **0,2 hingga 0,4**, "
            "yang mengindikasikan bahwa variasi distribusi relatif terhadap "
            "rata-rata masih berada dalam batas wajar. "
            "Oleh karena itu, distribusi pupuk dikategorikan **cukup stabil**, "
            "namun tetap memerlukan penguatan dalam perencanaan dan pengendalian distribusi."
        )

    else:
        karakter_cv = "Stabil"
        st.success("📌 Karakter Distribusi: **Stabil**")
        st.write(
            "➡️ **Penanganan:** "
            "Pola distribusi dapat dijadikan acuan perencanaan pada periode berikutnya "
            "serta sebagai referensi bagi kecamatan lain."
        )

        st.markdown(
            "📝 **Analisis:** "
            "**Standard Deviation (Std Dev)** yang rendah menunjukkan bahwa "
            "fluktuasi realisasi pupuk antar tahun relatif kecil. "
            "Nilai **Coefficient of Variation (CV)** yang berada di bawah **0,2** "
            "menandakan bahwa variasi distribusi sangat kecil dibandingkan "
            "rata-rata realisasi tahunan. "
            "Dengan demikian, distribusi pupuk di kecamatan ini dikategorikan **stabil**, "
            "karena pola realisasi cenderung konsisten dan dapat dijadikan acuan "
            "dalam perencanaan distribusi di periode berikutnya."
        )

    # =========================
    # RINGKASAN EDA
    # =========================
    st.markdown("---")
    st.subheader("🎯 Ringkasan EDA Kecamatan")

    st.markdown(
        f"""
        - Total realisasi pupuk selama periode analisis sebesar **{total:,.0f} ton**
        - Rata-rata distribusi tahunan sebesar **{rata2:,.0f} ton**
        - Kecamatan memiliki karakter distribusi **{karakter}** secara umum
        - Berdasarkan Coefficient of Variation (CV), tingkat stabilitas distribusi termasuk **{karakter_cv}**
        """
    )
