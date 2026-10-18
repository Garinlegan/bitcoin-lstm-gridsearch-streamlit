import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model


# KONFIGURASI
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATASET_PATH = BASE_DIR / "data" / "2021-2025aa.csv"
MODEL_DEFAULT_PATH = BASE_DIR / "model" / "default_2021-2025.h5"
MODEL_GRIDSEARCH_PATH = BASE_DIR / "model" / "gridsearch_2021-2025.h5"

TARGET_COLUMN = "Close"
WINDOW_SIZE = 30


# SETUP HALAMAN
st.set_page_config(
    page_title="Prediksi Harga Bitcoin - LSTM",
    layout="wide"
)

st.title("Prediksi Harga Bitcoin Menggunakan LSTM")
st.markdown("""
<style>
.block-container {
    padding-left: 5rem;
    padding-right: 5rem;
    padding-top: 2rem;
}
</style>
            
<p style='font-size:24px; text-align:justify;'>
Model ini menggunakan metode Long Short-Term Memory (LSTM) untuk memprediksi harga Bitcoin berdasarkan data historis.
LSTM adalah jenis Recurrent Neural Network (RNN) yang mampu menangkap pola jangka panjang dalam data time series.
Optimasi hyperparameter dilakukan menggunakan Grid Search untuk mendapatkan kombinasi parameter terbaik seperti 
jumlah neuron, batch size, dan learning rate sehingga performa model menjadi lebih optimal.
</p><br>
""", unsafe_allow_html=True)


# LOAD DATASET
@st.cache_data
def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    df = df.dropna(subset=["Date", TARGET_COLUMN])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# LOAD MODEL
@st.cache_resource
def load_models():
    model_default = load_model(MODEL_DEFAULT_PATH, compile=False)
    model_gridsearch = load_model(MODEL_GRIDSEARCH_PATH, compile=False)

    return model_default, model_gridsearch


# FUNGSI BANTUAN
def create_sequences(data, window_size):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)


def predict_future(model, last_data, scaler, n_days):
    predictions = []
    current_data = last_data.copy()

    for _ in range(n_days):
        input_data = current_data.reshape(1, WINDOW_SIZE, 1)
        pred = model.predict(input_data, verbose=0)[0][0]

        predictions.append(pred)
        current_data = np.append(current_data[1:], pred)

    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)

    return predictions.flatten()


def evaluate_model(model, X_test, y_test, scaler):
    pred_scaled = model.predict(X_test, verbose=0)

    y_actual = scaler.inverse_transform(y_test)
    y_pred = scaler.inverse_transform(pred_scaled)

    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)

    return y_actual.flatten(), y_pred.flatten(), mse, rmse, mae


# MAIN PROGRAM
df = load_dataset()
model_default, model_gridsearch = load_models()
# n_future = 1
st.sidebar.header("⚙️ Pengaturan")
n_future = st.sidebar.number_input(
    "Jumlah hari prediksi ke depan",
    min_value=1,
    max_value=30,
    value=7
)


# TAMPILKAN DATA HISTORIS
st.subheader("Data Historis")
st.dataframe(df.tail(10), use_container_width=True)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["Date"], df[TARGET_COLUMN])
ax.set_title("Harga Penutupan Bitcoin")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Harga Close (USD)")
st.pyplot(fig)


# PREPROCESSING
scaler = MinMaxScaler(feature_range=(0, 1))

close_data = df[[TARGET_COLUMN]].values
scaled_data = scaler.fit_transform(close_data)

if len(scaled_data) <= WINDOW_SIZE:
    st.error("Jumlah data terlalu sedikit untuk window size yang digunakan.")
    st.stop()

X, y = create_sequences(scaled_data, WINDOW_SIZE)

split_index = int(len(X) * 0.8)

X_test = X[split_index:]
y_test = y[split_index:]


# TAB STREAMLIT
tab1, tab2 = st.tabs(["Prediksi Harga", "Perbandingan Model"])


# TAB 1: PREDIKSI HARGA
with tab1:
    st.subheader("Prediksi Harga Bitcoin ke Depan")

    pilihan_model = st.radio(
        "Pilih model:",
        ["Model Default", "Model Grid Search"],
        horizontal=True
    )

    if pilihan_model == "Model Default":
        model = model_default
    else:
        model = model_gridsearch

    if st.button("Jalankan Prediksi"):
        last_data = scaled_data[-WINDOW_SIZE:].flatten()
        hasil_prediksi = predict_future(model, last_data, scaler, n_future)

        tanggal_terakhir = df["Date"].iloc[-1]
        tanggal_prediksi = pd.date_range(
            start=tanggal_terakhir + pd.Timedelta(days=1),
            periods=n_future,
            freq="D"
        )

        hasil_df = pd.DataFrame({
            "Tanggal": tanggal_prediksi,
            "Prediksi Harga (USD)": hasil_prediksi
        })

        st.dataframe(hasil_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["Date"].iloc[-30:], df[TARGET_COLUMN].iloc[-30:], label="Data Historis")
        ax.plot(tanggal_prediksi, hasil_prediksi, marker="o", linestyle="--", label="Prediksi")
        ax.set_title(f"Prediksi Harga Bitcoin - {pilihan_model}")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga (USD)")
        ax.legend()
        st.pyplot(fig)


# TAB 2: PERBANDINGAN MODEL
with tab2:
    st.subheader("Perbandingan Model Default dan Grid Search")

    if st.button("Jalankan Evaluasi Model"):
        y_actual, pred_default, mse_default, rmse_default, mae_default = evaluate_model(
            model_default, X_test, y_test, scaler
        )

        _, pred_gridsearch, mse_gridsearch, rmse_gridsearch, mae_gridsearch = evaluate_model(
            model_gridsearch, X_test, y_test, scaler
        )

        penurunan_rmse = ((rmse_default - rmse_gridsearch) / rmse_default) * 100

        col1, col2, col3 = st.columns(3)

        col1.metric("RMSE Default", f"{rmse_default:,.2f}")
        col2.metric("RMSE Grid Search", f"{rmse_gridsearch:,.2f}", delta=f"-{penurunan_rmse:.2f}%")
        col3.metric("Penurunan RMSE", f"{penurunan_rmse:.2f}%")
        
        col4, col5 = st.columns(2)
        col4.metric("MAE Model Default", f"{mae_default:,.2f}")
        col5.metric("MAE Model Grid Search", f"{mae_gridsearch:,.2f}")

        hasil_evaluasi = pd.DataFrame({
            "Model": ["Default", "Grid Search"],
            "RMSE": [rmse_default, rmse_gridsearch],
            "MAE": [mae_default, mae_gridsearch]
        })

        st.subheader("Tabel Evaluasi")
        st.dataframe(hasil_evaluasi, use_container_width=True)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(y_actual, label="Data Aktual")
        ax.plot(pred_default, linestyle="--", label="Prediksi Default")
        ax.plot(pred_gridsearch, linestyle="--", label="Prediksi Grid Search")
        ax.set_title("Perbandingan Prediksi dengan Data Aktual")
        ax.set_xlabel("Index Data Uji")
        ax.set_ylabel("Harga (USD)")
        ax.legend()
        st.pyplot(fig)