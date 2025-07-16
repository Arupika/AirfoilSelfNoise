import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Untuk memuat model jika Anda menyimpannya

# --- Konfigurasi Halaman Dashboard ---
st.set_page_config(
    page_title="Dashboard Airfoil Self-Noise",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Bagian Judul dan Deskripsi ---
st.title("Dashboard Data dan Prediksi Airfoil Self-Noise")
st.markdown("Selamat datang di dashboard interaktif untuk menganalisis data kebisingan airfoil dan melihat prediksinya.")
st.markdown("---")

# --- Bagian 1: Memuat Data ---
try:
    df = pd.read_csv('AirfoilSelfNoise.csv')
    st.success("Data AirfoilSelfNoise berhasil dimuat!")

    if st.checkbox('Tampilkan Cuplikan Data Mentah', False):
        st.subheader("Cuplikan Data")
        st.dataframe(df.head())

except FileNotFoundError:
    st.error("Error: File 'AirfoilSelfNoise.csv' tidak ditemukan.")
    st.info("Pastikan file CSV berada di direktori yang sama dengan script dashboard_airfoil.py.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat data: {e}")
    st.stop()

st.markdown("---")

# --- Bagian 2: Analisis Data Eksplorasi Interaktif ---
st.header("Analisis Data Eksplorasi Interaktif")

# Menggunakan st.radio untuk memilih jenis tampilan
# Pilihan ditampilkan di sidebar untuk efisiensi ruang
analysis_option = st.radio(
    "Pilih Tampilan Analisis Data:",
    ("Statistik Deskriptif", "Distribusi Frekuensi (f)", "Scatter Plot (alpha vs SSPL)"),
    key="analysis_selector" # Kunci unik untuk widget ini
)

if analysis_option == "Statistik Deskriptif":
    st.subheader("Statistik Deskriptif Seluruh Data")
    st.write(df.describe())

elif analysis_option == "Distribusi Frekuensi (f)":
    st.subheader("Distribusi Frekuensi (f)")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    sns.histplot(df['f'], bins=30, kde=True, ax=ax_hist)
    ax_hist.set_title('Distribusi Frekuensi (f)')
    ax_hist.set_xlabel('Frekuensi (Hz)')
    ax_hist.set_ylabel('Jumlah Data')
    st.pyplot(fig_hist)

elif analysis_option == "Scatter Plot (alpha vs SSPL)":
    st.subheader("Hubungan Sudut Serang (alpha) dengan Sound Pressure Level (SSPL)")
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='alpha', y='SSPL', data=df, ax=ax_scatter, hue='U_infinity', size='c', sizes=(20, 200), alpha=0.7)
    ax_scatter.set_title('Scatter Plot: Sudut Serang vs. SPL')
    ax_scatter.set_xlabel('Sudut Serang (alpha, derajat)')
    ax_scatter.set_ylabel('Sound Pressure Level (SSPL, dB)')
    st.pyplot(fig_scatter)

st.markdown("---")

# --- Bagian 3: Prediksi Tingkat Kebisingan (SSPL) ---
st.header("Prediksi Tingkat Kebisingan (SSPL)")

model_path = 'linear_regression_model.pkl'

if st.button("Aktifkan Prediksi (Muat Model)"):
    try:
        model = joblib.load(model_path)
        st.success(f"Model prediksi berhasil dimuat dari '{model_path}'!")
        st.write("Gunakan slider di sidebar kiri untuk menginput nilai fitur dan mendapatkan prediksi SSPL.")

        st.sidebar.header("Input Fitur untuk Prediksi")
        frequency = st.sidebar.slider("1. Frekuensi (f) [Hz]",
                                      float(df['f'].min()), float(df['f'].max()), float(df['f'].mean()))
        angle = st.sidebar.slider("2. Sudut Serang (alpha) [derajat]",
                                  float(df['alpha'].min()), float(df['alpha'].max()), float(df['alpha'].mean()))
        chord_length = st.sidebar.slider("3. Panjang Chord (c) [meter]",
                                         float(df['c'].min()), float(df['c'].max()), float(df['c'].mean()))
        free_stream_velocity = st.sidebar.slider("4. Kecepatan Aliran Bebas (U_infinity) [m/s]",
                                                 float(df['U_infinity'].min()), float(df['U_infinity'].max()), float(df['U_infinity'].mean()))
        suction_side_displacement = st.sidebar.slider("5. Perpindahan Sisi Hisap (delta) [meter]",
                                                      float(df['delta'].min()), float(df['delta'].max()), float(df['delta'].mean()))

        input_data = np.array([[frequency, angle, chord_length, free_stream_velocity, suction_side_displacement]])
        prediction = model.predict(input_data)[0]

        st.subheader("Hasil Prediksi SSPL")
        st.markdown(f"Dengan input fitur yang diberikan, Tingkat Tekanan Suara (Sound Pressure Level / SSPL) yang diprediksi adalah: **<span style='color:blue; font-size:24px;'>{prediction:.2f} dB</span>**", unsafe_allow_html=True)

    except FileNotFoundError:
        st.error(f"Error: Model '{model_path}' tidak ditemukan.")
        st.info("Pastikan Anda sudah melatih model dan menyimpannya sebagai `linear_regression_model.pkl` dan file tersebut berada di direktori yang sama.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat atau menggunakan model: {e}")
        st.info("Cek apakah nama kolom dan urutan input slider sesuai dengan model yang Anda latih.")

st.markdown("---")

# --- Bagian 4: Informasi Tambahan (Sidebar) ---
st.sidebar.header("Tentang Dashboard Ini")
st.sidebar.info("Dashboard ini dibuat menggunakan **Streamlit** untuk memvisualisasikan data **Airfoil Self-Noise** dan menampilkan prediksi dari model **Regresi Linier Berganda**.")
st.sidebar.markdown("Anda dapat menyesuaikan kode untuk menambahkan lebih banyak fitur, plot, atau model.")
st.sidebar.markdown("---")
st.sidebar.write("Dibuat dengan ❤️ oleh Anda")