import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

analysis_option = st.radio(
    "Pilih Tampilan Analisis Data:",
    ("Statistik Deskriptif", "Heatmap Korelasi", "Scatter Plot: Semua Fitur vs SSPL"), # Opsi baru
    key="analysis_selector"
)

if analysis_option == "Statistik Deskriptif":
    st.subheader("Statistik Deskriptif Seluruh Data")
    st.write(df.describe())

elif analysis_option == "Heatmap Korelasi":
    st.subheader("Heatmap Korelasi Antar Fitur dan Target")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
    ax_corr.set_title('Matriks Korelasi')
    st.pyplot(fig_corr)

elif analysis_option == "Scatter Plot: Semua Fitur vs SSPL":
    st.subheader("Scatter Plot: Fitur vs Sound Pressure Level (SSPL)")
    # Daftar fitur yang ingin di-plot (kecuali target SSPL itu sendiri)
    features_to_plot = ['f', 'alpha', 'c', 'U_infinity', 'delta']
    
    # Membuat plot untuk setiap fitur vs SSPL
    num_features = len(features_to_plot)
    # Sesuaikan ukuran grid plot agar tidak terlalu padat
    cols = 2 # Jumlah kolom dalam grid plot
    rows = (num_features + cols - 1) // cols # Hitung jumlah baris yang dibutuhkan
    
    fig_all_scatter, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = axes.flatten() # Ratakan array axes untuk iterasi mudah

    for i, feature in enumerate(features_to_plot):
        sns.scatterplot(x=feature, y='SSPL', data=df, ax=axes[i], alpha=0.6)
        axes[i].set_title(f'{feature} vs SSPL')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('SSPL (dB)')
    
    # Sembunyikan subplot yang kosong jika ada
    for j in range(i + 1, len(axes)):
        fig_all_scatter.delaxes(axes[j])

    plt.tight_layout() # Penyesuaian layout agar tidak tumpang tindih
    st.pyplot(fig_all_scatter)


st.markdown("---")

# --- Bagian 3: Prediksi Tingkat Kebisingan (SSPL) ---
st.header("Prediksi Tingkat Kebisingan (SSPL)")

model_path = 'linear_regression_model.pkl'

@st.cache_resource
def load_my_model(path):
    try:
        model = joblib.load(path)
        st.success(f"Model prediksi berhasil dimuat dari '{path}'!")
        return model
    except FileNotFoundError:
        st.error(f"Error: Model '{path}' tidak ditemukan.")
        st.info("Pastikan Anda sudah melatih model dan menyimpannya sebagai `linear_regression_model.pkl` dan file tersebut berada di direktori yang sama.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

model = load_my_model(model_path)

if model is not None:
    st.write("Gunakan slider di sidebar kiri untuk menginput nilai fitur dan mendapatkan prediksi SSPL.")

    st.sidebar.header("Input Fitur untuk Prediksi")
    
    # --- Penyesuaian Range Slider ---
    # Min value dari 0, Max value 1.1x dari nilai maks di data
    frequency = st.sidebar.slider("1. Frekuensi (f) [Hz]",
                                  0.0, float(df['f'].max() * 1.1), float(df['f'].mean()))
    angle = st.sidebar.slider("2. Sudut Serang (alpha) [derajat]",
                              0.0, float(df['alpha'].max() * 1.1), float(df['alpha'].mean()))
    chord_length = st.sidebar.slider("3. Panjang Chord (c) [meter]",
                                     0.0, float(df['c'].max() * 1.1), float(df['c'].mean()))
    free_stream_velocity = st.sidebar.slider("4. Kecepatan Aliran Bebas (U_infinity) [m/s]",
                                             0.0, float(df['U_infinity'].max() * 1.1), float(df['U_infinity'].mean()))
    suction_side_displacement = st.sidebar.slider("5. Perpindahan Sisi Hisap (delta) [meter]",
                                                  0.0, float(df['delta'].max() * 1.1), float(df['delta'].mean()))

    input_data = np.array([[frequency, angle, chord_length, free_stream_velocity, suction_side_displacement]])
    prediction = model.predict(input_data)[0]

    st.subheader("Hasil Prediksi SSPL")

    # --- Logika untuk menentukan warna dan kategori (Sesuai Konteks Pesawat Komersial) ---
    color = "black"
    category_text = "Tidak Terdefinisi"

    if prediction > 120:
        color = "red"
        category_text = "Buruk (Kontribusi kebisingan sangat tinggi)"
    elif 100 <= prediction <= 120:
        color = "orange"
        category_text = "Cukup Ideal (Kontribusi kebisingan moderat)"
    elif 80 <= prediction < 100:
        color = "blue"
        category_text = "Ideal (Kontribusi kebisingan rendah)"
    elif prediction < 80:
        color = "green"
        category_text = "Sangat Ideal (Kontribusi kebisingan sangat rendah)"
    
    st.markdown(f"Dengan input fitur yang diberikan, Tingkat Tekanan Suara (Sound Pressure Level / SSPL) yang diprediksi adalah: **<span style='color:{color}; font-size:24px;'>{prediction:.2f} dB</span>**", unsafe_allow_html=True)
    st.markdown(f"**Kategori Kebisingan:** <span style='color:{color}; font-size:18px;'>{category_text}</span>", unsafe_allow_html=True)

else:
    st.warning("Model prediksi belum tersedia atau gagal dimuat. Pastikan file `linear_regression_model.pkl` ada dan valid.")


st.markdown("---")

# --- Bagian 4: Informasi Tambahan (Sidebar) ---
st.sidebar.header("Tentang Dashboard Ini")
st.sidebar.info("Dashboard ini dibuat menggunakan **Streamlit** untuk memvisualisasikan data **Airfoil Self-Noise** dan menampilkan prediksi dari model **Regresi Linier Berganda**.")
st.sidebar.markdown("Anda dapat menyesuaikan kode untuk menambahkan lebih banyak fitur, plot, atau model.")
st.sidebar.markdown("---")
st.sidebar.write("Dibuat dengan ❤️ oleh Kelompok Kami")