# Airfoil Self-Noise Analysis & Prediction Dashboard

Proyek ini menghadirkan sebuah aplikasi web interaktif yang dibangun dengan **Streamlit** untuk analisis data dan prediksi *Sound Pressure Level* (SSPL) dari dataset Airfoil Self-Noise. Tujuannya adalah untuk memvisualisasikan karakteristik kebisingan airfoil serta menyediakan antarmuka pengguna yang intuitif untuk menguji skenario prediksi berdasarkan model *machine learning* yang telah dilatih.

### Penjelasan Variabel

| Variabel       | Keterangan Detail                                                                 | Satuan            | Tipe Data |
| :------------- | :-------------------------------------------------------------------------------- | :---------------- | :-------- |
| **`f`** | **Frekuensi Suara:** Frekuensi yang diukur dari gelombang suara yang dihasilkan oleh *airfoil*. | Hertz (Hz)        | Numerik   |
| **`alpha`** | **Sudut Serang (Angle of Attack):** Sudut antara *chord line* *airfoil* dan arah aliran udara relatif. | Derajat           | Numerik   |
| **`c`** | **Panjang Chord (Chord Length):** Jarak dari tepi depan ke tepi belakang *airfoil*.          | Meter (m)         | Numerik   |
| **`U_infinity`** | **Kecepatan Aliran Bebas (Free-stream Velocity):** Kecepatan udara di luar pengaruh *airfoil*. | Meter/detik (m/s) | Numerik   |
| **`delta`** | **Ketebalan Lapisan Batas Sisi Hisap (Suction Side Displacement Thickness):** Ketebalan lapisan batas perpindahan pada sisi hisap *airfoil*. | Meter (m)         | Numerik   |
| **`SSPL`** | **Sound Pressure Level (SPL):** Tingkat tekanan suara, yaitu **variabel target** yang ingin diprediksi. | Desibel (dB)      | Numerik   |

---
## 📂 Struktur Proyek

.
├── AirfoilSelfNoise.csv        # Dataset utama
├── dashboard_airfoil.py        # Kode utama aplikasi Streamlit
├── linear_regression_model.pkl # Model ML yang sudah terlatih (dihasilkan setelah melatih model)
└── README.md                   # File dokumentasi ini

**Fitur Utama:**

* **Pemuatan Data Efisien:** Mengintegrasikan proses pemuatan dataset `AirfoilSelfNoise.csv` dengan penanganan kesalahan yang robust.
* **Analisis Data Eksplorasi (EDA) Interaktif:** Menyajikan statistik deskriptif dan visualisasi data kunci (seperti distribusi frekuensi dan *scatter plot*) yang dapat dipilih pengguna melalui kontrol radio (`st.radio`) untuk eksplorasi dinamis.
* **Model Regresi Linier Berganda:** Memanfaatkan `scikit-learn` untuk membangun dan melatih model regresi linier berganda yang memprediksi SSPL berdasarkan lima fitur input (`f`, `alpha`, `c`, `U_infinity`, `delta`).
* **Evaluasi Model:** Menampilkan metrik evaluasi model standar (MAE, MSE, RMSE, R²) untuk mengukur kinerja prediksi.
* **Persistensi Model:** Menggunakan `joblib` untuk menyimpan dan memuat model yang telah dilatih (`linear_regression_model.pkl`), memungkinkan aplikasi web untuk langsung melakukan prediksi tanpa pelatihan ulang.
* **Antarmuka Prediksi Real-time:** Menyediakan *slider* interaktif di sidebar bagi pengguna untuk menginput nilai fitur secara dinamis. Model kemudian akan menghasilkan prediksi SSPL secara *real-time* berdasarkan input tersebut.
* **Deployment-Ready (Lokal):** Dirancang untuk dijalankan secara lokal menggunakan perintah `streamlit run`, memudahkan pengujian dan demonstrasi.

**Teknologi yang Digunakan:**

* **Python**
* **Streamlit:** Untuk membangun aplikasi web interaktif.
* **Pandas:** Untuk manipulasi dan analisis data.
* **NumPy:** Untuk komputasi numerik.
* **Matplotlib & Seaborn:** Untuk visualisasi data.
* **Scikit-learn:** Untuk pembangunan dan evaluasi model *machine learning*.
* **Joblib:** Untuk serialisasi dan deserialisasi model Python.

**Cara Menjalankan:**

1.  Kloning repositori ini.
2.  Instal dependensi yang diperlukan:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
    ```
3.  Pastikan file `AirfoilSelfNoise.csv` dan `linear_regression_model.pkl` (hasil dari pelatihan model) berada di direktori yang sama dengan `dashboard_airfoil.py`.
4.  Jalankan aplikasi Streamlit dari terminal:
    ```bash
    streamlit run dashboard_airfoil.py
    ```

---