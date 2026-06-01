import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Gunakan cache agar model tidak perlu dimuat ulang setiap kali tombol ditekan
@st.cache_resource
def load_components():
    model = load_model('model_stunting.h5')
    scaler = joblib.load('scaler.pkl')
    le_jk = joblib.load('le_jk.pkl')
    le_status = joblib.load('le_status.pkl')
    return model, scaler, le_jk, le_status

model, scaler, le_jk, le_status = load_components()

# Desain Tampilan Aplikasi
st.title("👶 Aplikasi Prediksi Risiko Stunting Balita")
st.write("Aplikasi ini menggunakan Artificial Neural Network (ANN) untuk memprediksi status gizi balita berdasarkan standar WHO.")

st.sidebar.header("Masukkan Data Balita")

# Input Pengguna di Sidebar
umur = st.sidebar.number_input("Umur (Bulan)", min_value=0, max_value=60, value=12)
jenis_kelamin = st.sidebar.selectbox("Jenis Kelamin", ['laki-laki', 'perempuan'])
tinggi_badan = st.sidebar.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=130.0, value=75.0, step=0.1)

# Tombol Prediksi
if st.button("Prediksi Status Gizi"):
    # 1. Ubah teks jenis kelamin menjadi angka seperti saat training
    jk_encoded = le_jk.transform([jenis_kelamin])[0]
    
    # 2. Susun data input
    input_data = np.array([[umur, jk_encoded, tinggi_badan]])
    
    # 3. Standarisasi data input (wajib agar model tidak salah hitung)
    input_scaled = scaler.transform(input_data)
    
    # 4. Prediksi dengan model ANN
    pred_prob = model.predict(input_scaled)
    pred_class = np.argmax(pred_prob, axis=1)[0]
    
    # 5. Kembalikan angka prediksi menjadi teks (normal, stunted, dll)
    hasil_prediksi = le_status.inverse_transform([pred_class])[0]
    
    # 6. Tampilkan Hasil yang Menarik
    st.markdown("### Hasil Prediksi:")
    if hasil_prediksi == 'normal' or hasil_prediksi == 'tinggi':
        st.success(f"Status Gizi: **{hasil_prediksi.upper()}** ✅")
    else:
        st.error(f"Status Gizi: **{hasil_prediksi.upper()}** ⚠️")
    
    st.write("*(Catatan: Ini adalah prediksi AI, pastikan untuk tetap berkonsultasi dengan dokter anak atau posyandu terdekat)*")
