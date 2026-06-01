import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cache agar loading lebih cepat
@st.cache_resource
def load_components():
    # Load model menggunakan joblib (bukan dari keras/tensorflow lagi)
    model = joblib.load('model_stunting_mlp.pkl')
    scaler = joblib.load('scaler.pkl')
    le_jk = joblib.load('le_jk.pkl')
    le_status = joblib.load('le_status.pkl')
    return model, scaler, le_jk, le_status

model, scaler, le_jk, le_status = load_components()

# Desain UI
st.title("👶 Aplikasi Prediksi Risiko Stunting Balita")
st.write("Aplikasi ini menggunakan Artificial Neural Network (ANN) untuk memprediksi status gizi balita berdasarkan standar WHO.")

st.sidebar.header("Masukkan Data Balita")
umur = st.sidebar.number_input("Umur (Bulan)", min_value=0, max_value=60, value=12)
jenis_kelamin = st.sidebar.selectbox("Jenis Kelamin", ['laki-laki', 'perempuan'])
tinggi_badan = st.sidebar.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=130.0, value=75.0, step=0.1)

# Logika Prediksi
if st.button("Prediksi Status Gizi"):
    jk_encoded = le_jk.transform([jenis_kelamin])[0]
    input_data = np.array([[umur, jk_encoded, tinggi_badan]])
    input_scaled = scaler.transform(input_data)
    
    # MLPClassifier di scikit-learn bisa langsung memprediksi kelas
    pred_class = model.predict(input_scaled)[0]
    hasil_prediksi = le_status.inverse_transform([pred_class])[0]
    
    st.markdown("### Hasil Prediksi:")
    if hasil_prediksi == 'normal' or hasil_prediksi == 'tinggi':
        st.success(f"Status Gizi: **{hasil_prediksi.upper()}** ✅")
    else:
        st.error(f"Status Gizi: **{hasil_prediksi.upper()}** ⚠️")
    
    st.write("*(Catatan: Ini adalah prediksi AI, pastikan untuk tetap berkonsultasi dengan dokter anak atau posyandu terdekat)*")
