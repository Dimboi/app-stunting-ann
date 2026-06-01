import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_components():
    # Cukup load model dan scaler saja (le_jk dan le_status tidak lagi dipakai)
    model = joblib.load('model_stunting_mlp.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_components()

# Desain UI
st.title("👶Aplikasi Prediksi Risiko Stunting Balita👶")
st.write("biasa lah ygy... ibu low iq")

st.sidebar.header("Masukkan Data Balita")
umur = st.sidebar.number_input("Umur (Bulan)", min_value=0, max_value=60, value=12)
jenis_kelamin = st.sidebar.selectbox("Jenis Kelamin", ['laki-laki', 'perempuan'])
tinggi_badan = st.sidebar.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=130.0, value=75.0, step=0.1)

# Logika Prediksi
if st.button("Prediksi Status Gizi"):
    
    # 1. MAPPING MANUAL JENIS KELAMIN 
    # (Sesuai urutan abjad saat kita melatih model: laki-laki=0, perempuan=1)
    jk_encoded = 0 if jenis_kelamin == 'laki-laki' else 1
    
    # 2. Susun dan skalakan input
    input_data = np.array([[umur, jk_encoded, tinggi_badan]])
    input_scaled = scaler.transform(input_data)
    
    # 3. Lakukan prediksi angka kelas
    pred_class = model.predict(input_scaled)[0]
    
    # 4. MAPPING MANUAL STATUS GIZI
    # (Sesuai urutan abjad LabelEncoder di Colab)
    status_mapping = {
        0: 'normal',
        1: 'severely stunted',
        2: 'stunted',
        3: 'tinggi'
    }
    hasil_prediksi = status_mapping[pred_class]
    
    # 5. Tampilkan Hasil
    st.markdown("### Hasil Prediksi:")
    if hasil_prediksi in ['normal', 'tinggi']:
        st.success(f"Status Gizi: **{hasil_prediksi.upper()}** ✅")
        st.write("*(Keluarga Cemara")
    else:
        st.error(f"Status Gizi: **{hasil_prediksi.upper()}** ⚠️")
        st.write("*(Hina aja ibu ibunya*")
