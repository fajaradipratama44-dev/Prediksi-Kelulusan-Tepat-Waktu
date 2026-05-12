import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

#1: Membaca/Load data dan model
model = joblib.load('model_random_forest.pkl')
scaler = joblib.load('scaler.pkl')
df = pd.read_csv('Dataset_Mahasiswa1_2022.csv')

#2: Membuat Design/Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #1e3d59;
    }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .reportview-container .sidebar-content { background-color: #262730; }
    h1 { color: #1e3d59; }
    </style>
    """, unsafe_allow_html=True)

#3: Membuat Sidebar Navigation
st.sidebar.title("Menu Utama")
menu = st.sidebar.radio("Pilih Menu:", 
    ["Dashboard Monitoring", "System Early Warning", "Prediksi Mandiri", "Daftar Mahasiswa Berisiko"])

#4: Menu Dashboard
if menu == "Dashboard Monitoring":
    st.title("Dashboard Monitoring Mahasiswa 2022")
    # Statistik Utama
    col1, col2, col3 = st.columns(3)
    jumlah_mahasiswa = len(df)
    ontime_count = len(df[df['Ontime'] == 1])
    late_count = len(df[df['Ontime'] == 0])
    col1.metric("Total Mahasiswa", f"{jumlah_mahasiswa}")
    col2.metric("Prediksi On-Time", f"{ontime_count}", f"{(ontime_count/jumlah_mahasiswa)*100:.1f}%")
    col3.metric("Prediksi Terlambat", f"{late_count}", f"-{(late_count/jumlah_mahasiswa)*100:.1f}%", delta_color="inverse")
    # Chart Distribusi Status
    st.divider()
    fig = px.pie(df, names='Status_Skripsi', title='Distribusi Status Skripsi Saat Ini')
    st.plotly_chart(fig, use_container_width=True)

#5: Menu Early Warning
elif menu == "System Early Warning":
    st.title("System Early Warning")
    st.write("Sistem ini mendeteksi mahasiswa berdasarkan 3 faktor penentu utama hasil pelatihan Model Random Forest.")

    #Menyiapkan Data & Feature Engineering yang sama dengan Model
    df_warning = df.copy()
    dict_status = {'Belum': 0, 'ACC Judul': 1, 'Sempro': 2, 'Semhas': 3, 'Skripsi': 4}
    df_warning['status_val'] = df_warning['Status_Skripsi'].map(dict_status)
    df_warning['Progress_Skripsi'] = df_warning['status_val'] / df_warning['SKS_Lulus']

    #Logika Early Warning berdasarkan "Faktor Penentu Utama" Model:
    # Faktor 1: Progress Skripsi (Rasio rendah)
    # Faktor 2: Jumlah Bimbingan (Masih minim)
    # Faktor 3: Status Skripsi (Tahapan rendah)

    kondisi_kritis = (
        (df_warning['Progress_Skripsi'] < 0.01) |  # Progres sangat lambat dibanding SKS
        (df_warning['Jumlah_Bimbingan'] < 10) |       # Faktor penentu ke-2
        (df_warning['status_val'] < 3)             # Faktor penentu ke-3 (Belum Semhas)
    )

    mahasiswa_berisiko = df_warning[kondisi_kritis].copy()

    #Memberikan Label Alasan berdasarkan fitur yang dipelajari Model
    keterangan = []
    for i, row in mahasiswa_berisiko.iterrows():
        sebab = []
        if row['Progress_Skripsi'] < 0.01: sebab.append("Rasio Progres Rendah")
        if row['Jumlah_Bimbingan'] < 10: sebab.append("Bimbingan Minim")
        if row['status_val'] < 3: sebab.append("Status Belum Aman")
        keterangan.append(", ".join(sebab))
    
    mahasiswa_berisiko['Indikator_Risiko'] = keterangan

    #Tampilan
    st.subheader(f"🚩 Terdeteksi {len(mahasiswa_berisiko)} Mahasiswa Berisiko")
    
    #Menampilkan tabel dengan fokus pada fitur penting
    st.dataframe(
        mahasiswa_berisiko[['NIM', 'Progress_Skripsi', 'Jumlah_Bimbingan', 'Status_Skripsi', 'Indikator_Risiko']],
        use_container_width=True
    )

    st.divider()
    st.subheader("Rekomendasi")
    st.info("""
    Berdasarkan Feature Importance dari model Random Forest, variabel di atas adalah penentu utama kelulusan. 
    Mahasiswa dalam daftar ini disarankan untuk:
    1. Meningkatkan Rasio Progres: Segera selesaikan revisi agar status skripsi naik sebanding dengan jumlah SKS.
    2. Konsistensi Bimbingan: Menambah frekuensi pertemuan dengan pembimbing (Target > 10 kali).
    """)

#6: Menu Fitur Prediksi
elif menu == "Prediksi Mandiri":
    st.title("🔍 Menu Prediksi Kelulusan")
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            ips_1 = st.number_input("IPS_Semester_1", 0.0, 4.0, 3.2)
            ips_2 = st.number_input("IPS_Semester_2", 0.0, 4.0, 3.1)
            ips_3 = st.number_input("IPS_Semester_3", 0.0, 4.0, 3.3)
            ips_4 = st.number_input("IPS_Semester_4", 0.0, 4.0, 3.2)
            ips_5 = st.number_input("IPS_Semester_5", 0.0, 4.0, 3.1)
            ips_6 = st.number_input("IPS_Semester_6", 0.0, 4.0, 3.3)
            ips_7 = st.number_input("IPS_Semester_7", 0.0, 4.0, 3.2)
            mk_gagal = st.number_input("Jumlah_MK_Gagal", 0, 10, 0)
        with col2:
            sks = st.number_input("SKS_Lulus", 0, 150, 140)
            bimb = st.number_input("Jumlah_Bimbingan", 0, 30, 12)
            stat = st.selectbox("Status_Skripsi", ['Belum', 'ACC Judul', 'Sempro', 'Semhas', 'Skripsi'])
        
        btn = st.form_submit_button("Analisis Data")

    if btn:
        # Preprocessing & Predict (Logika sama seperti sebelumnya)
        dict_status = {'Belum': 0, 'ACC Judul': 1, 'Sempro': 2, 'Semhas': 3, 'Skripsi': 4}
        status_enc = dict_status[stat]
        rata_ipk = (ips_1 + ips_2 + ips_3 + ips_4 + ips_5 + ips_6 + ips_7) / 7
        tren = 1 if ips_7 >= ips_1 else 0
        prog = status_enc / sks if sks > 0 else 0
        sks_bimb_norm = scaler.transform([[sks, bimb]])
        
        input_data = np.array([[rata_ipk, tren, mk_gagal, sks_bimb_norm[0][0], sks_bimb_norm[0][1], status_enc, prog]])
        res = model.predict(input_data)
        
        st.divider()
        if res[0] == 1:
            st.success("HASIL: ON TIME")
            st.balloons()
        else:
            st.error("HASIL: BERISIKO TERLAMBAT")
            st.subheader("Rekomendasi Tindakan:")
            if mk_gagal > 3: st.write("- Segera lakukan remedial atau perbaikan nilai mata kuliah gagal.")
            if bimb < 10: st.write("- Tingkatkan intensitas bimbingan minimal 2x seminggu.")
            if status_enc < 3: st.write("- Kejar pendaftaran Sempro dalam 1 bulan ke depan.")

#7: Menu daftar mahasiswa berisiko
elif menu == "Daftar Mahasiswa Berisiko":
    st.title("Daftar Mahasiswa Berisiko (Not On-Time)")
    
    #Menyiapkan fitur untuk prediksi massal
    df_pred = df.copy()
    dict_status = {'Belum': 0, 'ACC Judul': 1, 'Sempro': 2, 'Semhas': 3, 'Skripsi': 4}
    df_pred['Status_Skripsi_Encoded'] = df_pred['Status_Skripsi'].map(dict_status)
    df_pred['Progress_Skripsi'] = df_pred['Status_Skripsi_Encoded'] / df_pred['SKS_Lulus']
    df_pred['Rata_IP'] = df_pred[['IPS_Semester_1', 'IPS_Semester_2', 'IPS_Semester_3', 'IPS_Semester_4', 'IPS_Semester_5', 'IPS_Semester_6', 'IPS_Semester_7']].mean(axis=1)
    df_pred['Tren_IP'] = np.where(df_pred['IPS_Semester_7'] >= df_pred['IPS_Semester_1'], 1, 0)
    
    #Normalisasi SKS dan Bimbingan (menggunakan scaler yang sudah di-load)
    norm_values = scaler.transform(df_pred[['SKS_Lulus', 'Jumlah_Bimbingan']])
    df_pred['SKS_Lulus_Norm'] = norm_values[:, 0]
    df_pred['Jumlah_Bimbingan_Norm'] = norm_values[:, 1]
    
    #Susun fitur sesuai urutan saat training
    X_room = df_pred[['Rata_IP', 'Tren_IP', 'Jumlah_MK_Gagal', 'SKS_Lulus_Norm', 'Jumlah_Bimbingan_Norm', 'Status_Skripsi_Encoded', 'Progress_Skripsi']]
    
    #Model menebak ulang seluruh data
    df_pred['Prediksi_Model'] = model.predict(X_room)
    list_late = df_pred[df_pred['Prediksi_Model'] == 0].copy()
    
    st.write(f"Berdasarkan analisis Model Random Forest, terdapat **{len(list_late)}** mahasiswa yang diprediksi lulus tidak tepat waktu.")

    #Menampilkan tabel
    st.dataframe(
        list_late[['NIM', 'Jumlah_MK_Gagal', 'Jumlah_Bimbingan', 'Status_Skripsi', 'SKS_Lulus']],
        use_container_width=True
    )
    
    #Mengunduh hasilnya
    csv = list_late.to_csv(index=False).encode('utf-8')
    st.download_button("Download Daftar Risiko (CSV)", csv, "mahasiswa_berisiko.csv", "text/csv")