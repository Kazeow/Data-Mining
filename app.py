import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Fungsi untuk memuat dataset
def load_data():
    # Pastikan file 'regression.csv' ada di folder yang sama atau gunakan path yang benar
    return pd.read_csv('regression.csv')

# Fungsi utama aplikasi
def main():
    st.set_page_config(
        page_title="FAMA : Regression \n UAS DATA MINING",
        layout="wide",
        page_icon="📊"
    )

    # Header aplikasi
    st.markdown(
        """
        <h1 style="text-align:center; color:#2F4F4F;">FAMA : Regression \n UAS DATA MINING</h1>
        """,
        unsafe_allow_html=True
    )
    # Menambahkan foto untuk mempercantik
    st.subheader("Galeri Foto")

    # Foto dalam baris
    st.markdown("### 📸 IMAGE")
    cols = st.columns(7)  # Membuat 3 kolom
    with cols[0]:
        st.image("agevsage.png", caption="AGE VS AGE", use_container_width=True)
    with cols[1]:
        st.image("sexvssex.png", caption="SEX VS SEX", use_container_width=True)
    with cols[2]:
        st.image("bmivsmbi.png", caption="BMI VS BMI", use_container_width=True)
    with cols[3]:
        st.image("children.png", caption="CHILDREN VS CHILDREN", use_container_width=True)
    with cols[4]:
        st.image("region.png", caption="REGION VS REGION", use_container_width=True)
    with cols[5]:
        st.image("smoker.png", caption="SMOKER VS SMOKER", use_container_width=True)
    with cols[6]:
        st.image("charges.png", caption="CHARGES VS CHARGES", use_container_width=True)

    # Sidebar dengan logo
    st.sidebar.image("logo.png", use_container_width=True)
    st.sidebar.header("Navigasi Menu : ")
    section = st.sidebar.radio(
        "Menu:",
        ["Overview", "Data Preview", "Visualisasi", "Modeling"]
    )

    # Memuat data
    try:
        data = load_data()
    except FileNotFoundError:
        st.error("File dataset tidak ditemukan. Pastikan file `regression.csv` tersedia di folder.")
        return

    # Halaman Overview
    if section == "Overview":
        st.markdown(
            """
            <h3 style="text-align:center;">Selamat datang di FAMA</h3>
            <p style="text-align:center;">
               Gak ribet lagi, aplikasi ini bantu kita analisis data dan buat model yang akurat
            </p>
            """,
            unsafe_allow_html=True
        )

    # Halaman Data Preview
    elif section == "Data Preview":
        st.subheader("📄 Dataset Preview")
        st.write("Data yang digunakan dalam analisis:")
        st.dataframe(data)

    # Halaman Visualisasi
    elif section == "Visualisasi":
        st.subheader("📊 Visualisasi Data")
        st.write("Pilih variabel untuk membuat visualisasi.")
        
        # Dropdown untuk memilih kolom
        x_var = st.selectbox("Pilih Variabel X", options=data.columns)
        y_var = st.selectbox("Pilih Variabel Y", options=data.columns)
        color_var = st.selectbox("Pilih Variabel Warna (opsional)", options=[None] + list(data.columns))

        # Membuat grafik bar
        if x_var and y_var:
            st.write("Visualisasi hubungan antar variabel:")
            fig = px.bar(
                data,
                x=x_var,
                y=y_var,
                color=color_var,
                barmode="group",
                title=f"Bar Plot: {x_var} vs {y_var}"
            )
            st.plotly_chart(fig)

    # Halaman Modeling
    elif section == "Modeling":
        st.subheader("🤖 Analisis Regresi Logistik")
        st.write("Pilih variabel untuk membangun model regresi.")

        # Memilih variabel
        x_var = st.selectbox("Pilih Fitur (X)", options=data.columns)
        y_var = st.selectbox("Pilih Target (Y)", options=data.columns)

        # Memisahkan data
        X = data[[x_var]]
        y = data[y_var]

        # Membagi data menjadi train dan test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Membuat model regresi
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Menampilkan metrik evaluasi
        st.write("### 📉 Evaluasi Model")
        st.write(f"- **Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"- **R-squared (R2):** {r2_score(y_test, y_pred):.2f}")

        # Membuat grafik prediksi
        st.write("### 📈 Grafik Prediksi vs Aktual")
        fig = px.scatter(
            x=X_test[x_var],
            y=y_test,
            labels={'x': 'Aktual', 'y': 'Prediksi'},
            title="Prediksi vs Aktual"
        )
        fig.add_scatter(
            x=X_test[x_var],
            y=y_pred,
            mode='markers',
            name='Prediksi'
        )
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
