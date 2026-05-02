import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de página
st.set_page_config(page_title="Beijing Air Quality Predictor", layout="wide")

@st.cache_resource
def load_model():
    # mmap_mode='r' permite leer el archivo desde el disco sin cargarlo todo de golpe a la RAM
    return joblib.load('pipeline_rf.pkl', mmap_mode='r')

def main():
    st.title("🌬️ Predicción de Calidad del Aire (PM2.5)")
    st.markdown("""
    Esta aplicación utiliza un modelo de **Random Forest Regressor** para predecir la concentración de partículas finas en Beijing.
    """)

    # --- Sidebar: Configuración de entrada ---
    st.sidebar.header("Parámetros de Entrada")
    
    station = st.sidebar.selectbox("Estación Meteorológica", [
        "Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan", 
        "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Tiantan", 
        "Wanliu", "Wanshouxigong"
    ])
    
    wd = st.sidebar.selectbox("Dirección del Viento (wd)", [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    ])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Variables Meteorológicas")
        temp = st.slider("Temperatura (°C)", -20.0, 45.0, 15.0)
        pres = st.slider("Presión (hPa)", 980.0, 1050.0, 1010.0)
        dewp = st.slider("Punto de Rocío (°C)", -40.0, 30.0, 5.0)
        wspm = st.number_input("Velocidad Viento (m/s)", 0.0, 20.0, 1.5)
        rain = st.number_input("Lluvia (mm)", 0.0, 100.0, 0.0)

    with col2:
        st.subheader("🧪 Contaminantes")
        pm10 = st.number_input("PM10 (µg/m³)", 0.0, 1000.0, 50.0)
        so2 = st.number_input("SO2 (µg/m³)", 0.0, 500.0, 10.0)
        no2 = st.number_input("NO2 (µg/m³)", 0.0, 500.0, 30.0)
        co = st.number_input("CO (µg/m³)", 0.0, 10000.0, 800.0)
        o3 = st.number_input("O3 (µg/m³)", 0.0, 500.0, 40.0)

    # --- Procesamiento de Variables de Ingeniería ---
    # Calculamos variables automáticas basadas en tu configuración de entrenamiento
    temp_dewp_diff = temp - dewp
    log_co = np.log1p(co)
    log_so2 = np.log1p(so2)
    log_no2 = np.log1p(no2)
    
    # Crear DataFrame para el modelo
    input_df = pd.DataFrame([{
        "PM10": pm10, "SO2": so2, "NO2": no2, "CO": co, "O3": o3,
        "TEMP": temp, "PRES": pres, "DEWP": dewp, "RAIN": rain,
        "wd": wd, "WSPM": wspm, "station": station,
        "temp_dewp_diff": temp_dewp_diff,
        "log_CO": log_co, "log_SO2": log_so2, "log_NO2": log_no2,
        # Nota: Aquí deberías añadir hora_sin, mes_sin, etc. según la hora actual
        # o permitir que el usuario las elija. Por simplicidad usamos valores 0:
        "hora_sin": 0, "hora_cos": 1, "mes_sin": 0, "mes_cos": 1,
        "dow_sin": 0, "dow_cos": 1, "season": "Spring"
    }])

    # --- Predicción ---
    if st.button("🚀 Calcular Predicción PM2.5"):
        model = load_model()
        prediction_log = model.predict(input_df)
        # Aplicamos la transformación inversa (expm1 para revertir log1p)
        prediction = np.expm1(prediction_log)[0]

        st.success(f"### Resultado: {prediction:.2f} µg/m³")
        
        # Indicador visual de salud
        if prediction < 50:
            st.info("Calidad del Aire: **Buena** 🟢")
        elif prediction < 100:
            st.warning("Calidad del Aire: **Moderada** 🟡")
        else:
            st.error("Calidad del Aire: **No Saludable** 🔴")

    # --- Sección de Importancia de Variables ---
    st.divider()
    st.subheader("🔍 ¿Qué influye más en la predicción?")
    try:
        importancia = pd.read_csv("feature_importance.csv")
        st.bar_chart(importancia.set_index(importancia.columns[0]))
    except:
        st.info("Sube el archivo 'feature_importance.csv' para ver el análisis de variables.")

if __name__ == "__main__":
    main()