
import streamlit as st

st.set_page_config(page_title="Mi Aplicación Streamlit", layout="wide")

import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from googletrans import Translator
import time
from app import main as app_main
from app2 import main as app2_main
from app3 import main as app3_main
from app4 import main as app4_main  

def initialize_session_state():
    if 'combined_df' not in st.session_state:
        st.session_state.combined_df = None
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'invalid_df' not in st.session_state:
        st.session_state.invalid_df = None

def add_progress_bar_to_sidebar():
    progress_bar = st.sidebar.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    st.sidebar.success('¡Carga completada!')

def sidebar_design():
    st.sidebar.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #2E2E2E;
            padding: 20px;
            border-radius: 10px;
            color: #FFFFFF;
        }
        .sidebar .sidebar-content a {
            color: #1E90FF;
        }
        .sidebar .sidebar-content a:hover {
            color: #FFD700;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.sidebar.title("Navegación")
    add_progress_bar_to_sidebar()
    selection = st.sidebar.radio("Ir a", [
        "Extractor y validador de datos (Global)", 
        "Análisis de datos Normal (Propietario)", 
        "Análisis de datos avanzado - Predicciones (Propietario)",
        "Machine Learning (Propietario)"  
    ])
    return selection

def add_animations_and_widgets():
    with st.spinner('Cargando...'):
        time.sleep(2)
    st.success('¡Listo!')

def main():
    st.markdown(
        """
        <style>
        body {
            background-color: #121212;
            color: #FFFFFF;
        }
        .main {
            background-color: #1E1E1E;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.5);
        }
        .stButton>button {
            background-color: #1E90FF;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #FFD700;
        }
        .stProgress .st-bo {
            background-color: #1E90FF;
        }
        </style>
        """, unsafe_allow_html=True
    )

    selection = sidebar_design()
    initialize_session_state()
    add_animations_and_widgets()

    if selection == "Extractor y validador de datos (Global)":
        app_main()
    elif selection == "Análisis de datos Normal (Propietario)":
        app2_main()
    elif selection == "Análisis de datos avanzado - Predicciones (Propietario)":
        app3_main()
    elif selection == "Machine Learning (Propietario)":
        app4_main() 

if __name__ == "__main__":
    main()
