import streamlit as st
import pandas as pd
import io
from googletrans import Translator
import concurrent.futures

def initialize_session_state():
    state_vars = ['combined_df', 'filtered_df', 'invalid_df', 'preview_df', 'translation_progress']
    for var in state_vars:
        if var not in st.session_state:
            st.session_state[var] = None

def main():
    initialize_session_state()
    
    st.title("Extractor y Validador de Datos Excel")

    uploaded_file = st.file_uploader("Selecciona un archivo Excel", type="xlsx", accept_multiple_files=False)
    
    if uploaded_file:
        st.session_state.preview_df = None 
        preview_file(uploaded_file)
        if st.session_state.preview_df is not None:
            sheet_names = pd.ExcelFile(uploaded_file).sheet_names
            sheet_name = st.selectbox("Selecciona una hoja para vista previa", sheet_names)
            if sheet_name:
                preview_file(uploaded_file, sheet_name)
                st.write("Vista previa del archivo cargado:")
                st.dataframe(st.session_state.preview_df.head(20))

        if st.button("Procesar archivo"):
            process_file(uploaded_file)
    
    if st.session_state.combined_df is not None:
        st.dataframe(st.session_state.combined_df)
        
        if st.button("Traducir encabezados"):
            translate_headers()

        if st.button("Validar datos"):
            validate_data()
    
    if st.session_state.filtered_df is not None:
        st.dataframe(st.session_state.filtered_df)
        if st.button("Guardar archivo procesado"):
            save_file()

def preview_file(file, sheet_name=None):
    try:
        if sheet_name:
            df = pd.read_excel(file, engine='openpyxl', sheet_name=sheet_name, header=None)
        else:
            df = pd.read_excel(file, engine='openpyxl', header=None)
        st.session_state.preview_df = df
    except Exception as e:
        st.error(f"Error al previsualizar el archivo {file.name}: {e}")

def process_file(file):
    try:
        df = pd.read_excel(file, engine='openpyxl', header=None)
        header_row_index = detect_header_row(df)
        if header_row_index is None:
            st.warning(f"No se detectaron encabezados en el archivo {file.name}. Por favor selecciona manualmente.")
            return

        df.columns = df.iloc[header_row_index].tolist()
        df = df.drop(df.index[header_row_index])
        
        df.columns = handle_duplicate_columns(df.columns.tolist())

        st.session_state.combined_df = df
        st.success("Archivo procesado correctamente.")
    except Exception as e:
        st.error(f"Error al procesar el archivo {file.name}: {e}")

def detect_header_row(df):
    for i, row in df.iterrows():
        if row.notnull().all():
            return i
    return None

def handle_duplicate_columns(columns):
    from collections import Counter

    col_counts = Counter(columns)
    for col, count in col_counts.items():
        if count > 1:
            indices = [i for i, x in enumerate(columns) if x == col]
            for i in range(1, count):
                columns[indices[i]] = f"{col}_{i}"
    return columns

def validate_data():
    try:
        valid_rows = []
        invalid_rows = []

        for index, row in st.session_state.combined_df.iterrows():
            if row.notnull().all() and row.astype(str).apply(lambda x: x.strip() != "").all():
                valid_rows.append(row)
            else:
                invalid_rows.append(row)

        st.session_state.filtered_df = pd.DataFrame(valid_rows)
        st.session_state.invalid_df = pd.DataFrame(invalid_rows)
        st.success("Validación completada.")
    except Exception as e:
        st.error(f"Error durante la validación: {e}")

def translate_headers():
    try:
        st.session_state.translation_progress = st.progress(0)
        translator = Translator()
        df = st.session_state.combined_df.copy()

        translated_headers = []
        for i, header in enumerate(df.columns):
            translated_header = translator.translate(header, src='en', dest='es').text
            translated_headers.append(translated_header)
            st.session_state.translation_progress.progress((i + 1) / len(df.columns))

        df.columns = translated_headers
        st.session_state.combined_df = df
        st.success("Traducción de encabezados completada.")
    except Exception as e:
        st.error(f"Error durante la traducción: {e}")

def save_file():
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            st.session_state.filtered_df.to_excel(writer, sheet_name='Sheet1', index=False)
            st.session_state.invalid_df.to_excel(writer, sheet_name='Sheet2', index=False)
        buffer.seek(0)

        st.download_button(
            label="Descargar archivo procesado",
            data=buffer,
            file_name="archivo_procesado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Error al guardar el archivo: {e}")

if __name__ == "__main__":
    main()
