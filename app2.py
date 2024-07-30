import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import calendar

def clean_data(df):
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    return df.dropna()

def analyze_data(df, analysis_type, column):
    try:
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df['Mes'] = df['Fecha'].dt.to_period('M')

        if analysis_type == 'Promedio Mensual':
            result = df.groupby('Mes')[column].mean().reset_index(name='Promedio')
            return result
        elif analysis_type == 'Máximo Mensual':
            result = df.groupby('Mes')[column].max().reset_index(name='Máximo')
            return result
        elif analysis_type == 'Promedio por País':
            result = df.groupby('País')[column].mean().reset_index(name='Promedio')
            return result
        elif analysis_type == 'Tendencia a lo Largo del Tiempo':
            result = df.groupby('Fecha')[column].mean().reset_index(name='Promedio')
            return result
        elif analysis_type == 'Correlación entre Contaminantes':
            correlation = df.corr().loc[:, df.columns[3:-1]].reset_index()
            return correlation
        elif analysis_type == 'Tendencia Estacional':
            result = df.groupby(df['Fecha'].dt.month)[column].mean().reset_index(name='Promedio')
            result['Mes'] = result['Fecha'].apply(lambda x: calendar.month_name[x])
            return result[['Mes', 'Promedio']]
        elif analysis_type == 'Comparación entre Ciudades':
            result = df.groupby('Ciudad')[column].mean().reset_index(name='Promedio')
            return result
        else:
            grouped = df.groupby('Ciudad')[column].agg(
                mean='mean',
                sum='sum',
                median='median',
                std='std',
                max='max',
                min='min',
                IQR=lambda x: x.quantile(0.75) - x.quantile(0.25)
            ).reset_index()

            analysis_map = {
                'Promedio por Ciudad': lambda: grouped[['Ciudad', 'mean']],
                'Total por Ciudad': lambda: grouped[['Ciudad', 'sum']],
                'Mediana por Ciudad': lambda: grouped[['Ciudad', 'median']],
                'Desviación estándar por Ciudad': lambda: grouped[['Ciudad', 'std']],
                'Máximo por Ciudad': lambda: grouped[['Ciudad', 'max']],
                'Mínimo por Ciudad': lambda: grouped[['Ciudad', 'min']],
                'Rango Intercuartílico por Ciudad': lambda: grouped[['Ciudad', 'IQR']],
                'Tendencia': lambda: df[['Fecha', column]].sort_values(by='Fecha').reset_index(drop=True),
            }

            return analysis_map.get(analysis_type, lambda: pd.DataFrame())()

    except Exception as e:
        st.error(f"Error durante el análisis de datos: {e}")
        return pd.DataFrame()

def generate_graph(df, analysis_type, graph_type='bar'):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        if graph_type == 'bar':
            df.plot(kind='bar', x=df.columns[0], y=df.columns[1], ax=ax, color='skyblue')
        elif graph_type == 'line':
            df.plot(kind='line', x=df.columns[0], y=df.columns[1], ax=ax, marker='o')
        elif graph_type == 'scatter':
            df.plot(kind='scatter', x=df.columns[0], y=df.columns[1], ax=ax, color='darkblue')
        elif graph_type == 'pie':
            df.set_index(df.columns[0]).plot(kind='pie', y=df.columns[1], ax=ax, autopct='%1.1f%%', legend=False)

        ax.set_title(analysis_type)
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
        ax.grid(True)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error al generar el gráfico: {e}")
        return None

def save_graph(fig, format='png'):
    try:
        buf = BytesIO()
        fig.savefig(buf, format=format, bbox_inches='tight')
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Error al guardar el gráfico: {e}")
        return None

def main():
    st.title("Análisis de Datos - Normal")

    uploaded_file = st.file_uploader("Sube un archivo Excel", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file, sheet_name='Sheet1', engine='openpyxl')
        df = clean_data(df)
        st.write(f"Datos de la hoja 'Sheet1' del archivo Excel:")
        st.dataframe(df)

        analysis_type_default = 'Promedio por Ciudad'
        column_default = df.select_dtypes(include=['number']).columns[0] if not df.empty else None
        st.subheader("Análisis Básico")
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_columns:
            st.warning("No hay columnas numéricas en los datos. Por favor, seleccione un archivo con datos numéricos.")
        else:
            column = st.selectbox("Seleccione la columna para el análisis", numeric_columns, index=0)
            analysis_type = st.selectbox("Seleccione el tipo de análisis", 
                                         ["Promedio por Ciudad", 
                                          "Total por Ciudad", 
                                          "Mediana por Ciudad",
                                          "Desviación estándar por Ciudad",
                                          "Máximo por Ciudad",
                                          "Mínimo por Ciudad",
                                          "Rango Intercuartílico por Ciudad",
                                          "Promedio Mensual",
                                          "Máximo Mensual",
                                          "Promedio por País",
                                          "Tendencia a lo Largo del Tiempo",
                                          "Correlación entre Contaminantes",
                                          "Tendencia Estacional",
                                          "Comparación entre Ciudades"], index=0)

            graph_type = st.selectbox("Seleccione el tipo de gráfico", ["bar", "line", "scatter", "pie"], index=0)

            if st.button("Generar gráfico de datos"):
                analysis_result = analyze_data(df, analysis_type, column)
                if not analysis_result.empty:
                    st.write("Resultados del análisis de datos:")
                    st.dataframe(analysis_result)
                    st.write("Gráfico generado:")
                    fig = generate_graph(analysis_result, analysis_type, graph_type)

                    if fig:
                        st.pyplot(fig)
                        buffer = save_graph(fig, format='png')
                        if buffer:
                            filename = f"grafico_{analysis_type}.png"
                            st.download_button(
                                label="Guardar gráfico",
                                data=buffer,
                                file_name=filename,
                                mime="image/png"
                            )

if __name__ == "__main__":
    main()
