import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def main():
    st.title("Análisis Avanzado de Datos y Predicciones")

    uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("Datos cargados:")
        st.dataframe(df)

        if 'Fecha' in df.columns:
            try:
                df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
                df['Año'] = df['Fecha'].dt.year
                df['Mes'] = df['Fecha'].dt.month
                df['Día'] = df['Fecha'].dt.day
            except Exception as e:
                st.error(f"Error al convertir las fechas: {e}")

        st.subheader("Análisis Exploratorio de Datos")
        st.write("Estadísticas Descriptivas")
        st.write(df.describe())

        st.write("Correlaciones")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

        df['Contaminación'] = df[['No2', 'SO2', 'CO', 'O3']].sum(axis=1)
        critical_cities = df.groupby('Ciudad')['Contaminación'].mean().sort_values(ascending=False).head(10)
        st.write("Top 10 Ciudades Críticas")
        st.dataframe(critical_cities)

        critical_countries = df.groupby('País')['Contaminación'].mean().sort_values(ascending=False).head(10)
        st.write("Top 10 Países Críticos")
        st.dataframe(critical_countries)

        features = ['PM2.5', 'PM10', 'No2', 'SO2', 'CO', 'O3', 'Temperatura', 'Velocidad del viento']
        targets = {
            'PM2.5': 'PM2.5',
            'Temperatura': 'Temperatura',
            'Humedad': 'Humedad'
        }

        def train_and_evaluate(models, X_train, X_test, y_train, y_test):
            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                results[name] = {
                    "model": model,
                    "mse": mse,
                    "r2": r2,
                    "mae": mae,
                    "predictions": predictions
                }
            return results

        def display_model_results(results, target_name):
            st.subheader(f"Comparación de Modelos para {target_name}")
            fig, ax = plt.subplots(figsize=(12, 6))
            models_names = list(results.keys())
            mse_values = [results[name]['mse'] for name in models_names]
            mae_values = [results[name]['mae'] for name in models_names]
            r2_values = [results[name]['r2'] for name in models_names]

            x = np.arange(len(models_names))
            ax.bar(x - 0.3, mse_values, 0.3, label='MSE', color='orange')
            ax.bar(x, mae_values, 0.3, label='MAE', color='blue')
            ax.bar(x + 0.3, r2_values, 0.3, label='R2', color='green')
            ax.set_xlabel('Modelos')
            ax.set_ylabel('Valores')
            ax.set_title(f'Comparación de Modelos para {target_name}')
            ax.set_xticks(x)
            ax.set_xticklabels(models_names, rotation=45)
            ax.legend()
            st.pyplot(fig)

            st.write(f"Resultados detallados para {target_name}:")
            results_df = pd.DataFrame({
                "Modelo": models_names,
                "MSE": mse_values,
                "MAE": mae_values,
                "R2": r2_values
            })
            st.dataframe(results_df)

        if all(col in df.columns for col in features + list(targets.values())):
            X = df[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            results = {}
            for target_name, target_column in targets.items():
                y = df[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                models = {
                    "Regresión Lineal": LinearRegression(),
                    "Regresión Ridge": Ridge(),
                    "Regresión Lasso": Lasso(),
                    "Random Forest": RandomForestRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "SVR": SVR(),
                    "Red Neuronal": MLPRegressor(max_iter=500)
                }

                results[target_name] = train_and_evaluate(models, X_train, X_test, y_train, y_test)
                display_model_results(results[target_name], target_name)

            st.subheader("Selecciona un modelo para predicción")
            model_option = st.selectbox("Modelo", list(models.keys()))

            st.subheader("Predicción de Nuevos Datos")
            num_predictions = 10
            new_data = X.sample(num_predictions, random_state=42)
            st.write("Datos de entrada para predicciones:")
            st.dataframe(new_data)

            new_data_scaled = scaler.transform(new_data)

            predictions = {}
            for target_name, result in results.items():
                model = result[model_option]['model']
                predictions[target_name] = model.predict(new_data_scaled)

            st.write("Predicciones:")
            for target_name, preds in predictions.items():
                st.write(f"Predicciones de {target_name}:")
                st.write(preds)

            st.subheader("Gráficos de Predicciones")
            fig, axs = plt.subplots(3, 2, figsize=(15, 15))
            axs = axs.flatten()  

            for idx, (target_name, preds) in enumerate(predictions.items()):
                ax = axs[idx]
                ax.plot(range(num_predictions), preds, label=f"Predicciones de {target_name}", marker='o')
                ax.set_title(f"Predicciones de {target_name}")
                ax.set_xlabel("Índice de Predicción")
                ax.set_ylabel(target_name)
                ax.legend()


            for idx, target_name in enumerate(targets.keys()):
                ax = axs[3 + idx]  
                y_test = df[targets[target_name]]
                preds = results[target_name][model_option]['predictions']

          
                if len(y_test) == len(preds):
                    ax.scatter(y_test, preds, label=f"{target_name} Real vs Predicho")
                    ax.set_title(f"Comparación {target_name} Real vs Predicho")
                    ax.set_xlabel("Valor Real")
                    ax.set_ylabel("Valor Predicho")
                    ax.legend()
                else:
                    
                    fig_alt, ax_alt = plt.subplots(figsize=(10, 6))
                    ax_alt.hist(y_test, bins=20, alpha=0.5, label='Valores Reales')
                    ax_alt.hist(preds, bins=20, alpha=0.5, label='Predicciones')
                    ax_alt.set_title(f"Distribución de {target_name} Real vs Predicho")
                    ax_alt.set_xlabel(target_name)
                    ax_alt.set_ylabel("Frecuencia")
                    ax_alt.legend()
                    st.pyplot(fig_alt)

              
                    fig_box, ax_box = plt.subplots(figsize=(10, 6))
                    ax_box.boxplot([y_test, preds], labels=['Valores Reales', 'Predicciones'])
                    ax_box.set_title(f"Diagrama de Caja para {target_name}")
                    ax_box.set_ylabel(target_name)
                    st.pyplot(fig_box)

            st.pyplot(fig)

        else:
            st.error("Faltan algunas columnas necesarias en el archivo.")

if __name__ == "__main__":
    main()
