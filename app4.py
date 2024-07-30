import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.exceptions import NotFittedError

def main():
    st.title("Aplicación Avanzada de Machine Learning")
    st.write("Carga un archivo de datos, preprocesa, entrena un modelo, y visualiza los resultados.")

    uploaded_file = st.file_uploader("Elija un archivo Excel", type="xlsx")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("Datos cargados:")
        st.dataframe(df.head())

        st.write("Estadísticas descriptivas:")
        st.write(df.describe())


        st.write("Distribución de características numéricas:")
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        for feature in numeric_features:
            fig, ax = plt.subplots()
            sns.histplot(df[feature], kde=True, ax=ax)
            ax.set_title(f'Distribución de {feature}')
            st.pyplot(fig)

        st.write("Distribución de características categóricas:")
        categorical_features = df.select_dtypes(include=['object']).columns
        for feature in categorical_features:
            fig, ax = plt.subplots()
            sns.countplot(y=df[feature], ax=ax)
            ax.set_title(f'Distribución de {feature}')
            st.pyplot(fig)

        st.write("Preprocesando datos...")
        
        df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y-%m-%d %H:%M:%S')
        df['Año'] = df['Fecha'].dt.year
        df['Mes'] = df['Fecha'].dt.month
        df['Día'] = df['Fecha'].dt.day
        df.drop(['Fecha'], axis=1, inplace=True)

        df.fillna(method='ffill', inplace=True)

        features = df.drop('PM2.5', axis=1)
        target = df['PM2.5']
        
        numeric_features = features.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = features.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        models = {
            'RandomForestRegressor': RandomForestRegressor(random_state=42),
            'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
            'LinearRegression': LinearRegression()
        }

        model_selection = st.selectbox('Selecciona el modelo de Machine Learning', options=list(models.keys()))
        model = models[model_selection]
        
        param_grid = {
            'RandomForestRegressor': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30]
            },
            'GradientBoostingRegressor': {
                'model__n_estimators': [50, 100],
                'model__learning_rate': [0.01, 0.1, 0.2]
            },
            'LinearRegression': {}
        }

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])

        if param_grid.get(model_selection):
            st.write("Ajustando hiperparámetros...")
            grid_search = GridSearchCV(pipeline, param_grid[model_selection], cv=5, n_jobs=-1)
            grid_search.fit(features, target)
            st.write("Mejores parámetros encontrados:")
            st.write(grid_search.best_params_)
            best_model = grid_search.best_estimator_
        else:
            st.write("Entrenando el modelo...")
            best_model = pipeline.fit(features, target)

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        y_pred = best_model.predict(X_test)

        st.write("Evaluación del modelo:")
        st.write(f"Error Cuadrático Medio (MSE): {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"Error Absoluto Medio (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"R²: {r2_score(y_test, y_pred):.2f}")
        st.write(f"Explicación de Varianza: {explained_variance_score(y_test, y_pred):.2f}")

        st.write("Análisis de residuos:")
        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_title('Distribución de Residuos')
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.scatterplot(x=y_pred, y=residuals, ax=ax)
        ax.set_xlabel('Valores Predichos')
        ax.set_ylabel('Residuos')
        ax.set_title('Residuos vs Valores Predichos')
        st.pyplot(fig)

        try:
            if hasattr(best_model.named_steps['model'], 'feature_importances_'):
                feature_importances = best_model.named_steps['model'].feature_importances_
                feature_names = best_model.named_steps['preprocessor'].transformers_[0][1].named_steps['scaler'].get_feature_names_out()
                importance_df = pd.DataFrame({
                    'Característica': feature_names,
                    'Importancia': feature_importances
                }).sort_values(by='Importancia', ascending=False)

                st.write("Importancia de características:")
                st.dataframe(importance_df)

                fig, ax = plt.subplots()
                sns.barplot(x='Importancia', y='Característica', data=importance_df, ax=ax)
                ax.set_title('Importancia de Características')
                st.pyplot(fig)
        except NotFittedError:
            st.write("El modelo no se ha ajustado adecuadamente para mostrar la importancia de las características.")
            
        st.write("Comparación de valores reales y predichos:")
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_xlabel('Valores Reales')
        ax.set_ylabel('Valores Predichos')
        ax.set_title('Valores Reales vs Valores Predichos')
        st.pyplot(fig)
        
        st.write("Comparación de valores reales y predichos:")
        comparison_df = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})
        fig, ax = plt.subplots()
        sns.lineplot(data=comparison_df.reset_index(), x=comparison_df.index, y='Real', label='Real', ax=ax)
        sns.lineplot(data=comparison_df.reset_index(), x=comparison_df.index, y='Predicción', label='Predicción', ax=ax)
        ax.set_title('Comparación de Valores Reales y Predicciones')
        ax.set_xlabel('Índice')
        ax.set_ylabel('Valor')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
