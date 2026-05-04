# Customer Churn Prediction ML App

Proyecto end-to-end de Machine Learning para predecir churn de clientes de telecomunicaciones. Incluye EDA, preprocesamiento, comparacion de modelos, interpretabilidad y una app interactiva en Streamlit.

## Objetivo

Predecir la probabilidad de que un cliente cancele el servicio, usando variables demograficas, servicios contratados, tipo de contrato, metodo de pago y cargos mensuales/totales.

## Dataset

Dataset recomendado: **Telco Customer Churn**, basado en datos de muestra de IBM.

- Fuente principal: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Respaldo reproducible: [CSV raw en GitHub](https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv)

El dataset contiene 7,043 clientes y variables como `tenure`, `Contract`, `InternetService`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` y `Churn`.

## Estructura

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   └── README.md
├── models/
├── notebooks/
│   └── 01_eda_modeling_guide.ipynb
├── reports/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── predict.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Como Ejecutarlo

1. Crear entorno virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Descargar datos:

```bash
python src/data_loader.py
```

4. Generar EDA:

```bash
python src/eda.py
```

5. Entrenar y comparar modelos:

```bash
python src/train.py
```

6. Ejecutar app:

```bash
streamlit run app/streamlit_app.py
```

En Windows tambien puedes usar:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_streamlit.ps1
```

## EDA

El script `src/eda.py` genera visualizaciones en `reports/`:

- Distribucion de churn.
- Distribuciones de `tenure`, `MonthlyCharges` y `TotalCharges` por churn.
- Tasa de churn por tipo de contrato.
- Mapa de correlacion para variables numericas.

Hallazgos esperados:

- Clientes con contrato mensual suelen tener mayor churn.
- Menor `tenure` suele asociarse con mayor riesgo.
- Cargos mensuales altos pueden elevar el riesgo, especialmente con fibra optica y contrato mensual.

## Preprocesamiento

El pipeline limpia y transforma los datos de forma reproducible:

- Convierte `TotalCharges` a numerico y rellena nulos con la mediana.
- Convierte `SeniorCitizen` a categoria `Yes/No`.
- Elimina identificadores como `customerID`.
- Codifica variables categoricas con `OneHotEncoder`.
- Normaliza variables numericas con `StandardScaler`.

## Modelado

Se entrenan y comparan tres modelos:

- Regresion Logistica.
- Random Forest.
- XGBoost.

Metricas guardadas en `reports/model_metrics.csv`:

- Accuracy.
- Precision.
- Recall.
- F1.
- ROC-AUC.

El mejor modelo por ROC-AUC se guarda en `models/best_churn_model.joblib`.

## Interpretabilidad

El proyecto genera:

- `reports/feature_importance.csv`.
- `reports/feature_importance.png`.
- `reports/shap_summary.png`, si SHAP puede calcularse correctamente con el modelo ganador.

Esto permite explicar que variables impulsan el riesgo de churn, por ejemplo contrato mensual, baja antiguedad, metodo de pago o cargos altos.

## App Streamlit

La app permite ingresar datos de un cliente y devuelve:

- Probabilidad estimada de churn.
- Nivel de riesgo.
- Tabla de comparacion de modelos.
- Grafica de variables mas influyentes.

## Valor Para Portfolio

Este proyecto demuestra habilidades practicas de:

- Analisis exploratorio con pandas, matplotlib y seaborn.
- Pipelines de preprocesamiento con scikit-learn.
- Comparacion de modelos supervisados.
- Evaluacion con metricas de clasificacion relevantes para negocio.
- Interpretabilidad de modelos.
- Deploy local con Streamlit.
- Estructura profesional de repositorio.

## Mejoras Futuras

- Agregar validacion cruzada y busqueda de hiperparametros.
- Ajustar umbral de clasificacion segun costo de falsos negativos.
- Crear tests unitarios para limpieza y prediccion.
- Dockerizar la app.
- Deploy en Streamlit Community Cloud.
