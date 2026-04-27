# 🏥 Predictor de Supervivencia - Cirrosis Primaria Biliar (PBC)

## Descripción

Aplicación Streamlit para predecir la supervivencia de pacientes con Cirrosis Primaria Biliar (PBC) utilizando dos modelos de Machine Learning:

- **Regresión Logística**: 81.93% Accuracy
- **Red Neuronal (MLP)**: 74.70% Accuracy

## Uso Local

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar la aplicación
streamlit run app.py
```

La aplicación estará disponible en: **http://localhost:8501**

## Deployment a Streamlit Cloud

### Opción 1: GitHub + share.streamlit.io

1. **Sube este proyecto a GitHub** (repo público):
   ```
   .
   ├── app.py                    # Aplicación Streamlit
   ├── requirements.txt          # Dependencias
   ├── cirrhosis.csv            # Dataset original
   ├── cirrhosis_logistic_model.pkl
   ├── cirrhosis_logistic_scaler.pkl
   ├── cirrhosis_logistic_metrics.json
   ├── cirrhosis_nn_model.pkl
   ├── cirrhosis_nn_scaler.pkl
   └── cirrhosis_nn_metrics.json
   ```

2. **Ve a https://share.streamlit.io/**
3. **New App** → selecciona tu repo y branch → **Main** → **app.py** → **Deploy**

### Opción 2: CLI

```bash
pip install streamlit
streamlit deploy
```

## Estructura

| Archivo | Descripción |
|---------|-------------|
| `app.py` | Aplicación Streamlit completa |
| `requirements.txt` | Dependencias Python |
| `cirrhosis_*.pkl` | Modelos entrenados y scalers |
| `cirrhosis_*.json` | Métricas de evaluación |

## Características

### 📋 Predicción Individual

- Formulario con 17 características clínicas
- Selector de modelo en sidebar
- Predicción con probabilidad

### 📊 Análisis por Lote

- 6 métricas de desempeño (Accuracy, Precision, Recall, F1, Sensitivity, Specificity)
- Matriz de confusión visual

## Notas

⚠️ **Esta aplicación es una herramienta de apoyo clínico**. Las predicciones no reemplan el juicio médico profesional.

---

**Última actualización**: Abril 2026