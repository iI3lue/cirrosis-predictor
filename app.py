import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(
    page_title="Predicción de Supervivencia - Cirrosis",
    page_icon="🏥",
    layout="wide"
)

MODEL_DIR = os.path.dirname(os.path.realpath(__file__))

MODELOS = {
    'logistic': {
        'model': 'cirrhosis_logistic_model.pkl',
        'scaler': 'cirrhosis_logistic_scaler.pkl',
        'metrics': 'cirrhosis_logistic_metrics.json',
        'name': 'Regresión Logística'
    },
    'nn': {
        'model': 'cirrhosis_nn_model.pkl',
        'scaler': 'cirrhosis_nn_scaler.pkl',
        'metrics': 'cirrhosis_nn_metrics.json',
        'name': 'Red Neuronal'
    }
}

FEATURES = [
    'Age', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema',
    'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos',
    'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage', 'Drug'
]

@st.cache_resource
def load_model(model_key):
    model_path = os.path.join(MODEL_DIR, MODELOS[model_key]['model'])
    scaler_path = os.path.join(MODEL_DIR, MODELOS[model_key]['scaler'])
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

@st.cache_resource
def load_metrics(model_key):
    metrics_path = os.path.join(MODEL_DIR, MODELOS[model_key]['metrics'])
    with open(metrics_path, 'r') as f:
        return f.read()

def main():
    st.title("🏥 Predictor de Supervivencia")
    st.markdown("### Cirrosis Primaria Biliar (PBC)")

    with st.sidebar:
        st.header("⚙️ Configuración")
        model_key = st.selectbox(
            "Modelo de Predicción",
            options=['logistic', 'nn'],
            format_func=lambda x: f"{MODELOS[x]['name']} ({get_accuracy(x)}%)",
            key='model_selector'
        )
        
        st.markdown("---")
        st.markdown("**Accuracy del modelo seleccionado:**")
        accuracy = get_accuracy(model_key)
        st.metric("Accuracy", f"{accuracy}%")

    tab1, tab2 = st.tabs(["📋 Predicción Individual", "📊 Análisis por Lote"])

    with tab1:
        prediction_individual(model_key)
    
    with tab2:
        batch_analysis(model_key)

def get_accuracy(model_key):
    try:
        import json
        metrics_path = os.path.join(MODEL_DIR, MODELOS[model_key]['metrics'])
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return round(metrics['accuracy'] * 100, 2)
    except:
        return "--"

def prediction_individual(model_key):
    st.markdown("### Predicción Individual")
    st.markdown("Ingrese los datos del paciente para obtener una predicción de supervivencia")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Datos Demográficos")
        age = st.number_input("Edad (años)", min_value=18, max_value=120, value=50, key='age')
        sex = st.selectbox("Sexo", options=[0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino", key='sex')
        
        st.subheader("Síntomas Clínicos")
        ascites = st.selectbox("Ascitis", options=[0, 1, 0.5], format_func=lambda x: {0: "No", 1: "Sí", 0.5: "Leve"}[x], key='ascites')
        hepatomegaly = st.selectbox("Hepatomegalia", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí", key='hepatomegaly')
        spiders = st.selectbox("Arañas Vasculares", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí", key='spiders')
        edema = st.selectbox("Edema", options=[0, 1, 0.5], format_func=lambda x: {0: "No", 1: "Sí", 0.5: "Leve"}[x], key='edema')

    with col2:
        st.subheader("Análisis de Sangre")
        bilirubin = st.number_input("Bilirrubina (mg/dL)", min_value=0.0, value=1.0, step=0.1, key='bilirubin')
        cholesterol = st.number_input("Colesterol (mg/dL)", min_value=0.0, value=200.0, step=0.1, key='cholesterol')
        albumin = st.number_input("Albúmina (g/dL)", min_value=0.0, value=3.5, step=0.1, key='albumin')
        copper = st.number_input("Cobre (μg/L)", min_value=0.0, value=50.0, step=0.1, key='copper')
        alk_phos = st.number_input("Fosfatasa Alcalina (U/L)", min_value=0.0, value=100.0, step=0.1, key='alk_phos')
        sgot = st.number_input("SGOT (U/L)", min_value=0.0, value=80.0, step=0.1, key='sgot')
        tryglicerides = st.number_input("Triglicéridos (mg/dL)", min_value=0.0, value=120.0, step=0.1, key='tryglicerides')
        platelets = st.number_input("Plaquetas (10^9/L)", min_value=0.0, value=150.0, step=0.1, key='platelets')
        prothrombin = st.number_input("Tiempo de Protrombina (s)", min_value=0.0, value=10.0, step=0.1, key='prothrombin')

    st.subheader("Estadificación y Tratamiento")
    col_stage, col_drug = st.columns(2)
    with col_stage:
        stage = st.selectbox("Estadío", options=[1, 2, 3, 4], key='stage')
    with col_drug:
        drug = st.selectbox("Fármaco", options=[0, 1], format_func=lambda x: "Placebo" if x == 0 else "D-penicilamina", key='drug')

    features = {
        'Age': age, 'Sex': sex, 'Ascites': ascites, 'Hepatomegaly': hepatomegaly,
        'Spiders': spiders, 'Edema': edema, 'Bilirubin': bilirubin,
        'Cholesterol': cholesterol, 'Albumin': albumin, 'Copper': copper,
        'Alk_Phos': alk_phos, 'SGOT': sgot, 'Tryglicerides': tryglicerides,
        'Platelets': platelets, 'Prothrombin': prothrombin, 'Stage': stage, 'Drug': drug
    }

    if st.button("🔮 Predecir", type="primary"):
        with st.spinner("Cargando modelo..."):
            try:
                model, scaler = load_model(model_key)
                X = np.array([[features[f] for f in FEATURES]])
                X_scaled = scaler.transform(X)
                prediction = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0]

                prob_survival = proba[0] if prediction == 0 else proba[1]
                
                if prediction == 1:
                    label = "🔴 ALTO RIESGO"
                    description = "El modelo predice mayor probabilidad de muerte"
                else:
                    label = "🟢 BAJO RIESGO"
                    description = "El modelo predice mayor probabilidad de supervivencia"
                
                st.markdown("---")
                st.markdown(f"## Resultado: {label}")
                st.markdown(f"**{description}**")
                st.progress(1 - prob_survival if prediction == 1 else prob_survival)
                st.markdown(f"**Probabilidad de supervivencia:** {proba[0]*100:.1f}%")
                st.markdown(f"**Probabilidad de muerte:** {proba[1]*100:.1f}%")
                
            except Exception as e:
                st.error(f"Error en la predicción: {str(e)}")

def batch_analysis(model_key):
    st.markdown("### Análisis por Lote")
    st.markdown("Métricas y matriz de confusión del modelo en el conjunto de prueba")

    import json

    try:
        metrics_path = os.path.join(MODEL_DIR, MODELOS[model_key]['metrics'])
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        with col2:
            st.metric("Precision", f"{metrics['precision']*100:.2f}%")
        with col3:
            st.metric("Recall", f"{metrics['recall']*100:.2f}%")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")
        with col5:
            st.metric("Sensitivity", f"{metrics['sensitivity']*100:.2f}%")
        with col6:
            st.metric("Specificity", f"{metrics['specificity']*100:.2f}%")

        st.markdown("---")
        st.markdown("### 🔲 Matriz de Confusión")

        cm = metrics['confusion_matrix']
        cm_data = pd.DataFrame(
            [[cm['TN'], cm['FP']], [cm['FN'], cm['TP']]],
            index=['Actual: Supervivencia', 'Actual: Muerte'],
            columns=['Predicho: Supervivencia', 'Predicho: Muerte']
        )
        st.dataframe(cm_data.style.format("{:d}").background_gradient(cmap="Blues", axis=None), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("TP (Verdaderos Positivos)", cm['TP'])
        with c2:
            st.metric("TN (Verdaderos Negativos)", cm['TN'])
        with c3:
            st.metric("FP (Falsos Positivos)", cm['FP'])
        with c4:
            st.metric("FN (Falsos Negativos)", cm['FN'])

    except Exception as e:
        st.error(f"Error al cargar métricas: {str(e)}")

if __name__ == "__main__":
    main()