import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

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

EXPECTED_COLUMNS = [
    'ID', 'N_Days', 'Status', 'Drug', 'Age', 'Sex', 'Ascites', 'Hepatomegaly',
    'Spiders', 'Edema', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
    'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage'
]

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

def clear_model_cache():
    """Clear the model cache to force reload after retrain"""
    if 'load_model' in st.cache_resource:
        st.cache_resource.clear()

def get_accuracy(model_key):
    try:
        metrics_path = os.path.join(MODEL_DIR, MODELOS[model_key]['metrics'])
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return round(metrics['accuracy'] * 100, 2)
    except:
        return "--"

def validate_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    
    if missing or extra:
        return df, False, missing, extra, None
    
    valid_status = {'D', 'C', 'CL'}
    invalid_status = [s for s in df['Status'].unique() if s not in valid_status]
    
    if invalid_status:
        return df, False, [], [], f"Valores inválidos en Status: {invalid_status}"
    
    valid_sex = {'M', 'F'}
    invalid_sex = [s for s in df['Sex'].unique() if s not in valid_sex]
    
    if invalid_sex:
        return df, False, [], [], f"Valores inválidos en Sex: {invalid_sex}"
    
    valid_drug = {'D-penicillamine', 'Placebo'}
    invalid_drug = [d for d in df['Drug'].dropna().unique() if d not in valid_drug]
    
    if invalid_drug:
        return df, False, [], [], f"Valores inválidos en Drug: {invalid_drug}"
    
    return df, True, [], [], None

def generate_new_ids(df_uploaded, df_base):
    max_id = df_base['ID'].max()
    new_ids = list(range(int(max_id) + 1, int(max_id) + 1 + len(df_uploaded)))
    df_copy = df_uploaded.copy()
    df_copy['ID'] = new_ids
    return df_copy

def preprocess_dataframe(df):
    data = df.copy()
    
    data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})
    data['Drug'] = data['Drug'].map({'D-penicillamine': 1, 'Placebo': 0})
    
    binary_vars = ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
    for var in binary_vars:
        data[var] = data[var].map({'Y': 1, 'N': 0, 'S': 0.5})
    
    data['Status'] = data['Status'].map({'D': 1, 'C': 0, 'CL': 0})
    
    feature_cols = [col for col in data.columns if col not in ['ID', 'N_Days', 'Status']]
    X = data[feature_cols]
    y = data['Status']
    
    valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
    
    return X[valid_idx], y[valid_idx], valid_idx

def retrain_model(model_key, df_combined):
    X, y, valid_idx = preprocess_dataframe(df_combined)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_key == 'logistic':
        model = LogisticRegression()
    else:
        model = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.01, activation='tanh')
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'sensitivity': float(recall_score(y_test, y_pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'confusion_matrix': {
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        }
    }
    
    model_path = os.path.join(MODEL_DIR, MODELOS[model_key]['model'])
    scaler_path = os.path.join(MODEL_DIR, MODELOS[model_key]['scaler'])
    metrics_path = os.path.join(MODEL_DIR, MODELOS[model_key]['metrics'])
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    return model, scaler, metrics

def predict_batch(model, scaler, df_uploaded):
    X_batch, y_batch, valid_idx = preprocess_dataframe(df_uploaded)
    
    if len(X_batch) == 0:
        return pd.DataFrame()
    
    X_scaled = scaler.transform(X_batch)
    
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    results = pd.DataFrame({
        'ID': df_uploaded.loc[valid_idx, 'ID'].values,
        'Status_Real': df_uploaded.loc[valid_idx, 'Status'].values,
        'Prediccion': ['Muerte' if p == 1 else 'Supervivencia' for p in predictions],
        'Probabilidad_Supervivencia': np.round(probabilities[:, 0], 4),
        'Probabilidad_Muerte': np.round(probabilities[:, 1], 4)
    })
    
    return results

def main():
    st.title("🏥 Predictor de Supervivencia")
    st.markdown("### Cirrosis Primaria Biliar (PBC)")

    tab1, tab2 = st.tabs(["👤 Predicción Individual", "📊 Predicción por Lote"])

    with tab1:
        st.subheader("Ingrese los datos del paciente")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Datos Demográficos**")
            age = st.number_input("Edad (años)", min_value=18, max_value=120, value=50, key='age')
            sex = st.selectbox("Sexo", options=[0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino", key='sex')
            
            st.markdown("**Síntomas Clínicos**")
            ascites = st.selectbox("Ascitis", options=[0, 1, 0.5], format_func=lambda x: {0: "No", 1: "Sí", 0.5: "Leve"}[x], key='ascites')
            hepatomegaly = st.selectbox("Hepatomegalia", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí", key='hepatomegaly')
            spiders = st.selectbox("Arañas Vasculares", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí", key='spiders')
            edema = st.selectbox("Edema", options=[0, 1, 0.5], format_func=lambda x: {0: "No", 1: "Sí", 0.5: "Leve"}[x], key='edema')

        with col2:
            st.markdown("**Análisis de Sangre**")
            bilirubin = st.number_input("Bilirrubina (mg/dL)", min_value=0.0, value=1.0, step=0.1, key='bilirubin')
            cholesterol = st.number_input("Colesterol (mg/dL)", min_value=0.0, value=200.0, step=0.1, key='cholesterol')
            albumin = st.number_input("Albúmina (g/dL)", min_value=0.0, value=3.5, step=0.1, key='albumin')
            copper = st.number_input("Cobre (μg/L)", min_value=0.0, value=50.0, step=0.1, key='copper')
            alk_phos = st.number_input("Fosfatasa Alcalina (U/L)", min_value=0.0, value=100.0, step=0.1, key='alk_phos')
            sgot = st.number_input("SGOT (U/L)", min_value=0.0, value=80.0, step=0.1, key='sgot')
            tryglicerides = st.number_input("Triglicéridos (mg/dL)", min_value=0.0, value=120.0, step=0.1, key='tryglicerides')
            platelets = st.number_input("Plaquetas (10^9/L)", min_value=0.0, value=150.0, step=0.1, key='platelets')
            prothrombin = st.number_input("Tiempo de Protrombina (s)", min_value=0.0, value=10.0, step=0.1, key='prothrombin')

        col_stage, col_drug = st.columns(2)
        with col_stage:
            stage = st.selectbox("Estadío (1-4)", options=[1, 2, 3, 4], key='stage')
        with col_drug:
            drug = st.selectbox("Fármaco", options=[0, 1], format_func=lambda x: "Placebo" if x == 0 else "D-penicilamina", key='drug')

        features = {
            'Age': age, 'Sex': sex, 'Ascites': ascites, 'Hepatomegaly': hepatomegaly,
            'Spiders': spiders, 'Edema': edema, 'Bilirubin': bilirubin,
            'Cholesterol': cholesterol, 'Albumin': albumin, 'Copper': copper,
            'Alk_Phos': alk_phos, 'SGOT': sgot, 'Tryglicerides': tryglicerides,
            'Platelets': platelets, 'Prothrombin': prothrombin, 'Stage': stage, 'Drug': drug
        }

        st.markdown("---")
        
        col_model, col_btn = st.columns([3, 1])
        with col_model:
            model_key = st.selectbox(
                "Seleccione el Modelo",
                options=['logistic', 'nn'],
                format_func=lambda x: f"{MODELOS[x]['name']} ({get_accuracy(x)}% Accuracy)",
                key='model_selector'
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🔮 Predecir", type="primary", use_container_width=True)

        if predict_btn:
            with st.spinner("Cargando modelo..."):
                try:
                    model, scaler = load_model(model_key)
                    X = np.array([[features[f] for f in FEATURES]])
                    X_scaled = scaler.transform(X)
                    prediction = model.predict(X_scaled)[0]
                    proba = model.predict_proba(X_scaled)[0]
                    
                    if prediction == 1:
                        label = "🔴 ALTO RIESGO"
                        description = "El modelo predice mayor probabilidad de muerte"
                    else:
                        label = "🟢 BAJO RIESGO"
                        description = "El modelo predice mayor probabilidad de supervivencia"
                    
                    st.markdown("---")
                    st.markdown(f"## Resultado: {label}")
                    st.markdown(f"**{description}**")
                    st.progress(1 - proba[0] if prediction == 1 else proba[0])
                    st.markdown(f"**Probabilidad de supervivencia:** {proba[0]*100:.1f}%")
                    st.markdown(f"**Probabilidad de muerte:** {proba[1]*100:.1f}%")
                    
                except Exception as e:
                    st.error(f"Error en la predicción: {str(e)}")

        with st.expander("📊 Métricas del modelo"):
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
            
            st.markdown("### Matriz de Confusión")
            cm = metrics['confusion_matrix']
            
            fig, ax = plt.subplots(figsize=(5, 4))
            cm_array = np.array([[cm['TN'], cm['FP']], [cm['FN'], cm['TP']]])
            conf_matrix = ConfusionMatrixDisplay(confusion_matrix=cm_array, display_labels=['Supervivencia', 'Muerte'])
            conf_matrix.plot(ax=ax, cmap='Blues')
            st.pyplot(fig)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("TP", cm['TP'])
            with c2:
                st.metric("TN", cm['TN'])
            with c3:
                st.metric("FP", cm['FP'])
            with c4:
                st.metric("FN", cm['FN'])

    with tab2:
        st.subheader("📊 Predicción por Lote")

        st.info(
            "📋 **Requisitos del CSV:**\n\n"
            "- Exactamente 20 columnas: " + ", ".join(EXPECTED_COLUMNS) + "\n"
            "- **Status** debe contener: D (muerte), C (supervivencia), CL (censurado)\n"
            "- **Sex** debe contener: M (masculino), F (femenino)\n"
            "- **Drug** debe contener: D-penicillamine, Placebo\n"
            "- Los IDs se generarán automáticamente\n"
            "- El dataset se combinará con la base de datos original y se reentrenará el modelo seleccionado"
        )

        csv_template = pd.DataFrame(columns=EXPECTED_COLUMNS)
        csv_bytes = csv_template.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Template CSV",
            data=csv_bytes,
            file_name="template_cirrosis.csv",
            mime="text/csv"
        )

        uploaded_file = st.file_uploader("Sube tu CSV aquí", type=['csv'], key='batch_upload')

        if uploaded_file:
            df_uploaded, is_valid, missing, extra, error_msg = validate_csv(uploaded_file)
            
            if not is_valid:
                st.error("❌ **CSV inválido:**")
                if missing:
                    st.error(f"Columnas faltantes: {missing}")
                if extra:
                    st.warning(f"Columnas extra: {extra}")
                if error_msg:
                    st.error(error_msg)
            else:
                st.success(f"✅ **CSV válido:** {len(df_uploaded)} registros listos para procesar")
                
                if st.button("🔄 Procesar Lote", type="primary"):
                    with st.spinner("Preprocesando datos..."):
                        try:
                            original_path = os.path.join(MODEL_DIR, 'cirrhosis.csv')
                            updated_path = os.path.join(MODEL_DIR, 'cirrhosis_updated.csv')
                            
                            df_original = pd.read_csv(original_path)
                            
                            with st.spinner("Generando nuevos IDs..."):
                                df_with_ids = generate_new_ids(df_uploaded, df_original)
                            
                            with st.spinner("Combinando con base de datos original..."):
                                df_combined = pd.concat([df_original, df_with_ids], ignore_index=True)
                                df_combined.to_csv(updated_path, index=False)
                            
                            with st.spinner(f"Reentrenando modelo: {MODELOS[model_key]['name']}..."):
                                model, scaler, metrics = retrain_model(model_key, df_combined)
                            
                            clear_model_cache()
                            
                            with st.spinner("Generando predicciones..."):
                                results = predict_batch(model, scaler, df_with_ids)
                            
                            if len(results) > 0:
                                st.success("✅ **Lote procesado correctamente**")
                                
                                st.markdown("### Resultados de Predicción")
                                st.dataframe(results, use_container_width=True)
                                
                                csv_results = results.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Descargar predicciones",
                                    data=csv_results,
                                    file_name="predicciones_lote.csv",
                                    mime="text/csv"
                                )
                                
                                st.markdown("### Métricas del Modelo Reentrenado")
                                
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
                                
                                st.markdown("### Matriz de Confusión")
                                cm = metrics['confusion_matrix']
                                
                                fig, ax = plt.subplots(figsize=(5, 4))
                                cm_array = np.array([[cm['TN'], cm['FP']], [cm['FN'], cm['TP']]])
                                conf_matrix = ConfusionMatrixDisplay(confusion_matrix=cm_array, display_labels=['Supervivencia', 'Muerte'])
                                conf_matrix.plot(ax=ax, cmap='Blues')
                                st.pyplot(fig)
                                
                                c1, c2, c3, c4 = st.columns(4)
                                with c1:
                                    st.metric("TP", cm['TP'])
                                with c2:
                                    st.metric("TN", cm['TN'])
                                with c3:
                                    st.metric("FP", cm['FP'])
                                with c4:
                                    st.metric("FN", cm['FN'])
                                
                                st.info(
                                    f"📁 **Archivos actualizados:**\n\n"
                                    f"- `cirrhosis_updated.csv` - Base de datos combinada ({len(df_combined)} registros)\n"
                                    f"- `{MODELOS[model_key]['model']}` - Modelo reentrenado\n"
                                    f"- `{MODELOS[model_key]['scaler']}` - Scaler actualizado\n"
                                    f"- `{MODELOS[model_key]['metrics']}` - Métricas actualizadas\n\n"
                                    f"Nuevo accuracy: **{metrics['accuracy']*100:.2f}%**"
                                )
                            else:
                                st.warning("⚠️ No se pudieron generar predicciones. Todos los registros tenían valores faltantes.")
                            
                        except Exception as e:
                            st.error(f"Error al procesar el lote: {str(e)}")
                            st.error("La base de datos original no fue modificada.")

if __name__ == "__main__":
    main()
