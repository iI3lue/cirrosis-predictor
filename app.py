import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# ─────────────────────────────────────────────
#  CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PBC Survival Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Fondo general */
.stApp {
    background: #f0f4f8;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f2744 0%, #1a3d6b 60%, #0d3358 100%);
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #e8f0fb !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c5d8f5 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #fff !important;
}

/* ── Header principal ── */
.main-header {
    background: linear-gradient(135deg, #0f2744 0%, #1e5096 100%);
    border-radius: 18px;
    padding: 32px 40px;
    margin-bottom: 28px;
    color: white;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.main-header::after {
    content: '';
    position: absolute;
    bottom: -60px; right: 80px;
    width: 280px; height: 280px;
    background: rgba(255,255,255,0.03);
    border-radius: 50%;
}
.main-header h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}
.main-header p {
    font-size: 1rem;
    opacity: 0.75;
    margin: 0;
    font-weight: 300;
}
.header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 12px;
    color: #a8d4ff;
}

/* ── Cards / Secciones ── */
.section-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(15,39,68,0.06);
}
.section-title {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #64748b;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 2px solid #f1f5f9;
}

/* ── Resultado de predicción ── */
.result-card-high {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border: 2px solid #f87171;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 20px 0;
}
.result-card-low {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border: 2px solid #4ade80;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 20px 0;
}
.result-label {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.result-desc {
    font-size: 0.95rem;
    opacity: 0.8;
    margin-bottom: 18px;
}
.prob-row {
    display: flex;
    gap: 24px;
    margin-top: 12px;
}
.prob-box {
    background: rgba(255,255,255,0.55);
    border-radius: 12px;
    padding: 12px 20px;
    min-width: 140px;
    text-align: center;
}
.prob-box .val {
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
}
.prob-box .lbl {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    opacity: 0.65;
    margin-top: 2px;
}

/* ── Métricas ── */
[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 18px !important;
}
[data-testid="stMetric"] label {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: #64748b !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.4rem !important;
    color: #0f2744 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #e8edf5;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 8px 22px !important;
    color: #64748b !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #0f2744 !important;
    box-shadow: 0 2px 6px rgba(15,39,68,0.12) !important;
}

/* ── Botones ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a56db 0%, #0f2744 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(26,86,219,0.35) !important;
}
.stButton > button:not([kind="primary"]) {
    border-radius: 10px !important;
    border: 1.5px solid #cbd5e1 !important;
    font-weight: 500 !important;
}

/* ── Inputs ── */
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stNumberInput > div > div > input:focus,
.stSelectbox > div > div:focus-within {
    border-color: #1a56db !important;
    box-shadow: 0 0 0 3px rgba(26,86,219,0.12) !important;
}

/* ── Alerts ── */
.stAlert {
    border-radius: 12px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Sidebar métricas ── */
.sidebar-metric {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    text-align: center;
}
.sidebar-metric .sm-val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #7dd3fc;
    font-family: 'DM Mono', monospace;
}
.sidebar-metric .sm-lbl {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.6;
    margin-top: 2px;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.1);
    margin: 18px 0;
}
.model-badge {
    display: inline-block;
    background: rgba(125,211,252,0.15);
    border: 1px solid rgba(125,211,252,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #7dd3fc;
    margin-bottom: 8px;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #1a56db !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #1a56db, #0f2744) !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.realpath(__file__))

MODELOS = {
    'logistic': {
        'model':   'cirrhosis_logistic_model.pkl',
        'scaler':  'cirrhosis_logistic_scaler.pkl',
        'metrics': 'cirrhosis_logistic_metrics.json',
        'name':    'Regresión Logística'
    },
    'nn': {
        'model':   'cirrhosis_nn_model.pkl',
        'scaler':  'cirrhosis_nn_scaler.pkl',
        'metrics': 'cirrhosis_nn_metrics.json',
        'name':    'Red Neuronal'
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

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_key):
    model_path  = os.path.join(MODEL_DIR, MODELOS[model_key]['model'])
    scaler_path = os.path.join(MODEL_DIR, MODELOS[model_key]['scaler'])
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def clear_model_cache():
    """Limpia la caché de recursos para forzar recarga del modelo."""
    st.cache_resource.clear()


def get_metrics(model_key):
    try:
        metrics_path = os.path.join(MODEL_DIR, MODELOS[model_key]['metrics'])
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def validate_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra   = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    if missing or extra:
        return df, False, missing, extra, None

    invalid_status = [s for s in df['Status'].unique() if s not in {'D', 'C', 'CL'}]
    if invalid_status:
        return df, False, [], [], f"Valores inválidos en Status: {invalid_status}"

    invalid_sex = [s for s in df['Sex'].unique() if s not in {'M', 'F'}]
    if invalid_sex:
        return df, False, [], [], f"Valores inválidos en Sex: {invalid_sex}"

    invalid_drug = [d for d in df['Drug'].dropna().unique() if d not in {'D-penicillamine', 'Placebo'}]
    if invalid_drug:
        return df, False, [], [], f"Valores inválidos en Drug: {invalid_drug}"

    return df, True, [], [], None


def generate_new_ids(df_uploaded, df_base):
    max_id  = df_base['ID'].max()
    new_ids = list(range(int(max_id) + 1, int(max_id) + 1 + len(df_uploaded)))
    df_copy = df_uploaded.copy()
    df_copy['ID'] = new_ids
    return df_copy


def preprocess_dataframe(df):
    data = df.copy()
    data['Sex']  = data['Sex'].map({'M': 1, 'F': 0})
    data['Drug'] = data['Drug'].map({'D-penicillamine': 1, 'Placebo': 0})
    for var in ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema']:
        data[var] = data[var].map({'Y': 1, 'N': 0, 'S': 0.5})
    data['Status'] = data['Status'].map({'D': 1, 'C': 0, 'CL': 0})

    feature_cols = [c for c in data.columns if c not in ['ID', 'N_Days', 'Status']]
    X = data[feature_cols]
    y = data['Status']
    valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
    return X[valid_idx], y[valid_idx], valid_idx


def retrain_model(model_key, df_combined):
    X, y, _ = preprocess_dataframe(df_combined)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if model_key == 'logistic':
        model = LogisticRegression(max_iter=500)
    else:
        model = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.01, activation='tanh', max_iter=500)

    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    cm          = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'accuracy':    float(accuracy_score(y_test, y_pred)),
        'precision':   float(precision_score(y_test, y_pred, zero_division=0)),
        'recall':      float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score':    float(f1_score(y_test, y_pred, zero_division=0)),
        'sensitivity': float(recall_score(y_test, y_pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'confusion_matrix': {
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)
        }
    }

    joblib.dump(model,  os.path.join(MODEL_DIR, MODELOS[model_key]['model']))
    joblib.dump(scaler, os.path.join(MODEL_DIR, MODELOS[model_key]['scaler']))
    with open(os.path.join(MODEL_DIR, MODELOS[model_key]['metrics']), 'w') as f:
        json.dump(metrics, f)

    return model, scaler, metrics


def predict_batch(model, scaler, df_uploaded):
    X_batch, y_batch, valid_idx = preprocess_dataframe(df_uploaded)
    if len(X_batch) == 0:
        return pd.DataFrame()

    X_scaled      = scaler.transform(X_batch)
    predictions   = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    results = pd.DataFrame({
        'ID':                         df_uploaded.loc[valid_idx, 'ID'].values,
        'Status_Real':                df_uploaded.loc[valid_idx, 'Status'].values,
        'Predicción':                 ['⚠️ Muerte' if p == 1 else '✅ Supervivencia' for p in predictions],
        'Prob. Supervivencia':        np.round(probabilities[:, 0] * 100, 1),
        'Prob. Muerte':               np.round(probabilities[:, 1] * 100, 1),
    })
    return results


def render_metrics_panel(metrics, title="Métricas del Modelo"):
    """Renderiza un panel de métricas reutilizable."""
    st.markdown(f"#### {title}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy",    f"{metrics['accuracy']   *100:.2f}%")
    c2.metric("Precision",   f"{metrics['precision']  *100:.2f}%")
    c3.metric("Recall",      f"{metrics['recall']     *100:.2f}%")

    c4, c5, c6 = st.columns(3)
    c4.metric("F1-Score",    f"{metrics['f1_score']   *100:.2f}%")
    c5.metric("Sensitivity", f"{metrics['sensitivity']*100:.2f}%")
    c6.metric("Specificity", f"{metrics['specificity']*100:.2f}%")

    st.markdown("##### Matriz de Confusión")
    cm_data = metrics['confusion_matrix']
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')
    cm_arr = np.array([[cm_data['TN'], cm_data['FP']], [cm_data['FN'], cm_data['TP']]])
    disp   = ConfusionMatrixDisplay(confusion_matrix=cm_arr, display_labels=['Supervivencia', 'Muerte'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title("", pad=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TP", cm_data['TP'])
    c2.metric("TN", cm_data['TN'])
    c3.metric("FP", cm_data['FP'])
    c4.metric("FN", cm_data['FN'])


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 8px 0 20px 0;">
            <div style="font-size:1.5rem; font-weight:700; letter-spacing:-0.5px;">🫀 PBC Predictor</div>
            <div style="font-size:0.75rem; opacity:0.5; margin-top:2px;">Cirrosis Biliar Primaria</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:1.2px; opacity:0.5; margin-bottom:12px;">Modelo Activo</div>', unsafe_allow_html=True)

        model_key = st.selectbox(
            "Seleccionar modelo",
            options=['logistic', 'nn'],
            format_func=lambda x: MODELOS[x]['name'],
            label_visibility='collapsed'
        )

        metrics = get_metrics(model_key)
        if metrics:
            st.markdown(f"""
            <div class="model-badge">{MODELOS[model_key]['name']}</div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="sidebar-metric">
                <div class="sm-val">{metrics['accuracy']*100:.1f}%</div>
                <div class="sm-lbl">Accuracy</div>
            </div>
            <div class="sidebar-metric">
                <div class="sm-val">{metrics['f1_score']*100:.1f}%</div>
                <div class="sm-lbl">F1-Score</div>
            </div>
            <div class="sidebar-metric">
                <div class="sm-val">{metrics['specificity']*100:.1f}%</div>
                <div class="sm-lbl">Specificity</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Sin métricas disponibles para este modelo.")

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.7rem; opacity:0.4; line-height:1.6;">
            Predictor de supervivencia para pacientes con cirrosis biliar primaria (PBC).<br><br>
            Basado en el dataset de la Clínica Mayo (1974–1984).
        </div>
        """, unsafe_allow_html=True)

    return model_key


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    model_key = render_sidebar()

    # Header
    st.markdown("""
    <div class="main-header">
        <div class="header-badge">Diagnóstico Predictivo</div>
        <h1>Predictor de Supervivencia</h1>
        <p>Cirrosis Biliar Primaria &nbsp;·&nbsp; Modelos de Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["👤  Predicción Individual", "📊  Predicción por Lote"])

    # ──────────────────────────────────────────
    #  TAB 1 · PREDICCIÓN INDIVIDUAL
    # ──────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Datos Demográficos y Síntomas</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Datos del Paciente**")
            age  = st.number_input("Edad (años)",   min_value=18, max_value=120, value=50)
            sex  = st.selectbox("Sexo", options=[0, 1],
                                format_func=lambda x: "Femenino" if x == 0 else "Masculino")
            drug = st.selectbox("Fármaco", options=[0, 1],
                                format_func=lambda x: "Placebo" if x == 0 else "D-penicilamina")
            stage = st.selectbox("Estadío de la Enfermedad", options=[1, 2, 3, 4],
                                 format_func=lambda x: f"Estadío {x}")

        with col2:
            st.markdown("**Síntomas Clínicos**")
            ascites     = st.selectbox("Ascitis",           options=[0, 0.5, 1],
                                       format_func=lambda x: {0: "No", 0.5: "Leve", 1: "Sí"}[x])
            hepatomegaly = st.selectbox("Hepatomegalia",    options=[0, 1],
                                        format_func=lambda x: "No" if x == 0 else "Sí")
            spiders     = st.selectbox("Arañas Vasculares", options=[0, 1],
                                       format_func=lambda x: "No" if x == 0 else "Sí")
            edema       = st.selectbox("Edema",             options=[0, 0.5, 1],
                                       format_func=lambda x: {0: "No", 0.5: "Leve", 1: "Sí"}[x])

        with col3:
            st.markdown("**Análisis de Laboratorio**")
            bilirubin    = st.number_input("Bilirrubina (mg/dL)",         min_value=0.0, value=1.0,   step=0.1)
            cholesterol  = st.number_input("Colesterol (mg/dL)",          min_value=0.0, value=200.0, step=1.0)
            albumin      = st.number_input("Albúmina (g/dL)",             min_value=0.0, value=3.5,   step=0.1)
            copper       = st.number_input("Cobre (μg/L)",                min_value=0.0, value=50.0,  step=1.0)
            alk_phos     = st.number_input("Fosfatasa Alcalina (U/L)",    min_value=0.0, value=100.0, step=1.0)
            sgot         = st.number_input("SGOT (U/L)",                  min_value=0.0, value=80.0,  step=0.1)
            tryglicerides = st.number_input("Triglicéridos (mg/dL)",      min_value=0.0, value=120.0, step=1.0)
            platelets    = st.number_input("Plaquetas (×10⁹/L)",          min_value=0.0, value=150.0, step=1.0)
            prothrombin  = st.number_input("Tiempo de Protrombina (s)",   min_value=0.0, value=10.0,  step=0.1)

        st.markdown('</div>', unsafe_allow_html=True)

        predict_btn = st.button("🔮  Predecir Supervivencia", type="primary", use_container_width=False)

        if predict_btn:
            model, scaler = load_model(model_key)
            if model is None:
                st.error("❌ No se encontraron los archivos del modelo. Verifica que los `.pkl` estén en el mismo directorio.")
            else:
                features = {
                    'Age': age, 'Sex': sex, 'Ascites': ascites, 'Hepatomegaly': hepatomegaly,
                    'Spiders': spiders, 'Edema': edema, 'Bilirubin': bilirubin,
                    'Cholesterol': cholesterol, 'Albumin': albumin, 'Copper': copper,
                    'Alk_Phos': alk_phos, 'SGOT': sgot, 'Tryglicerides': tryglicerides,
                    'Platelets': platelets, 'Prothrombin': prothrombin, 'Stage': stage, 'Drug': drug
                }
                try:
                    X          = np.array([[features[f] for f in FEATURES]])
                    X_scaled   = scaler.transform(X)
                    prediction = model.predict(X_scaled)[0]
                    proba      = model.predict_proba(X_scaled)[0]

                    surv_pct = proba[0] * 100
                    mort_pct = proba[1] * 100

                    if prediction == 1:
                        card_class   = "result-card-high"
                        label_text   = "🔴 ALTO RIESGO"
                        description  = "El modelo predice mayor probabilidad de fallecimiento"
                    else:
                        card_class   = "result-card-low"
                        label_text   = "🟢 BAJO RIESGO"
                        description  = "El modelo predice mayor probabilidad de supervivencia"

                    st.markdown(f"""
                    <div class="{card_class}">
                        <div class="result-label">{label_text}</div>
                        <div class="result-desc">{description}</div>
                        <div class="prob-row">
                            <div class="prob-box">
                                <div class="val">{surv_pct:.1f}%</div>
                                <div class="lbl">Supervivencia</div>
                            </div>
                            <div class="prob-box">
                                <div class="val">{mort_pct:.1f}%</div>
                                <div class="lbl">Muerte</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Barra de probabilidad de supervivencia
                    st.markdown("**Probabilidad de supervivencia**")
                    st.progress(surv_pct / 100)

                except Exception as e:
                    st.error(f"❌ Error en la predicción: {str(e)}")

        # Expander de métricas
        metrics = get_metrics(model_key)
        if metrics:
            with st.expander("📊 Ver métricas del modelo seleccionado"):
                render_metrics_panel(metrics)
        else:
            st.info("ℹ️ No hay métricas guardadas para este modelo aún.")

    # ──────────────────────────────────────────
    #  TAB 2 · PREDICCIÓN POR LOTE
    # ──────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📂 Carga de Dataset</div>', unsafe_allow_html=True)

        st.info(
            "**Requisitos del CSV:**  "
            "Exactamente 20 columnas: `" + "`, `".join(EXPECTED_COLUMNS) + "`\n\n"
            "- **Status** → `D` (muerte) · `C` (supervivencia) · `CL` (censurado)\n"
            "- **Sex** → `M` · `F`\n"
            "- **Drug** → `D-penicillamine` · `Placebo`\n"
            "- Los IDs se generan automáticamente · El modelo seleccionado se reentrenará con los nuevos datos"
        )

        csv_template = pd.DataFrame(columns=EXPECTED_COLUMNS)
        st.download_button(
            label="📥 Descargar plantilla CSV",
            data=csv_template.to_csv(index=False).encode('utf-8'),
            file_name="template_cirrosis.csv",
            mime="text/csv"
        )

        st.markdown('</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Sube tu CSV aquí", type=['csv'])

        if uploaded_file:
            df_uploaded, is_valid, missing, extra, error_msg = validate_csv(uploaded_file)

            if not is_valid:
                if missing:
                    st.error(f"❌ Columnas faltantes: `{'`, `'.join(missing)}`")
                if extra:
                    st.warning(f"⚠️ Columnas extra ignoradas: `{'`, `'.join(extra)}`")
                if error_msg:
                    st.error(f"❌ {error_msg}")
            else:
                st.success(f"✅ CSV válido · **{len(df_uploaded)} registros** listos para procesar")
                st.dataframe(df_uploaded.head(5), use_container_width=True)

                if st.button("🔄 Procesar Lote", type="primary"):
                    try:
                        original_path = os.path.join(MODEL_DIR, 'cirrhosis.csv')
                        updated_path  = os.path.join(MODEL_DIR, 'cirrhosis_updated.csv')

                        if not os.path.exists(original_path):
                            st.error("❌ No se encontró `cirrhosis.csv` en el directorio de la app.")
                            st.stop()

                        df_original = pd.read_csv(original_path)

                        with st.status("Procesando lote…", expanded=True) as status:
                            st.write("📌 Generando nuevos IDs…")
                            df_with_ids = generate_new_ids(df_uploaded, df_original)

                            st.write("🔗 Combinando con base de datos original…")
                            df_combined = pd.concat([df_original, df_with_ids], ignore_index=True)
                            df_combined.to_csv(updated_path, index=False)

                            st.write(f"🧠 Reentrenando modelo: **{MODELOS[model_key]['name']}**…")
                            model, scaler, metrics = retrain_model(model_key, df_combined)

                            st.write("🔮 Generando predicciones…")
                            clear_model_cache()
                            results = predict_batch(model, scaler, df_with_ids)

                            status.update(label="✅ Lote procesado correctamente", state="complete")

                        if len(results) > 0:
                            st.markdown("---")
                            st.markdown("#### Resultados de Predicción")
                            st.dataframe(results, use_container_width=True)

                            st.download_button(
                                label="📥 Descargar predicciones",
                                data=results.to_csv(index=False).encode('utf-8'),
                                file_name="predicciones_lote.csv",
                                mime="text/csv"
                            )

                            st.markdown("---")
                            render_metrics_panel(metrics, title="Métricas del Modelo Reentrenado")

                            st.info(
                                f"**Archivos actualizados:**\n"
                                f"- `cirrhosis_updated.csv` — base de datos combinada ({len(df_combined)} registros)\n"
                                f"- `{MODELOS[model_key]['model']}` — modelo reentrenado\n"
                                f"- Nuevo accuracy: **{metrics['accuracy']*100:.2f}%**"
                            )
                        else:
                            st.warning("⚠️ No se generaron predicciones. Todos los registros tenían valores faltantes.")

                    except Exception as e:
                        st.error(f"❌ Error al procesar el lote: {str(e)}")
                        st.error("La base de datos original no fue modificada.")


if __name__ == "__main__":
    main()
