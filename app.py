import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# ══════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PBC Survival Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
#  PALETA OSCURA
# ══════════════════════════════════════════════════════════
DARK = {
    'bg':        '#0d1117',
    'surface':   '#161b22',
    'surface2':  '#1c2128',
    'border':    '#30363d',
    'border2':   '#21262d',
    'text':      '#e6edf3',
    'text2':     '#8b949e',
    'accent':    '#58a6ff',
    'accent2':   '#388bfd',
    'green':     '#3fb950',
    'green_bg':  '#0f2a1a',
    'green_bdr': '#238636',
    'red':       '#f85149',
    'red_bg':    '#2a0f0f',
    'red_bdr':   '#6e1c1c',
    'yellow':    '#d29922',
    'purple':    '#bc8cff',
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
    color: {DARK['text']} !important;
}}
.stApp, .main, .block-container {{
    background: {DARK['bg']} !important;
}}
.block-container {{
    padding-top: 1.5rem !important;
    max-width: 1280px;
}}

[data-testid="stSidebar"] {{
    background: {DARK['surface']} !important;
    border-right: 1px solid {DARK['border']} !important;
}}
[data-testid="stSidebar"] * {{ color: {DARK['text']} !important; }}
[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: {DARK['surface2']} !important;
    border: 1px solid {DARK['border']} !important;
    border-radius: 8px !important;
}}

.app-header {{
    background: linear-gradient(135deg, #0d1f3c 0%, #0f2a52 60%, #0b1e3a 100%);
    border: 1px solid {DARK['border']};
    border-radius: 14px;
    padding: 28px 36px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}}
.app-header::after {{
    content: '🫀';
    position: absolute;
    right: 40px; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.06;
}}
.app-header .badge {{
    display: inline-block;
    background: rgba(88,166,255,0.12);
    border: 1px solid rgba(88,166,255,0.25);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: {DARK['accent']};
    margin-bottom: 10px;
}}
.app-header h1 {{
    font-size: 1.8rem;
    font-weight: 700;
    color: {DARK['text']} !important;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}}
.app-header p {{
    color: {DARK['text2']} !important;
    font-size: 0.88rem;
    margin: 0;
}}

.card {{
    background: {DARK['surface']};
    border: 1px solid {DARK['border']};
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 18px;
}}
.card-title {{
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: {DARK['text2']};
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid {DARK['border2']};
}}

[data-testid="stMetric"] {{
    background: {DARK['surface2']} !important;
    border: 1px solid {DARK['border']} !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
}}
[data-testid="stMetric"] label {{
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: {DARK['text2']} !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.3rem !important;
    color: {DARK['text']} !important;
}}
[data-testid="stMetricDelta"] {{
    font-family: 'JetBrains Mono', monospace !important;
}}

.pred-high {{
    background: {DARK['red_bg']};
    border: 1px solid {DARK['red_bdr']};
    border-left: 4px solid {DARK['red']};
    border-radius: 12px;
    padding: 22px 26px;
    margin: 16px 0;
}}
.pred-low {{
    background: {DARK['green_bg']};
    border: 1px solid {DARK['green_bdr']};
    border-left: 4px solid {DARK['green']};
    border-radius: 12px;
    padding: 22px 26px;
    margin: 16px 0;
}}
.pred-label {{
    font-size: 1.35rem;
    font-weight: 700;
    margin-bottom: 4px;
}}
.pred-desc {{
    font-size: 0.85rem;
    color: {DARK['text2']};
    margin-bottom: 16px;
}}
.prob-row {{ display: flex; gap: 14px; flex-wrap: wrap; }}
.prob-box {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 12px 20px;
    min-width: 120px;
    text-align: center;
}}
.prob-val {{
    font-size: 1.35rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}}
.prob-lbl {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: {DARK['text2']};
    margin-top: 3px;
}}

.pred-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.86rem;
    background: {DARK['surface']};
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid {DARK['border']};
}}
.pred-table thead tr {{
    background: {DARK['surface2']};
    border-bottom: 2px solid {DARK['border']};
}}
.pred-table thead th {{
    padding: 12px 16px;
    text-align: left;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {DARK['text2']};
}}
.pred-table tbody tr {{ border-bottom: 1px solid {DARK['border2']}; }}
.pred-table tbody tr:hover {{ background: {DARK['surface2']}; }}
.pred-table tbody td {{ padding: 11px 16px; color: {DARK['text']}; vertical-align: middle; }}
.col-id {{ font-family: 'JetBrains Mono', monospace; color: {DARK['text2']}; font-size: 0.78rem; }}
.col-prob {{ font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }}
.badge-death {{
    display: inline-flex; align-items: center; gap: 5px;
    background: {DARK['red_bg']}; border: 1px solid {DARK['red_bdr']};
    color: {DARK['red']}; border-radius: 20px;
    padding: 3px 11px; font-size: 0.76rem; font-weight: 600;
}}
.badge-surv {{
    display: inline-flex; align-items: center; gap: 5px;
    background: {DARK['green_bg']}; border: 1px solid {DARK['green_bdr']};
    color: {DARK['green']}; border-radius: 20px;
    padding: 3px 11px; font-size: 0.76rem; font-weight: 600;
}}
.prob-bar-wrap {{ display: flex; align-items: center; gap: 10px; }}
.prob-bar-bg {{
    flex: 1; height: 6px;
    background: {DARK['surface2']}; border-radius: 3px; overflow: hidden; min-width: 70px;
}}
.prob-bar-fill-d {{ height: 100%; background: {DARK['red']}; border-radius: 3px; }}
.prob-bar-fill-s {{ height: 100%; background: {DARK['green']}; border-radius: 3px; }}

.sm-box {{
    background: {DARK['surface2']};
    border: 1px solid {DARK['border']};
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    text-align: center;
}}
.sm-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem; font-weight: 600; color: {DARK['accent']};
}}
.sm-lbl {{
    font-size: 0.62rem; text-transform: uppercase;
    letter-spacing: 1px; color: {DARK['text2']}; margin-top: 2px;
}}
.sidebar-sep {{
    border: none; border-top: 1px solid {DARK['border']}; margin: 14px 0;
}}
.model-tag {{
    display: inline-block;
    background: rgba(88,166,255,0.1); border: 1px solid rgba(88,166,255,0.2);
    color: {DARK['accent']}; border-radius: 20px;
    padding: 2px 10px; font-size: 0.62rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px;
}}

.stNumberInput > div > div > input {{
    background: {DARK['surface2']} !important;
    border: 1px solid {DARK['border']} !important;
    border-radius: 8px !important;
    color: {DARK['text']} !important;
}}
.stSelectbox > div > div {{
    background: {DARK['surface2']} !important;
    border: 1px solid {DARK['border']} !important;
    border-radius: 8px !important;
    color: {DARK['text']} !important;
}}
[data-baseweb="popover"] > div {{
    background: {DARK['surface2']} !important;
    border: 1px solid {DARK['border']} !important;
}}
li[role="option"] {{
    background: {DARK['surface2']} !important;
    color: {DARK['text']} !important;
}}
li[role="option"]:hover {{ background: {DARK['surface']} !important; }}

.stButton > button[kind="primary"] {{
    background: {DARK['accent2']} !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: #fff !important;
}}
.stButton > button[kind="primary"]:hover {{ opacity: 0.85 !important; }}
.stButton > button:not([kind="primary"]) {{
    background: {DARK['surface2']} !important;
    border: 1px solid {DARK['border']} !important;
    border-radius: 8px !important;
    color: {DARK['text']} !important;
}}
.stButton > button:not([kind="primary"]):hover {{ border-color: {DARK['accent']} !important; }}

.stAlert {{
    border-radius: 10px !important;
    background: {DARK['surface2']} !important;
}}
.stProgress > div > div > div {{
    background: {DARK['accent']} !important; border-radius: 4px !important;
}}
.stProgress > div > div {{
    background: {DARK['surface2']} !important; border-radius: 4px !important;
}}
[data-testid="stFileUploader"] {{
    background: {DARK['surface2']} !important;
    border: 1px dashed {DARK['border']} !important;
    border-radius: 10px !important;
}}
.stDownloadButton > button {{
    background: {DARK['surface2']} !important;
    border: 1px solid {DARK['border']} !important;
    border-radius: 8px !important;
    color: {DARK['text']} !important;
    font-weight: 500 !important;
}}
.stDownloadButton > button:hover {{ border-color: {DARK['accent']} !important; }}
.stMarkdown p, .stMarkdown li {{ color: {DARK['text']} !important; }}
label {{ color: {DARK['text2']} !important; font-size: 0.82rem !important; font-weight: 500 !important; }}
.nav-label {{
    font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 1.2px; color: {DARK['text2']}; margin-bottom: 6px;
}}
</style>
"""

# ══════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════
MODEL_DIR = os.path.dirname(os.path.realpath(__file__))

MODELOS = {
    'logistic': {
        'model':   'cirrhosis_logistic_model.pkl',
        'scaler':  'cirrhosis_logistic_scaler.pkl',
        'metrics': 'cirrhosis_logistic_metrics.json',
        'name':    'Regresión Logística',
    },
    'nn': {
        'model':   'cirrhosis_nn_model.pkl',
        'scaler':  'cirrhosis_nn_scaler.pkl',
        'metrics': 'cirrhosis_nn_metrics.json',
        'name':    'Red Neuronal (MLP)',
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

EXAMPLE_VALUES = {
    'age': 58, 'sex': 0, 'drug': 1, 'stage': 3,
    'ascites': 1.0, 'hepatomegaly': 1, 'spiders': 1, 'edema': 0.5,
    'bilirubin': 6.2, 'cholesterol': 230.0, 'albumin': 2.8,
    'copper': 110.0, 'alk_phos': 1250.0, 'sgot': 98.5,
    'tryglicerides': 115.0, 'platelets': 160.0, 'prothrombin': 11.8
}

# ══════════════════════════════════════════════════════════
#  HELPERS — MODELO
# ══════════════════════════════════════════════════════════
@st.cache_resource
def load_model(model_key):
    mp = os.path.join(MODEL_DIR, MODELOS[model_key]['model'])
    sp = os.path.join(MODEL_DIR, MODELOS[model_key]['scaler'])
    if not os.path.exists(mp) or not os.path.exists(sp):
        return None, None
    return joblib.load(mp), joblib.load(sp)


def clear_model_cache():
    st.cache_resource.clear()


def get_metrics(model_key):
    try:
        path = os.path.join(MODEL_DIR, MODELOS[model_key]['metrics'])
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def save_metrics(model_key, metrics):
    path = os.path.join(MODEL_DIR, MODELOS[model_key]['metrics'])
    with open(path, 'w') as f:
        json.dump(metrics, f)


# ══════════════════════════════════════════════════════════
#  HELPERS — DATOS
# ══════════════════════════════════════════════════════════
def validate_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra   = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing or extra:
        return df, False, missing, extra, None
    inv_status = [s for s in df['Status'].unique() if s not in {'D', 'C', 'CL'}]
    if inv_status:
        return df, False, [], [], f"Valores inválidos en Status: {inv_status}"
    inv_sex = [s for s in df['Sex'].unique() if s not in {'M', 'F'}]
    if inv_sex:
        return df, False, [], [], f"Valores inválidos en Sex: {inv_sex}"
    inv_drug = [d for d in df['Drug'].dropna().unique() if d not in {'D-penicillamine', 'Placebo'}]
    if inv_drug:
        return df, False, [], [], f"Valores inválidos en Drug: {inv_drug}"
    return df, True, [], [], None


def generate_new_ids(df_uploaded, df_base):
    max_id = df_base['ID'].max()
    df_copy = df_uploaded.copy()
    df_copy['ID'] = list(range(int(max_id) + 1, int(max_id) + 1 + len(df_uploaded)))
    return df_copy


def preprocess_dataframe(df):
    data = df.copy()
    data['Sex']  = data['Sex'].map({'M': 1, 'F': 0})
    data['Drug'] = data['Drug'].map({'D-penicillamine': 1, 'Placebo': 0})
    for var in ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema']:
        data[var] = data[var].map({'Y': 1, 'N': 0, 'S': 0.5})
    data['Status'] = data['Status'].map({'D': 1, 'C': 0, 'CL': 0})
    feat_cols = [c for c in data.columns if c not in ['ID', 'N_Days', 'Status']]
    X = data[feat_cols]
    y = data['Status']
    valid = ~(X.isnull().any(axis=1) | y.isnull())
    return X[valid], y[valid], valid


def compute_metrics_from_model(model, scaler, df):
    X, y, _ = preprocess_dataframe(df)
    if len(X) == 0:
        return None
    X_s  = scaler.transform(X)
    pred = model.predict(X_s)
    cm   = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy':    float(accuracy_score(y, pred)),
        'precision':   float(precision_score(y, pred, zero_division=0)),
        'recall':      float(recall_score(y, pred, zero_division=0)),
        'f1_score':    float(f1_score(y, pred, zero_division=0)),
        'sensitivity': float(recall_score(y, pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'confusion_matrix': {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)},
        'n_samples': int(len(y)),
    }


def retrain_model(model_key, df_combined):
    X, y, _ = preprocess_dataframe(df_combined)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    if model_key == 'logistic':
        model = LogisticRegression(max_iter=500, random_state=42)
    else:
        model = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.01,
                               activation='tanh', max_iter=500, random_state=42)
    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'accuracy':    float(accuracy_score(y_te, y_pred)),
        'precision':   float(precision_score(y_te, y_pred, zero_division=0)),
        'recall':      float(recall_score(y_te, y_pred, zero_division=0)),
        'f1_score':    float(f1_score(y_te, y_pred, zero_division=0)),
        'sensitivity': float(recall_score(y_te, y_pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'confusion_matrix': {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)},
        'n_samples': int(len(y)),
    }
    joblib.dump(model,  os.path.join(MODEL_DIR, MODELOS[model_key]['model']))
    joblib.dump(scaler, os.path.join(MODEL_DIR, MODELOS[model_key]['scaler']))
    save_metrics(model_key, metrics)
    return model, scaler, metrics


def predict_batch(model, scaler, df):
    X, y, valid = preprocess_dataframe(df)
    if len(X) == 0:
        return pd.DataFrame()
    X_s   = scaler.transform(X)
    preds = model.predict(X_s)
    proba = model.predict_proba(X_s)
    return pd.DataFrame({
        'ID':               df.loc[valid, 'ID'].values,
        'Status_Real':      df.loc[valid, 'Status'].values,
        '_pred_int':        preds,
        'Prob_Supervivencia': np.round(proba[:, 0] * 100, 1),
        'Prob_Muerte':      np.round(proba[:, 1] * 100, 1),
    })


# ══════════════════════════════════════════════════════════
#  VISUALIZACIÓN
# ══════════════════════════════════════════════════════════
def dark_cm_figure(cm_data, title=""):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')
    cm_arr = np.array([[cm_data['TN'], cm_data['FP']],
                        [cm_data['FN'], cm_data['TP']]])
    ax.imshow(cm_arr, cmap='Blues', aspect='auto', vmin=0)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Supervivencia', 'Muerte'], color='#8b949e', fontsize=9)
    ax.set_yticklabels(['Supervivencia', 'Muerte'], color='#8b949e', fontsize=9,
                        rotation=90, va='center')
    ax.set_xlabel('Predicho', color='#8b949e', fontsize=9, labelpad=8)
    ax.set_ylabel('Real',     color='#8b949e', fontsize=9, labelpad=8)
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    thresh = cm_arr.max() / 2
    for i in range(2):
        for j in range(2):
            color = 'white' if cm_arr[i, j] > thresh else '#8b949e'
            ax.text(j, i, str(cm_arr[i, j]),
                    ha='center', va='center', fontsize=20,
                    fontweight='bold', color=color, fontfamily='monospace')
    if title:
        ax.set_title(title, color='#e6edf3', fontsize=10, fontweight='600', pad=12)
    plt.tight_layout()
    return fig


def render_metrics_grid(metrics):
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy",     f"{metrics['accuracy']   *100:.2f}%")
    c2.metric("Precision",    f"{metrics['precision']  *100:.2f}%")
    c3.metric("Recall",       f"{metrics['recall']     *100:.2f}%")
    c4, c5, c6 = st.columns(3)
    c4.metric("F1-Score",     f"{metrics['f1_score']   *100:.2f}%")
    c5.metric("Sensibilidad", f"{metrics['sensitivity']*100:.2f}%")
    c6.metric("Especificidad",f"{metrics['specificity']*100:.2f}%")


def render_cm_panel(metrics, title="Matriz de Confusión Global"):
    st.markdown(f"<div class='card-title'>{title}</div>", unsafe_allow_html=True)
    col_cm, col_info = st.columns([2, 1])
    cm = metrics['confusion_matrix']
    n  = metrics.get('n_samples', cm['TP']+cm['TN']+cm['FP']+cm['FN'])
    with col_cm:
        fig = dark_cm_figure(cm)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    with col_info:
        st.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:6px;">
            <div style="background:#0f2a1a; border:1px solid #238636; border-radius:10px;
                        padding:14px; text-align:center;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.6rem;
                             font-weight:700; color:#3fb950;">{cm['TP']}</div>
                <div style="font-size:0.62rem; text-transform:uppercase; letter-spacing:1px;
                             color:#8b949e; margin-top:2px;">Verdadero +</div>
            </div>
            <div style="background:#1a2a3a; border:1px solid #1f6feb; border-radius:10px;
                        padding:14px; text-align:center;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.6rem;
                             font-weight:700; color:#58a6ff;">{cm['TN']}</div>
                <div style="font-size:0.62rem; text-transform:uppercase; letter-spacing:1px;
                             color:#8b949e; margin-top:2px;">Verdadero −</div>
            </div>
            <div style="background:#2a1a1a; border:1px solid #6e1c1c; border-radius:10px;
                        padding:14px; text-align:center;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.6rem;
                             font-weight:700; color:#f85149;">{cm['FP']}</div>
                <div style="font-size:0.62rem; text-transform:uppercase; letter-spacing:1px;
                             color:#8b949e; margin-top:2px;">Falso +</div>
            </div>
            <div style="background:#2a1e0a; border:1px solid #7d4e17; border-radius:10px;
                        padding:14px; text-align:center;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.6rem;
                             font-weight:700; color:#d29922;">{cm['FN']}</div>
                <div style="font-size:0.62rem; text-transform:uppercase; letter-spacing:1px;
                             color:#8b949e; margin-top:2px;">Falso −</div>
            </div>
        </div>
        <div style="margin-top:12px; padding:10px 14px; background:#1c2128;
                    border:1px solid #30363d; border-radius:8px;
                    font-size:0.78rem; color:#8b949e; text-align:center;">
            {n:,} muestras evaluadas
        </div>
        """, unsafe_allow_html=True)


def render_batch_table(results_df):
    rows = ""
    for _, row in results_df.iterrows():
        pred = int(row['_pred_int'])
        ps   = row['Prob_Supervivencia']
        pm   = row['Prob_Muerte']
        if pred == 1:
            badge = '<span class="badge-death">💀 Muerte</span>'
            pct   = pm
            bar   = f'<div class="prob-bar-fill-d" style="width:{min(pct,100):.0f}%"></div>'
        else:
            badge = '<span class="badge-surv">✅ Supervivencia</span>'
            pct   = ps
            bar   = f'<div class="prob-bar-fill-s" style="width:{min(pct,100):.0f}%"></div>'

        rows += f"""
        <tr>
            <td class="col-id">#{int(row['ID'])}</td>
            <td style="text-align:center; color:#8b949e;">{row['Status_Real']}</td>
            <td>{badge}</td>
            <td>
                <div class="prob-bar-wrap">
                    <div class="prob-bar-bg"><div>{bar}</div></div>
                    <span class="col-prob" style="min-width:40px;">{pct:.1f}%</span>
                </div>
            </td>
            <td class="col-prob" style="color:#3fb950;">{ps:.1f}%</td>
            <td class="col-prob" style="color:#f85149;">{pm:.1f}%</td>
        </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto; border-radius:12px;">
    <table class="pred-table">
        <thead><tr>
            <th>ID</th><th>Real</th><th>Predicción</th>
            <th style="min-width:160px;">Confianza</th>
            <th>% Superv.</th><th>% Muerte</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:10px 0 16px 0;">
            <div style="font-size:1.3rem; font-weight:700; color:{DARK['text']};">🫀 PBC Predictor</div>
            <div style="font-size:0.7rem; color:{DARK['text2']}; margin-top:2px;">Cirrosis Biliar Primaria</div>
        </div>
        <hr class="sidebar-sep">
        <div style="font-size:0.62rem; text-transform:uppercase; letter-spacing:1.2px;
                    color:{DARK['text2']}; margin-bottom:8px;">Modelo activo</div>
        """, unsafe_allow_html=True)

        model_key = st.selectbox(
            "Modelo", options=['logistic', 'nn'],
            format_func=lambda x: MODELOS[x]['name'],
            label_visibility='collapsed'
        )

        metrics = get_metrics(model_key)
        if metrics:
            st.markdown(f'<div class="model-tag">{MODELOS[model_key]["name"]}</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="sm-box">
                <div class="sm-val">{metrics['accuracy']*100:.1f}%</div>
                <div class="sm-lbl">Accuracy</div>
            </div>
            <div class="sm-box">
                <div class="sm-val">{metrics['f1_score']*100:.1f}%</div>
                <div class="sm-lbl">F1-Score</div>
            </div>
            <div class="sm-box">
                <div class="sm-val">{metrics['specificity']*100:.1f}%</div>
                <div class="sm-lbl">Especificidad</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Sin métricas guardadas aún.")

        st.markdown(f"""
        <hr class="sidebar-sep">
        <div style="font-size:0.66rem; color:{DARK['text2']}; line-height:1.8;">
            Dataset: Clínica Mayo<br>
            Período: 1974 – 1984<br>
            Variables: 17 predictoras
        </div>
        """, unsafe_allow_html=True)

    return model_key


# ══════════════════════════════════════════════════════════
#  SECCIÓN INDIVIDUAL
# ══════════════════════════════════════════════════════════
def section_individual(model_key):
    # Panel de métricas
    metrics = get_metrics(model_key)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Métricas del Modelo Actual</div>', unsafe_allow_html=True)
    if metrics:
        render_metrics_grid(metrics)
    else:
        st.info("ℹ️ No hay métricas guardadas para este modelo.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Formulario
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📋 Datos del Paciente</div>', unsafe_allow_html=True)

    use_example = st.button("🧪  Rellenar con datos de ejemplo", type="secondary")

    def v(key, default):
        return EXAMPLE_VALUES[key] if use_example else default

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<div style='color:{DARK['text2']};font-size:0.72rem;font-weight:600;margin-bottom:8px;'>DATOS GENERALES</div>", unsafe_allow_html=True)
        age   = st.number_input("Edad (años)",  min_value=18, max_value=120, value=v('age', 50))
        sex   = st.selectbox("Sexo", options=[0,1], index=v('sex',0),
                              format_func=lambda x: "Femenino" if x==0 else "Masculino")
        drug  = st.selectbox("Fármaco", options=[0,1], index=v('drug',0),
                              format_func=lambda x: "Placebo" if x==0 else "D-penicilamina")
        stage = st.selectbox("Estadío", options=[1,2,3,4], index=v('stage',1)-1,
                              format_func=lambda x: f"Estadío {x}")

    with col2:
        st.markdown(f"<div style='color:{DARK['text2']};font-size:0.72rem;font-weight:600;margin-bottom:8px;'>SÍNTOMAS CLÍNICOS</div>", unsafe_allow_html=True)
        OPT_YN  = [0, 1];  OPT_YNE = [0, 0.5, 1]
        LBL_YN  = lambda x: "No" if x==0 else "Sí"
        LBL_YNE = lambda x: {0:"No", 0.5:"Leve", 1:"Sí"}[x]

        ascites      = st.selectbox("Ascitis",           options=OPT_YNE, index=OPT_YNE.index(v('ascites',0)),      format_func=LBL_YNE)
        hepatomegaly = st.selectbox("Hepatomegalia",     options=OPT_YN,  index=OPT_YN.index(v('hepatomegaly',0)),  format_func=LBL_YN)
        spiders      = st.selectbox("Arañas Vasculares", options=OPT_YN,  index=OPT_YN.index(v('spiders',0)),       format_func=LBL_YN)
        edema        = st.selectbox("Edema",             options=OPT_YNE, index=OPT_YNE.index(v('edema',0)),        format_func=LBL_YNE)

    with col3:
        st.markdown(f"<div style='color:{DARK['text2']};font-size:0.72rem;font-weight:600;margin-bottom:8px;'>LABORATORIO</div>", unsafe_allow_html=True)
        bilirubin     = st.number_input("Bilirrubina (mg/dL)",      min_value=0.0, value=float(v('bilirubin',1.0)),      step=0.1)
        cholesterol   = st.number_input("Colesterol (mg/dL)",       min_value=0.0, value=float(v('cholesterol',200.0)),  step=1.0)
        albumin       = st.number_input("Albúmina (g/dL)",          min_value=0.0, value=float(v('albumin',3.5)),        step=0.1)
        copper        = st.number_input("Cobre (μg/L)",             min_value=0.0, value=float(v('copper',50.0)),        step=1.0)
        alk_phos      = st.number_input("Fosfatasa Alcalina (U/L)", min_value=0.0, value=float(v('alk_phos',100.0)),     step=1.0)
        sgot          = st.number_input("SGOT (U/L)",               min_value=0.0, value=float(v('sgot',80.0)),          step=0.1)
        tryglicerides = st.number_input("Triglicéridos (mg/dL)",    min_value=0.0, value=float(v('tryglicerides',120.0)),step=1.0)
        platelets     = st.number_input("Plaquetas (×10⁹/L)",       min_value=0.0, value=float(v('platelets',150.0)),    step=1.0)
        prothrombin   = st.number_input("Protrombina (s)",          min_value=0.0, value=float(v('prothrombin',10.0)),   step=0.1)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔮  Predecir Supervivencia", type="primary"):
        model, scaler = load_model(model_key)
        if model is None:
            st.error("❌ Archivos `.pkl` no encontrados.")
            return
        feats = {
            'Age':age,'Sex':sex,'Ascites':ascites,'Hepatomegaly':hepatomegaly,
            'Spiders':spiders,'Edema':edema,'Bilirubin':bilirubin,
            'Cholesterol':cholesterol,'Albumin':albumin,'Copper':copper,
            'Alk_Phos':alk_phos,'SGOT':sgot,'Tryglicerides':tryglicerides,
            'Platelets':platelets,'Prothrombin':prothrombin,'Stage':stage,'Drug':drug
        }
        try:
            X     = np.array([[feats[f] for f in FEATURES]])
            X_s   = scaler.transform(X)
            pred  = model.predict(X_s)[0]
            proba = model.predict_proba(X_s)[0]
            s_pct = proba[0]*100; m_pct = proba[1]*100

            cls  = "pred-high" if pred==1 else "pred-low"
            lbl  = "🔴 ALTO RIESGO — Muerte prevista" if pred==1 else "🟢 BAJO RIESGO — Supervivencia prevista"
            desc = ("El modelo predice alta probabilidad de fallecimiento en el período de seguimiento."
                    if pred==1 else
                    "El modelo predice alta probabilidad de supervivencia en el período de seguimiento.")
            gs   = "#3fb950" if pred==0 else "#f85149"
            gm   = "#f85149" if pred==1 else "#3fb950"

            st.markdown(f"""
            <div class="{cls}">
                <div class="pred-label">{lbl}</div>
                <div class="pred-desc">{desc}</div>
                <div class="prob-row">
                    <div class="prob-box">
                        <div class="prob-val" style="color:#3fb950;">{s_pct:.1f}%</div>
                        <div class="prob-lbl">Supervivencia</div>
                    </div>
                    <div class="prob-box">
                        <div class="prob-val" style="color:#f85149;">{m_pct:.1f}%</div>
                        <div class="prob-lbl">Muerte</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"<div style='color:#8b949e;font-size:0.75rem;margin-bottom:4px;'>Probabilidad de supervivencia</div>", unsafe_allow_html=True)
            st.progress(s_pct/100)

        except Exception as e:
            st.error(f"❌ Error: {e}")


# ══════════════════════════════════════════════════════════
#  SECCIÓN POR LOTES
# ══════════════════════════════════════════════════════════
def section_lotes(model_key):
    # Métricas
    metrics = get_metrics(model_key)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Métricas del Modelo Actual</div>', unsafe_allow_html=True)
    if metrics:
        render_metrics_grid(metrics)
    else:
        st.info("ℹ️ No hay métricas para este modelo.")
    st.markdown('</div>', unsafe_allow_html=True)

    # CM global actual
    if metrics:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_cm_panel(metrics, "🔲 Matriz de Confusión Global Actual")
        st.markdown('</div>', unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📂 Cargar Dataset por Lotes</div>', unsafe_allow_html=True)
    st.info(
        "**Columnas requeridas (20):** `" + "` · `".join(EXPECTED_COLUMNS) + "`\n\n"
        "**Status:** `D` muerte · `C` supervivencia · `CL` censurado  |  "
        "**Sex:** `M` · `F`  |  **Drug:** `D-penicillamine` · `Placebo`"
    )
    csv_template = pd.DataFrame(columns=EXPECTED_COLUMNS)
    st.download_button("📥 Descargar plantilla CSV",
                       data=csv_template.to_csv(index=False).encode('utf-8'),
                       file_name="template_cirrosis.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Arrastra o selecciona tu CSV", type=['csv'])
    if not uploaded_file:
        return

    df_up, is_valid, missing, extra, err = validate_csv(uploaded_file)
    if not is_valid:
        if missing: st.error(f"❌ Columnas faltantes: `{'`, `'.join(missing)}`")
        if extra:   st.warning(f"⚠️ Columnas extra: `{'`, `'.join(extra)}`")
        if err:     st.error(f"❌ {err}")
        return

    st.success(f"✅ CSV válido — **{len(df_up):,} registros** cargados")
    with st.expander("👁 Vista previa (5 primeras filas)"):
        st.dataframe(df_up.head(5), use_container_width=True)

    if not st.button("🔄  Procesar Lote y Reentrenar", type="primary"):
        return

    try:
        original_path = os.path.join(MODEL_DIR, 'cirrhosis.csv')
        updated_path  = os.path.join(MODEL_DIR, 'cirrhosis_updated.csv')
        if not os.path.exists(original_path):
            st.error("❌ No se encontró `cirrhosis.csv`.")
            return

        df_original    = pd.read_csv(original_path)
        metrics_before = get_metrics(model_key)

        with st.status("Procesando…", expanded=True) as status:
            st.write("📌 Asignando nuevos IDs…")
            df_ids = generate_new_ids(df_up, df_original)
            st.write("🔗 Combinando datasets…")
            df_combined = pd.concat([df_original, df_ids], ignore_index=True)
            df_combined.to_csv(updated_path, index=False)
            st.write(f"🧠 Reentrenando **{MODELOS[model_key]['name']}**…")
            model, scaler, metrics_after = retrain_model(model_key, df_combined)
            st.write("🔮 Generando predicciones…")
            clear_model_cache()
            results = predict_batch(model, scaler, df_ids)
            status.update(label="✅ Procesado correctamente", state="complete")

        if len(results) == 0:
            st.warning("⚠️ Sin predicciones — registros con valores faltantes.")
            return

        # Tabla de predicciones
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔮 Resultados por Lote</div>', unsafe_allow_html=True)
        render_batch_table(results)
        export_df = results.drop(columns=['_pred_int']).copy()
        export_df.insert(2, 'Predicción', results['_pred_int'].map({1:'Muerte',0:'Supervivencia'}))
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        st.download_button("📥 Descargar predicciones CSV",
                           data=export_df.to_csv(index=False).encode('utf-8'),
                           file_name="predicciones_lote.csv", mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

        # CM de los datos del CSV cargado
        st.markdown('<div class="card">', unsafe_allow_html=True)
        cm_batch = compute_metrics_from_model(model, scaler, df_ids)
        if cm_batch:
            render_cm_panel(cm_batch, f"🔲 Matriz de Confusión — CSV Cargado ({len(df_ids)} registros)")
        st.markdown('</div>', unsafe_allow_html=True)

        # Comparación antes / después
        if metrics_before:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🔀 Comparación: Antes vs Después del Reentrenamiento</div>', unsafe_allow_html=True)
            col_b, col_sep, col_a = st.columns([5, 1, 5])
            cb = metrics_before['confusion_matrix']
            ca = metrics_after['confusion_matrix']

            with col_b:
                st.markdown(f"""<div style="text-align:center; background:{DARK['surface2']};
                    border:1px solid {DARK['border']}; border-radius:10px;
                    padding:8px 0 4px; margin-bottom:10px;">
                    <span style="font-size:0.65rem; font-weight:700; text-transform:uppercase;
                                 letter-spacing:1px; color:{DARK['text2']};">📦 Antes</span><br>
                    <span style="font-size:0.7rem; color:{DARK['text2']};">Dataset original</span>
                </div>""", unsafe_allow_html=True)
                fig_b = dark_cm_figure(cb)
                st.pyplot(fig_b, use_container_width=False); plt.close(fig_b)
                b1,b2,b3,b4 = st.columns(4)
                b1.metric("TP",cb['TP']); b2.metric("TN",cb['TN'])
                b3.metric("FP",cb['FP']); b4.metric("FN",cb['FN'])
                st.metric("Accuracy", f"{metrics_before['accuracy']*100:.2f}%")
                st.metric("F1-Score", f"{metrics_before['f1_score']*100:.2f}%")

            with col_sep:
                st.markdown(f"""<div style="display:flex;align-items:center;justify-content:center;
                    height:100%;padding-top:90px;">
                    <span style="font-size:2rem;color:{DARK['text2']};">→</span></div>""",
                    unsafe_allow_html=True)

            with col_a:
                st.markdown(f"""<div style="text-align:center; background:{DARK['green_bg']};
                    border:1px solid {DARK['green_bdr']}; border-radius:10px;
                    padding:8px 0 4px; margin-bottom:10px;">
                    <span style="font-size:0.65rem; font-weight:700; text-transform:uppercase;
                                 letter-spacing:1px; color:{DARK['green']};">✅ Después</span><br>
                    <span style="font-size:0.7rem; color:{DARK['text2']};">Dataset combinado</span>
                </div>""", unsafe_allow_html=True)
                fig_a = dark_cm_figure(ca)
                st.pyplot(fig_a, use_container_width=False); plt.close(fig_a)
                a1,a2,a3,a4 = st.columns(4)
                a1.metric("TP",ca['TP'],delta=ca['TP']-cb['TP'])
                a2.metric("TN",ca['TN'],delta=ca['TN']-cb['TN'])
                a3.metric("FP",ca['FP'],delta=ca['FP']-cb['FP'],delta_color="inverse")
                a4.metric("FN",ca['FN'],delta=ca['FN']-cb['FN'],delta_color="inverse")
                acc_d=(metrics_after['accuracy']-metrics_before['accuracy'])*100
                f1_d =(metrics_after['f1_score'] -metrics_before['f1_score'] )*100
                st.metric("Accuracy",f"{metrics_after['accuracy']*100:.2f}%",delta=f"{acc_d:+.2f}%")
                st.metric("F1-Score",f"{metrics_after['f1_score']*100:.2f}%", delta=f"{f1_d:+.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        # CM global actualizada
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_cm_panel(metrics_after, "🌐 Matriz de Confusión Global Actualizada")
        st.markdown('</div>', unsafe_allow_html=True)

        st.info(
            f"Dataset combinado: `cirrhosis_updated.csv` ({len(df_combined):,} registros) · "
            f"Nuevo accuracy: **{metrics_after['accuracy']*100:.2f}%**"
        )

    except Exception as e:
        st.error(f"❌ Error al procesar el lote: {e}")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    model_key = render_sidebar()

    st.markdown("""
    <div class="app-header">
        <div class="badge">DIAGNÓSTICO PREDICTIVO</div>
        <h1>PBC Survival Predictor</h1>
        <p>Cirrosis Biliar Primaria &nbsp;·&nbsp; Clínica Mayo 1974–1984 &nbsp;·&nbsp; Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='nav-label'>Sección</div>", unsafe_allow_html=True)
    seccion = st.selectbox(
        "Sección",
        options=["👤  Predicción Individual", "📊  Predicción por Lotes"],
        label_visibility='collapsed'
    )
    st.markdown("<div style='margin-bottom:18px;'></div>", unsafe_allow_html=True)

    if "Individual" in seccion:
        section_individual(model_key)
    else:
        section_lotes(model_key)


if __name__ == "__main__":
    main()
