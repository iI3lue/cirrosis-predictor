import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import joblib
import json
import os

# Load data
df = pd.read_csv('cirrhosis.csv')

# Create copy
data = df.copy()

# Encoding categorical variables
data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})
data['Drug'] = data['Drug'].map({'D-penicillamine': 1, 'Placebo': 0})

binary_vars = ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
for var in binary_vars:
    data[var] = data[var].map({'Y': 1, 'N': 0, 'S': 0.5})

data['Status'] = data['Status'].map({'D': 1, 'C': 0, 'CL': 0})

# Select features
feature_cols = [col for col in data.columns if col not in ['ID', 'N_Days', 'Status']]
X = data[feature_cols].copy()
y = data['Status'].copy()

# Handle missing values
valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
X = X[valid_idx]
y = y[valid_idx]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
logreg = LogisticRegression(max_iter=1000, random_state=42, verbose=1)
logreg.fit(X_train_scaled, y_train)

# Predictions
y_pred = logreg.predict(X_test_scaled)
y_pred_proba = logreg.predict_proba(X_test_scaled)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = recall

# Save model and scaler
joblib.dump(logreg, 'cirrhosis_logistic_model.pkl')
joblib.dump(scaler, 'cirrhosis_logistic_scaler.pkl')

# Save test data
test_data = {
    "X_test": X_test.values.tolist(),
    "y_test": y_test.values.tolist(),
    "y_pred": y_pred.tolist(),
    "feature_names": feature_cols
}
with open('cirrhosis_logistic_test_data.json', 'w') as f:
    json.dump(test_data, f, indent=2)

# Save metrics
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "sensitivity": float(sensitivity),
    "specificity": float(specificity),
    "confusion_matrix": {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn)
    }
}
with open('cirrhosis_logistic_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n" + "="*60)
print("REGRESION LOGISTICA - MODELO DE CIRROSIS")
print("="*60)
print(f"\nAccuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1-Score: {f1*100:.2f}%")
print(f"\nModels saved: cirrhosis_logistic_model.pkl, cirrhosis_logistic_scaler.pkl")
print("Metrics saved: cirrhosis_logistic_metrics.json")
