import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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

# Train with RandomizedSearchCV
param_grid = {
    "hidden_layer_sizes": [(8, 16, 32), (64,), (128, 64), (50, 50)],
    "activation": ["relu", "tanh"],
    "alpha": [0.001, 0.01, 0.1],
    "learning_rate": ["constant", "adaptive"]
}

mlp = MLPClassifier(max_iter=15000, random_state=42)

grid_search = RandomizedSearchCV(
    mlp,
    param_grid,
    cv=3,
    n_jobs=-1,
    scoring="accuracy",
    n_iter=20,
    random_state=42,
    verbose=1
)

print("Training Neural Network...")
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Predict
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = recal

# Save model and scaler
joblib.dump(best_model, 'cirrhosis_nn_model.pkl')
joblib.dump(scaler, 'cirrhosis_nn_scaler.pkl')

# Save test data
test_data = {
    "X_test": X_test.values.tolist(),
    "y_test": y_test.values.tolist(),
    "y_pred": y_pred.tolist(),
    "feature_names": feature_cols
}
with open('cirrhosis_nn_test_data.json', 'w') as f:
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
    },
    "best_params": grid_search.best_params_,
    "best_cv_score": float(grid_search.best_score_)
}
with open('cirrhosis_nn_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n" + "="*60)
print("RED NEURONAL - MODELO DE CIRROSIS")
print("="*60)
print(f"\nAccuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1-Score: {f1*100:.2f}%")
print(f"Specificity: {specificity*100:.2f}%")
print(f"\nModels saved: cirrhosis_nn_model.pkl, cirrhosis_nn_scaler.pkl")
print("Metrics saved: cirrhosis_nn_metrics.json")
