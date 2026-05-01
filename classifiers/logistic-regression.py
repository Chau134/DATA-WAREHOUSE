import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

# ===== LOAD DATA =====
df = pd.read_csv("../Processed_Data/all_subjects_features.csv")

# ===== FEATURE ENGINEERING =====
df = df.sort_index()

df['Delta_diff'] = df['Power_Delta'].diff().fillna(0)
df['Theta_diff'] = df['Power_Theta'].diff().fillna(0)

df['Alpha_diff'] = df['Power_Alpha'].diff().fillna(0)
df['Beta_diff'] = df['Power_Beta'].diff().fillna(0)

df['Delta_Theta_ratio'] = df['Power_Delta'] / (df['Power_Theta'] + 1e-6)
df['Alpha_Beta_ratio'] = df['Power_Alpha'] / (df['Power_Beta'] + 1e-6)

df['Rolling_Mean_Delta'] = df['Power_Delta'].rolling(5, min_periods=1).mean()
df['Rolling_Mean_Theta'] = df['Power_Theta'].rolling(5, min_periods=1).mean()

# ===== SPLIT =====
X = df.drop(columns=['Label'])
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# ===== SCALE (BẮT BUỘC cho Logistic) =====
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===== SMOTE (vừa phải) =====
sm = SMOTE(sampling_strategy={1: 220}, k_neighbors=3, random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# ===== MODEL =====
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs'
)

model.fit(X_train, y_train)

# ===== PREDICT =====
y_pred = model.predict(X_test)

# ===== EVALUATE =====
print("\n===== LOGISTIC REGRESSION =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred, zero_division=0))

# ===== SAVE =====
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/logistic.pkl")
joblib.dump(scaler, "models/logistic_scaler.pkl")