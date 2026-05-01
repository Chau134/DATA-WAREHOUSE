import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ===== LOAD DATA =====
df = pd.read_csv("../Processed_Data/all_subjects_features.csv")

# ===== FEATURE ENGINEERING (CẢI THIỆN MẠNH) =====
df = df.sort_index()

# Feature cũ
df['Delta_diff'] = df['Power_Delta'].diff().fillna(0)
df['Theta_diff'] = df['Power_Theta'].diff().fillna(0)

# 🔥 thêm feature mới
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

# ===== SCALE =====
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===== SMOTE (CỨU N1 NHƯNG KHÔNG OVERFIT) =====
sm = SMOTE(sampling_strategy={1: 300}, k_neighbors=3, random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# ===== CLASS WEIGHT (NHẸ) =====
class_counts = pd.Series(y_train).value_counts().to_dict()
total = len(y_train)

weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}

if 1 in weights:
    weights[1] *= 2.0  # nhẹ thôi (đã có SMOTE)

sample_weights = pd.Series(y_train).map(weights)

# ===== MODEL (TUNED) =====
model = XGBClassifier(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.5,
    min_child_weight=5,
    objective='multi:softprob',  # 🔥 quan trọng
    num_class=len(np.unique(y_train)),
    eval_metric='mlogloss',
    random_state=42
)

# ===== TRAIN =====
model.fit(X_train, y_train, sample_weight=sample_weights)

# ===== PREDICT =====
y_prob = model.predict_proba(X_test)
y_pred = np.argmax(y_prob, axis=1)

# ===== EVALUATE =====
print("\n===== XGBOOST IMPROVED =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nUnique predicted:", np.unique(y_pred))

# # ===== SAVE =====
# os.makedirs("models", exist_ok=True)
# joblib.dump(model, "models/xgb_improved.pkl")
# joblib.dump(scaler, "models/xgb_scaler.pkl")