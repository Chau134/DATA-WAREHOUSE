import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score

from imblearn.over_sampling import SMOTE

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping

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

# ===== SCALE =====
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===== SMOTE (NHẸ HƠN RF/XGB) =====
sm = SMOTE(sampling_strategy={1: 220}, k_neighbors=3, random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# ===== ONE-HOT =====
encoder = OneHotEncoder(sparse_output=False)
y_train_ohe = encoder.fit_transform(y_train.values.reshape(-1,1))
y_test_ohe = encoder.transform(y_test.values.reshape(-1,1))

# ===== CLASS WEIGHT =====
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

if 1 in class_weights:
    class_weights[1] *= 1.5  # boost nhẹ N1

# ===== MODEL =====
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    
    Dense(128, activation='relu'),
    Dropout(0.3),  # 🔥 chống overfit
    
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(len(classes), activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ===== EARLY STOPPING =====
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ===== TRAIN =====
model.fit(
    X_train, y_train_ohe,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ===== PREDICT =====
y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)

# ===== EVALUATE =====
print("\n===== ANN IMPROVED =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred, zero_division=0))

# ===== SAVE =====
os.makedirs("models", exist_ok=True)
model.save("models/ann_improved.h5")
joblib.dump(scaler, "models/ann_scaler.pkl")
joblib.dump(encoder, "models/ann_encoder.pkl")