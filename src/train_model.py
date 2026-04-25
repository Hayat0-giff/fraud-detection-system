import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

print(" TRAINING STARTED")

# LOAD DATA
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# BALANCE DATA
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# MODEL
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_res, y_train_res)

# PREDICTIONS
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# EVALUATION
print("\n📊 CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

print("\n📈 ROC AUC SCORE")
print(roc_auc_score(y_test, y_prob))

# SAVE MODEL
joblib.dump(model, "model/fraud_model.pkl")

print("\n✅ MODEL TRAINED AND SAVED")