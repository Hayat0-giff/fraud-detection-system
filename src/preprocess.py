import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

print("Loading dataset...")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(BASE_DIR, "data", "creditcard.csv")

df = pd.read_csv(file_path)

X = df.drop("Class", axis=1)
y = df["Class"]

print("Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Applying SMOTE...")

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Training model...")

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train_res, y_train_res)

print("Saving model...")

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fraud_model.pkl")

print("MODEL CREATED SUCCESSFULLY!")
