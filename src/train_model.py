# =========================================================
# FRAUD DETECTION MODEL TRAINING (CLEAN VERSION)
# =========================================================

import pandas as pd                  # Handles dataset (CSV → table format)
import joblib                        # Saves trained model to disk
import os                            # Handles file paths safely

from sklearn.model_selection import train_test_split  # Splits data into train/test
from sklearn.ensemble import RandomForestClassifier    # ML model (tree-based classifier)

# =========================================================
# LOAD DATASET
# =========================================================

# Read dataset from local folder
df = pd.read_csv("data/creditcard.csv")

# =========================================================
# FEATURES + TARGET
# =========================================================

# X = input features (ALL columns except target)
# These include: V1–V28, Time, Amount
X = df.drop("Class", axis=1)

# y = target label (0 = legit, 1 = fraud)
y = df["Class"]

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

# Split data into training and testing sets
# stratify=y keeps fraud ratio balanced
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# MODEL INITIALIZATION
# =========================================================

# RandomForest = multiple decision trees voting system
model = RandomForestClassifier(
    n_estimators=30,   # number of trees (faster = lower)
    max_depth=10,      # limits complexity (faster + prevents overfitting)
    n_jobs=-1,         # uses all CPU cores (speed boost)
    random_state=42
)

# =========================================================
# TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)

# =========================================================
# SAVE MODEL
# =========================================================

# Create folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save trained model to file
joblib.dump(model, "model/fraud_model.pkl")

print("✅ Model trained and saved successfully")
