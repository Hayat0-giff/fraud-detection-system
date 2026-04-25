# 🏦 FraudShield AI – Fraud Detection System

 Overview
FraudShield AI is a machine learning-powered web application that detects fraudulent credit card transactions in real time. It includes explainable AI, a live dashboard, and a secure transaction logging system.

The goal of this project is to demonstrate an end-to-end ML system from model inference to deployment with an interactive UI.



# ⚙️ Features

 🔍 Real-time fraud detection
 📊 Fraud probability scoring
 🧠 SHAP explainability (AI decision interpretation)
 📁 Transaction history tracking using SQLite
 🔐 Admin login panel
 📈 Live dashboard analytics
 💻 Streamlit-based web interface



 🧠 Machine Learning Model

 Trained on a highly imbalanced fraud dataset
 Uses PCA-transformed features (V1–V28)
 Binary classification (Fraud / Legit)
 Outputs probability score for risk evaluation



## 🧠 Explainable AI (SHAP)

SHAP is used to explain model predictions by showing:

- Which features increase fraud risk
- Which features reduce fraud risk
- Relative importance of each feature

This makes the model interpretable instead of a black box.

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- SHAP
- Streamlit
- Plotly
- SQLite

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
