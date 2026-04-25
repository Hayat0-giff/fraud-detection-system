# =========================================================
# FRAUDSHIELD AI - PRODUCTION FINTECH VERSION
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import shap
import sqlite3
from datetime import datetime

# =========================================================
# MODEL
# =========================================================

model = joblib.load("model/fraud_model.pkl")
features = model.feature_names_in_

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_explainer(model)

# =========================================================
# DATABASE
# =========================================================

conn = sqlite3.connect("fraud_logs.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT,
    amount REAL,
    txn_time REAL,
    prediction INTEGER,
    probability REAL,
    risk TEXT,
    user TEXT
)
""")
conn.commit()

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🏦",
    layout="wide"
)

# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>
.block-container {
    padding: 2rem;
    background-color: #f7f9fc;
}

[data-testid="stMetric"] {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.06);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #111111 !important;
}

h1, h2, h3 {
    color: #111111 !important;
}

p, span, div {
    color: #111111;
}

[data-testid="stSidebar"] {
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

# =========================================================
# NAVIGATION
# =========================================================

st.sidebar.title("🏦 FraudShield AI")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Fraud Detection", "History", "Admin Panel", "About"]
)

# =========================================================
# ADMIN LOGIN
# =========================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🔐 Admin Login")

user = st.sidebar.text_input("Username")
pw = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    if user == "hayat" and pw == "311451":
        st.session_state.is_admin = True
        st.sidebar.success("Authenticated")
    else:
        st.sidebar.error("Invalid Credentials")

# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("🏦 Fraud Detection Dashboard")

    df = pd.read_sql_query("SELECT * FROM transactions", conn)

    total = len(df)
    fraud = len(df[df["prediction"] == 1]) if total > 0 else 0
    legit = total - fraud

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", total)
    col2.metric("Fraud Cases", fraud)
    col3.metric("Legit Cases", legit)

    st.markdown("### 📊 Recent Transactions")

    if total > 0:
        st.dataframe(df.tail(10))
    else:
        st.info("No transactions yet.")

# =========================================================
# FRAUD DETECTION
# =========================================================

elif page == "Fraud Detection":

    st.title("🔍 Live Fraud Detection Engine")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Transaction Amount", value=50.0)

    with col2:
        txn_time = st.number_input("Transaction Time (seconds)", value=1000.0)

    if st.button("Analyze Transaction"):

        sample = pd.DataFrame(np.zeros((1, len(features))), columns=features)

        amount_s = np.log1p(amount)
        time_s = np.log1p(txn_time)

        for f in features:
            lf = f.lower()

            if lf == "amount":
                sample[f] = amount_s
            elif lf in ["time", "txn_time"]:
                sample[f] = time_s
            elif lf.startswith("v"):
                sample[f] = np.random.normal(0, 0.3)
            else:
                sample[f] = 0

        prob = model.predict_proba(sample)[0][1]
        pred = model.predict(sample)[0]

        prob = float(np.clip(prob, 0.01, 0.99))

        risk = "LOW" if prob < 0.3 else "MEDIUM" if prob < 0.7 else "HIGH"

        col1, col2, col3 = st.columns(3)

        col1.metric("ML Probability", f"{prob:.4f}")
        col2.metric("Risk Level", risk)
        col3.metric("Decision", "FRAUD ❌" if pred == 1 else "SAFE ✅")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Fraud Risk Score"},
            gauge={"axis": {"range": [0, 100]}}
        ))

        st.plotly_chart(fig, use_container_width=True)

        # =========================================================
        # 🧠 SHAP SECTION (ONLY MODIFIED PART)
        # =========================================================

        st.markdown("## 🧠 AI Explanation Layer")

        try:
            shap_values = explainer.shap_values(sample)
            shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap_vals = np.array(shap_vals).reshape(-1)

            min_len = min(len(features), len(shap_vals))

            shap_df = pd.DataFrame({
                "Feature": features[:min_len],
                "Impact": shap_vals[:min_len]
            })

            shap_df = shap_df.sort_values(by="Impact", key=abs, ascending=False)

            # =====================================================
            # ADDED TRANSLATION LAYER (NEW ONLY)
            # =====================================================

            def explain_feature(name):
                name_lower = name.lower()

                if name_lower == "amount":
                    return "Transaction amount behavior"
                elif "time" in name_lower:
                    return "Transaction timing pattern"
                elif name_lower.startswith("v"):
                    return "Hidden behavioral pattern (anonymized feature)"
                else:
                    return name

            st.markdown("### 🔍 Key Drivers")

            for _, row in shap_df.head(6).iterrows():

                label = explain_feature(row["Feature"])

                if row["Impact"] > 0:
                    st.write(f"🔴 {label} → increases fraud risk")
                else:
                    st.write(f"🟢 {label} → reduces fraud risk")

        except:
            st.warning("SHAP unavailable")

        # =========================================================
        # HISTORY SAVE
        # =========================================================

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Amount": amount,
            "Txn_Time": txn_time,
            "Prediction": int(pred),
            "Probability": float(prob),
            "Risk": risk
        })

        cursor.execute("""
            INSERT INTO transactions (time, amount, txn_time, prediction, probability, risk, user)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%H:%M:%S"),
            amount,
            txn_time,
            int(pred),
            float(prob),
            risk,
            user
        ))

        conn.commit()

# =========================================================
# HISTORY
# =========================================================

elif page == "History":

    st.title("📁 Transaction History")

    df = pd.DataFrame(st.session_state.history)

    if df.empty:
        st.info("No history yet.")
    else:
        st.dataframe(df)
        st.bar_chart(df["Prediction"].value_counts())

# =========================================================
# ADMIN PANEL
# =========================================================

elif page == "Admin Panel":

    if not st.session_state.is_admin:
        st.error("Admin access required")
    else:
        st.title("🔐 Admin Panel")

        df = pd.read_sql_query("SELECT * FROM transactions", conn)

        st.dataframe(df)
        st.bar_chart(df["risk"].value_counts())

# =========================================================
# ABOUT
# =========================================================

else:

    st.title("FraudShield AI")

    st.markdown("""
### 💼 Production Fraud Detection System

✔ ML prediction  
✔ SHAP explainability  
✔ Risk engine  
✔ Admin system  
✔ Transaction logging  
✔ Clean fintech UI  
""")
