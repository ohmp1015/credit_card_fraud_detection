import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.subheader("Developed by Ohm Patel")
st.markdown("Detect fraudulent transactions using Logistic Regression model.")
st.markdown("---")

# Train model on balanced data
@st.cache_resource
def train_model():
    data = pd.read_csv("creditcard.csv")
    normal = data[data.Class == 0]
    fraud = data[data.Class == 1]
    normal_sample = normal.sample(n=492, random_state=42)
    balanced_data = pd.concat([normal_sample, fraud], axis=0)

    X = balanced_data.drop('Class', axis=1)
    Y = balanced_data['Class']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    model = linear_model.LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    return model, X.columns.tolist()

# Load model and feature list
model, feature_cols = train_model()

# --- Section 1: Batch Prediction ---
# --- Section 1: Batch Prediction ---
st.header("📁 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with transaction data (without 'Class' column)", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)

    if 'Class' in input_data.columns:
        input_data = input_data.drop('Class', axis=1)

    if list(input_data.columns) != feature_cols:
        st.error("❌ Uploaded data columns do not match expected features.")
        st.text(f"Expected columns:\n{feature_cols}")
    else:
        st.success("✅ File validated. Making predictions...")
        predictions = model.predict(input_data)
        input_data['Prediction'] = predictions
        input_data['Prediction_Label'] = input_data['Prediction'].apply(lambda x: 'Fraud' if x == 1 else 'Normal')

        st.subheader("🔎 Prediction Results:")
        st.dataframe(input_data[['Prediction', 'Prediction_Label']].head(10))

        # Summary
        st.subheader("📊 Prediction Summary")
        summary = input_data['Prediction_Label'].value_counts().reset_index()
        summary.columns = ['Transaction Type', 'Count']
        st.table(summary)

        # Pie chart
        st.subheader("🧁 Transaction Distribution")
        st.plotly_chart(
            {
                "data": [
                    {
                        "labels": summary['Transaction Type'],
                        "values": summary['Count'],
                        "type": "pie",
                        "hole": 0.4,
                        "marker": {"colors": ["#1f77b4", "#ff4136"]},
                    }
                ],
                "layout": {"margin": {"l": 0, "r": 0, "t": 0, "b": 0}},
            }
        )

        # Total fraud amount vs normal
        input_data['Amount'] = input_data['Amount'].astype(float)
        amount_summary = input_data.groupby('Prediction_Label')['Amount'].sum().reset_index()
        amount_summary.columns = ['Transaction Type', 'Total Amount']
        st.subheader("💰 Total Amount per Transaction Type")
        st.table(amount_summary)

        # Download results
        csv = input_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Predictions as CSV",
            data=csv,
            file_name='fraud_predictions.csv',
            mime='text/csv',
        )


# --- Section 2: Manual Prediction ---
st.header("✍️ Enter Transaction Details for Prediction")

with st.form("manual_input_form"):
    cols = st.columns(3)
    user_input = {}
    for idx, col in enumerate(feature_cols):
        with cols[idx % 3]:
            val = st.number_input(f"{col}", value=0.0, step=0.1, format="%.4f")
            user_input[col] = val

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    label = "Fraud" if prediction == 1 else "Normal"

    st.success(f"🧾 **Prediction:** {label}")
    st.info(f"Raw Output: {prediction}")