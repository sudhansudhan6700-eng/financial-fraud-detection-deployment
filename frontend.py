import streamlit as st
import requests

st.set_page_config(page_title="Fraud Detection", page_icon="💳")

st.title("💳 Financial Fraud Detection")
st.write("Enter transaction details to check whether a transaction is fraudulent.")

API_URL = "http://127.0.0.1:8000/predict"

features = {}

features["Time"] = st.number_input("Time", value=1000.0)

for i in range(1, 29):
    features[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

features["Amount"] = st.number_input("Amount", value=100.0, min_value=0.0)

if st.button("Check Transaction"):
    try:
        response = requests.post(
            API_URL,
            json={"features": features}
        )

        if response.status_code == 200:
            result = response.json()

            if result["prediction"] == 1:
                st.error("⚠️ Fraudulent Transaction Detected!")
            else:
                st.success("✅ Legitimate Transaction")

            st.write("Fraud probability:", result["fraud_probability"])
        else:
            st.error(response.text)

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Please start the backend first.")