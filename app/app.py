from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"
THRESHOLD = 0.50


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("Customer Churn Predictor")
st.write("Enter a customer's details to estimate the chance they will leave.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer")
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Has partner", ["No", "Yes"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly = st.number_input("Monthly charges ($)", min_value=18.0, max_value=120.0,
                              value=70.0, step=0.5)

with col2:
    st.subheader("Account")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])

st.subheader("Services")
phone = st.selectbox("Phone service", ["Yes", "No"])
if phone == "No":
    multiple_lines = "No"
else:
    multiple_lines = st.selectbox("Multiple lines", ["No", "Yes"])

internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
addon_names = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
               "TechSupport", "StreamingTV", "StreamingMovies"]
if internet == "No":
    addons = {name: "No" for name in addon_names}
    st.info("No internet service, so internet add-ons are set to No.")
else:
    addons = {name: st.selectbox(name, ["No", "Yes"]) for name in addon_names}

if st.button("Predict churn risk"):
    customer = pd.DataFrame([{
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        **addons,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
    }])

    prob = model.predict_proba(customer)[0, 1]
    st.metric("Churn probability", f"{prob:.0%}")
    if prob >= THRESHOLD:
        st.error("High risk: consider a retention offer.")
    else:
        st.success("Lower risk.")