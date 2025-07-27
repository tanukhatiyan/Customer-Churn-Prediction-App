import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

# Load saved components
model = tf.keras.models.load_model('model.h5')
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

# Expected column order
expected_order = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain'
]

# Streamlit UI
st.title("Customer Churn Prediction")

# User Inputs
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", value=0.0)
credit_score = st.number_input("Credit Score", value=600)
estimated_salary = st.number_input("Estimated Salary", value=50000.0)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare input
input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary,
    'Geography': geography
}
df = pd.DataFrame([input_data])

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform(df[['Geography']]).toarray()
geo_columns = onehot_encoder_geo.get_feature_names_out(['Geography'])
geo_df = pd.DataFrame(geo_encoded, columns=geo_columns)

# Merge and clean up
df = df.drop('Geography', axis=1)
df = pd.concat([df, geo_df], axis=1)

# Add missing one-hot columns (if any)
for col in expected_order:
    if col not in df.columns:
        df[col] = 0

# Ensure correct order
df = df[expected_order]

# Scale and predict
scaled_input = scaler.transform(df)
prediction = model.predict(scaled_input)[0][0]
churn_result = "Customer is likely to churn." if prediction > 0.5 else "Customer is unlikely to churn."

# Display result
st.subheader("Prediction Result")
st.write(f"**Probability of churn:** {prediction:.2f}")
st.success(churn_result if prediction <= 0.5 else "")
st.error(churn_result if prediction > 0.5 else "")
