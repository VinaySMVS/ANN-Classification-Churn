
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# load the trained model
model = tf.keras.models.load_model("model.h5")

# load the pickle file scaler and encoder
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('lable_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

# steamlit APP

st.title("🏦 Customer Churn Prediction")
st.write("Enter customer details below")

# -----------------------------
# Input Fields
# -----------------------------

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=600
)

geography = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=92,
    value=40
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=3
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=60000.0
)

num_of_products = st.number_input(
    "Number Of Products",
    min_value=1,
    max_value=4,
    value=2
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [1, 0]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [1, 0]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

# -----------------------------
# Create Dictionary
# -----------------------------

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One hot encode 
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))

# combine one hot encoded data with input data

input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)

# scale input data 
input_data_scaled = scaler.transform(input_data)

# predict Churn
Prediction = model.predict(input_data_scaled)
predict_prob = Prediction[0][0]

if predict_prob > 0.5:
    st.error("⚠️ The customer is likely to churn")
else:
    st.success("✅ The customer is not likely to churn")