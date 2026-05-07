import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------

model = tf.keras.models.load_model("regression_model.h5")

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('lable_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
}

.prediction-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #1E1E1E;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: #00FFAA;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------

st.title("💰 Customer Salary Prediction")
st.write("Predict estimated salary using ANN Regression Model")

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("📋 Customer Details")

credit_score = st.sidebar.slider(
    "Credit Score",
    300,
    900,
    650
)

geography = st.sidebar.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.sidebar.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.sidebar.slider(
    "Age",
    18,
    92,
    35
)

tenure = st.sidebar.slider(
    "Tenure",
    0,
    10,
    5
)

balance = st.sidebar.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

num_of_products = st.sidebar.slider(
    "Number Of Products",
    1,
    4,
    2
)

has_cr_card = st.sidebar.selectbox(
    "Has Credit Card",
    [1, 0]
)

is_active_member = st.sidebar.selectbox(
    "Is Active Member",
    [1, 0]
)

# -----------------------------
# Create Input DataFrame
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
    'Exited': [0]
})

# -----------------------------
# One Hot Encoding
# -----------------------------

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)

# -----------------------------
# Scaling
# -----------------------------

input_data_scaled = scaler.transform(input_data)

# -----------------------------
# Prediction
# -----------------------------

if st.button("💡 Predict Salary"):

    prediction = model.predict(input_data_scaled)

    predicted_salary = prediction[0][0]

    st.markdown(f"""
    <div class="prediction-box">
        Predicted Salary 💰<br><br>
        ₹ {predicted_salary:,.2f}
    </div>
    """, unsafe_allow_html=True)

    st.progress(85)

    st.balloons()