import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

# === Page Config ===
st.set_page_config(page_title="California Housing Price Prediction", layout="centered")

# === Custom CSS ===
st.markdown(
    """
    <style>
    .main {
        background-color: #f9fcff;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #0e7490 !important;
    }
    label, .stMarkdown p, .stMarkdown span {
        color: #0e7490 !important;
        font-weight: 500;
    }
    .feature-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .prediction-box {
        background-color: #ecfdf5;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #10b981;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: #065f46;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === Dropbox Model URL ===
MODEL_URL = "https://www.dropbox.com/scl/fi/2jl97wjcn4u2krg75iovp/final_california_model.joblib?rlkey=j0rac835tusneeo7ime9wne6c&st=ftoc93d6&dl=1"

@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    return joblib.load(BytesIO(response.content))

# Load model
model = load_model()

# === Title ===
st.title("🏠 California Housing Price Prediction")
st.markdown(
    "<p style='text-align:center; font-size:18px;'>"
    "Use this app to <b>predict the median house value</b> "
    "based on property and demographic details in California."
    "</p>",
    unsafe_allow_html=True
)

# === Input ===
st.markdown("## 📋 Input Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("🏘 **Housing Median Age**")
    housing_median_age = st.number_input("Housing Median Age", value=20.0, min_value=1.0, max_value=52.0, step=1.0)
    
    st.markdown("🚪 **Total Rooms**")
    total_rooms = st.number_input("Total Rooms", value=2000.0, min_value=1.0, step=1.0)
    
    st.markdown("🛏 **Total Bedrooms**")
    total_bedrooms = st.number_input("Total Bedrooms", value=400.0, min_value=1.0, step=1.0)

with col2:
    st.markdown("👥 **Population**")
    population = st.number_input("Population", value=1000.0, min_value=1.0, step=1.0)
    
    st.markdown("🏡 **Households**")
    households = st.number_input("Households", value=300.0, min_value=1.0, step=1.0)
    
    st.markdown("💰 **Median Income (in 10k USD)**")
    median_income = st.number_input("Median Income", value=3.0, min_value=0.1, step=0.1)

st.markdown("</div>", unsafe_allow_html=True)

# Ocean proximity dropdown
st.markdown("## 🌊 Ocean Proximity")
st.markdown("**Select the category of ocean proximity:**")
ocean_proximity = st.selectbox(
    "Ocean Proximity",
    options=['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
)
st.markdown("</div>", unsafe_allow_html=True)

# === Prediction ===
if st.button("🔍 Predict House Value"):
    input_df = pd.DataFrame([{
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    prediction = model.predict(input_df)[0]

    st.markdown("---")
    st.subheader("💡 Prediction Result")
    st.markdown(
        f"<div class='prediction-box'>🏡 Predicted Median House Value: ${prediction:,.2f}</div><br>",
        unsafe_allow_html=True
    )
