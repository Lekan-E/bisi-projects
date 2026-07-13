import pandas as pd
import pickle
import streamlit as st
from datetime import datetime

# Load pretrained model

rf_pickle = open('models/RFmodel.pkl', 'rb')
rf_model = pickle.load(rf_pickle)
rf_pickle.close()

# page title and description
st.title('House Price Prediction')
st.write("""
This application estimates the fair transaction price
of a property based on building features and more.   
""")

# prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("House Feature Details")

    # Property Type
    PropertyType = st.selectbox('Property Type', options=['Bungalow', 'Condo'])

    # Basement
    Basement = st.selectbox('Basement', options=['Yes','No'])

    # Beds
    options_bed = [1,2,3,4,5]
    Beds = st.segmented_control('Beds', options_bed, selection_mode="single")

    # baths
    options_bath = [1,2,3,4,5,6]
    Baths = st.segmented_control('Baths', options_bath, selection_mode="single")

    # Year Built
    YearBuilt = st.slider('Year Built', 1880, 2014, 1982)

    # Year Sold
    YearSold = st.slider('Year Sold', 1993, 2016, 2007)

    # Sqft
    
    # Lot Size

    # Property Tax

    # Insurance

