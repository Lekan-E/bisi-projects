import pandas as pd
import pickle
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load pretrained model

rf_pickle = open('models/RFmodel.pkl', 'rb')
rf_model = pickle.load(rf_pickle)
rf_pickle.close()

# page title and description
st.title('House Price Prediction')
st.write("""
This application estimates the fair transaction price of a property
based on its features. Fill in the details below and click **Predict Price**.
""")

# Year Built
YearBuilt = st.slider('Year Built', 1880, 2014, 1982)

# prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Property Details")

    # Year Sold - cannot be earlier than Year Built
    YearSold = st.slider(
        'Year Sold', min_value=YearBuilt, max_value=2016,
        value=min(max(YearBuilt, 2007), 2016),
        help="Cannot be earlier than the Year Built."
    )
    
    # Property Type
    PropertyType = st.selectbox('Property Type', options=['Bungalow', 'Condo'])

    # Basement
    Basement = st.selectbox('Basement', options=['Yes','No'])

    # Beds
    Beds = st.selectbox('Bedrooms', options=[1, 2, 3, 4, 5], index=2)

    # baths
    Baths = st.selectbox('Bathrooms', options=[1, 2, 3, 4, 5, 6], index=2)

    # Sqft
    Sqft = st.number_input(
            "Living Area (sq ft)", min_value=200, max_value=6000,
            value=1500, step=50,
            help="Suggested range: 200 - 6,000 sq ft"
        )

    # Lot Size
    LotSize = st.number_input(
            "Lot Size (sq ft)", min_value=0, max_value=436481,
            value=5000, step=100,
            help="Suggested range: 0 - 436,481 sq ft"
        )

    # Property Tax
    PropertyTax = st.number_input(
            "Annual Property Tax ($)", min_value=88, max_value=4508,
            value=450, step=10,
            help="Suggested range: $88 - $4,508"
        )

    # Insurance
    Insurance = st.number_input(
            "Annual Insurance ($)", min_value=30, max_value=1374,
            value=140, step=10,
            help="Suggested range: $30 - $1,374"
        )

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

# Handle the engineered/dummy variables to pass to the model
if submitted:
    # Property type dummy (get_dummies with drop_first=True keeps only Condo)
    property_type_Condo = 1 if PropertyType == "Condo" else 0

    # Basement as binary
    basement = 1 if Basement == "Yes" else 0

    # Engineered features, computed the same way as during training
    popular = 1 if (Beds == 2 and Baths == 2) else 0
    recession = 1 if (2010 <= YearSold <= 2013) else 0
    property_age = YearSold - YearBuilt

    # Prepare the input for prediction. This has to go in the same order as it was trained
    prediction_input = [[
        YearSold, PropertyTax, Insurance, Beds, Baths, Sqft, YearBuilt,
        LotSize, basement, popular, recession, property_age, property_type_Condo
    ]]

    # Make prediction
    new_prediction = rf_model.predict(prediction_input)

    # Display result
    st.subheader("Prediction Result:")
    st.write(f"Estimated Price: ${new_prediction[0]:,.0f}")