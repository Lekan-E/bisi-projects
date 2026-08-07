import pandas as pd
import pickle
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from logger import get_logger
logger = get_logger(__name__)


# Load the trained model
try:
    with open('models/RFmodel.pkl', 'rb') as file:
        rf_model = pickle.load(file)

    logger.info("Model loaded successfully.")
    
except Exception:
    logger.exception("Failed to load model.")
    st.error("The prediction model could not be loaded. Please contact the app administrator.")
    st.stop()

st.title("House Price Prediction")
st.write("""
This app estimates the fair transaction price of a property based on its features, using a
Random Forest model trained on historical sales data. Fill in the details below and click
**Predict Price**.
""")

# Year Built is used to bound the Year Sold input below, so it's collected outside the form
YearBuilt = st.slider('Year Built', 1880, 2014, 1982)

with st.form("user_inputs"):
    st.subheader("Property Details")

    # Year Sold cannot be earlier than Year Built
    YearSold = st.slider(
        'Year Sold', min_value=YearBuilt, max_value=2016,
        value=min(max(YearBuilt, 2007), 2016),
        help="Cannot be earlier than the Year Built."
    )

    PropertyType = st.selectbox('Property Type', options=['Bungalow', 'Condo'])
    Basement = st.selectbox('Basement', options=['Yes', 'No'])
    Beds = st.selectbox('Bedrooms', options=[1, 2, 3, 4, 5], index=2)
    Baths = st.selectbox('Bathrooms', options=[1, 2, 3, 4, 5, 6], index=2)

    Sqft = st.number_input(
        "Living Area (sq ft)", min_value=200, max_value=6000,
        value=1500, step=50,
        help="Suggested range: 200 - 6,000 sq ft"
    )

    LotSize = st.number_input(
        "Lot Size (sq ft)", min_value=0, max_value=436481,
        value=5000, step=100,
        help="Suggested range: 0 - 436,481 sq ft"
    )

    PropertyTax = st.number_input(
        "Annual Property Tax ($)", min_value=88, max_value=4508,
        value=450, step=10,
        help="Suggested range: $88 - $4,508"
    )

    Insurance = st.number_input(
        "Annual Insurance ($)", min_value=30, max_value=1374,
        value=140, step=10,
        help="Suggested range: $30 - $1,374"
    )

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

if submitted:
    logger.info(
        "User query received: YearBuilt=%s, YearSold=%s, PropertyType=%s, Basement=%s, "
        "Beds=%s, Baths=%s, Sqft=%s, LotSize=%s, PropertyTax=%s, Insurance=%s",
        YearBuilt, YearSold, PropertyType, Basement, Beds, Baths, Sqft, LotSize,
        PropertyTax, Insurance,
    )

    try:
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
        logger.info("Prediction result: %s", new_prediction[0])

        # Display result
        st.subheader("Prediction Result:")
        st.metric("Estimated Price", f"${new_prediction[0]:,.0f}")

        # Echo back the submitted details so the user can confirm what was predicted on
        st.subheader("Submitted Property Details:")
        submitted_details = pd.DataFrame({
            "Field": [
                "Year Built", "Year Sold", "Property Type", "Basement", "Bedrooms",
                "Bathrooms", "Living Area (sq ft)", "Lot Size (sq ft)",
                "Annual Property Tax ($)", "Annual Insurance ($)",
            ],
            "Value": [
                str(YearBuilt), str(YearSold), PropertyType, Basement, str(Beds),
                str(Baths), f"{Sqft:,}", f"{LotSize:,}",
                f"{PropertyTax:,}", f"{Insurance:,}",
            ],
        })
        st.table(submitted_details.set_index("Field"))
    except Exception:
        logger.exception("Prediction failed for the submitted query.")
        st.error("Something went wrong while making the prediction. Please check your inputs and try again.")

st.write(
    """We used a machine learning (Random Forest) model to predict this price. The chart below
    shows the model's overall feature importance (learned from the training data)."""
)
st.image("feature_importance.png")
