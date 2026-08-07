import os
import pandas as pd
import pickle
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from logger import get_logger
logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load pretrained model and the scaler used fr training
try:
    with open(os.path.join(BASE_DIR, 'models', 'RFmodel.pkl'), 'rb') as file:
        rf_model = pickle.load(file)

    with open(os.path.join(BASE_DIR, 'models', 'scaler.pkl'), 'rb') as file:
        scaler = pickle.load(file)

    logger.info("Model and scaler loaded successfully.")
except Exception:
    logger.exception("Failed to load model or scaler.")
    st.error("The prediction model could not be loaded.")
    st.stop()

st.title("Credit Loan Eligibility Predictor")
st.write("""
This app predicts whether a loan applicant is eligible for a loan
based on various personal and financial characteristics.
""")

# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Loan Applicant Details")

    # Gender input
    Gender = st.selectbox("Gender", options=["Male", "Female"])

    # Marital Status
    Married = st.selectbox("Marital Status", options=["Yes", "No"])

    # Dependents
    Dependents = st.selectbox("Number of Dependents",
                               options=["0", "1", "2", "3+"])

    # Education
    Education = st.selectbox("Education Level",
                              options=["Graduate", "Not Graduate"])

    # Self Employment
    Self_Employed = st.selectbox("Self Employed", options=["Yes", "No"])

    # Applicant Income
    ApplicantIncome = st.number_input("Applicant Monthly Income",
                                       min_value=0,
                                       step=1000)

    # Coapplicant Income
    CoapplicantIncome = st.number_input("Coapplicant Monthly Income",
                                         min_value=0,
                                         step=1000)

    # Loan Amount
    LoanAmount = st.number_input("Loan Amount",
                                  min_value=0,
                                  step=1000)

    # Loan Amount Term
    Loan_Amount_Term = st.selectbox("Loan Amount Term (Months)",
                                    options=["360", "180", "240", "120", "60"])

    # Credit History
    Credit_History = st.selectbox("Credit History",
                                  options=["1", "0"])

    # Property Area
    Property_Area = st.selectbox("Property Area",
                                 options=["Urban", "Semiurban", "Rural"])

    # Submit button
    submitted = st.form_submit_button("Predict Loan Eligibility")

# Handle the dummy variables to pass to the model
if submitted:
    logger.info(
        "User query received: Gender=%s, Married=%s, Dependents=%s, Education=%s, "
        "Self_Employed=%s, ApplicantIncome=%s, CoapplicantIncome=%s, LoanAmount=%s, "
        "Loan_Amount_Term=%s, Credit_History=%s, Property_Area=%s",
        Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
        CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area,
    )

    try:
        Gender_Male = 0 if Gender == "Female" else 1
        Gender_Female = 1 if Gender == "Female" else 0

        Married_Yes = 1 if Married == "Yes" else 0
        Married_No = 1 if Married == "No" else 0

        # Handle dependents
        Dependents_0 = 1 if Dependents == "0" else 0
        Dependents_1 = 1 if Dependents == "1" else 0
        Dependents_2 = 1 if Dependents == "2" else 0
        Dependents_3 = 1 if Dependents == "3+" else 0

        Education_Graduate = 1 if Education == "Graduate" else 0
        Education_Not_Graduate = 1 if Education == "Not Graduate" else 0

        Self_Employed_Yes = 1 if Self_Employed == "Yes" else 0
        Self_Employed_No = 1 if Self_Employed == "No" else 0

        Property_Area_Rural = 1 if Property_Area == "Rural" else 0
        Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
        Property_Area_Urban = 1 if Property_Area == "Urban" else 0

        # Convert Loan Amount Term and Credit History to integers
        Loan_Amount_Term = int(Loan_Amount_Term)
        Credit_History = int(Credit_History)

        # Prepare the input for prediction. This has to go in the same order as it was trained
        prediction_input = [[ApplicantIncome, CoapplicantIncome, LoanAmount,
            Loan_Amount_Term, Credit_History, Gender_Female, Gender_Male,
            Married_No, Married_Yes, Dependents_0, Dependents_1,
            Dependents_2, Dependents_3, Education_Graduate,
            Education_Not_Graduate, Self_Employed_No, Self_Employed_Yes,
            Property_Area_Rural, Property_Area_Semiurban, Property_Area_Urban
        ]]

        # Scale the input the same way the training data was scaled
        prediction_input_scaled = scaler.transform(prediction_input)

        # Make prediction
        new_prediction = rf_model.predict(prediction_input_scaled)
        approval_probability = rf_model.predict_proba(prediction_input_scaled)[0][1]
        logger.info("Prediction result: %s (approval probability: %.4f)", new_prediction[0], approval_probability)

        # Display result
        st.subheader("Prediction Result:")
        if new_prediction[0] == 1:
            st.success("You are eligible for the loan! 🎉")
        else:
            st.error("Sorry, you are not eligible for the loan. 🚫")
        st.metric("Estimated Approval Probability", f"{approval_probability:.0%}")

        # Echo back the submitted details so the user can confirm what was predicted on
        st.subheader("Submitted Applicant Details:")
        submitted_details = pd.DataFrame({
            "Field": [
                "Gender", "Marital Status", "Dependents", "Education Level",
                "Self Employed", "Applicant Monthly Income", "Coapplicant Monthly Income",
                "Loan Amount", "Loan Amount Term (Months)", "Credit History", "Property Area",
            ],
            "Value": [
                Gender, Married, Dependents, Education, Self_Employed,
                f"{ApplicantIncome:,}", f"{CoapplicantIncome:,}", f"{LoanAmount:,}",
                str(Loan_Amount_Term), str(Credit_History), Property_Area,
            ],
        })
        st.table(submitted_details.set_index("Field"))
    except Exception:
        logger.exception("Prediction failed for the submitted query.")
        st.error("Something went wrong while making the prediction. Please check your inputs and try again.")

st.write(
    """We used a machine learning (Random Forest) model to predict your eligibility. The chart below shows the
    model's overall feature importance (learned from the training data)."""
)
st.image(os.path.join(BASE_DIR, "feature_importance.png"))
