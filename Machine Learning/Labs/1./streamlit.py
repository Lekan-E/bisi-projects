import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st

# Set the page title and description
st.title("Credit Loan Eligibility Predictor")
st.write("""
This app predicts whether a loan applicant is eligible for a loan 
based on various personal and financial characteristics.
""")

# # Optional password protection (remove if not needed)
# password_guess = st.text_input("Please enter your password?")
# # this password is stores in streamlit secrets
# if password_guess != st.secrets["password"]:
#     st.stop()

# Load the pre-trained model
rf_pickle = open("models/RFmodel.pkl", "rb")
rf_model = pickle.load(rf_pickle)
rf_pickle.close()


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

    # Make prediction
    new_prediction = rf_model.predict(prediction_input)

    # Display result
    st.subheader("Prediction Result:")
    if new_prediction[0] == 'Y':
        st.write("You are eligible for the loan!")
    else:
        st.write("Sorry, you are not eligible for the loan.")

st.write(
    """We used a machine learning (Random Forest) model to predict your eligibility, the features used in this prediction are ranked by relative
    importance below."""
)
st.image("feature_importance.png")
