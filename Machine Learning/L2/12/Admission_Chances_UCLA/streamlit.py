import pandas as pd
import pickle
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from logger import get_logger
logger = get_logger(__name__)

FEATURE_COLS = ['GRE_Score', 'TOEFL_Score', 'SOP', 'LOR', 'CGPA',
                'University_Rating_1', 'University_Rating_2', 'University_Rating_3',
                'University_Rating_4', 'University_Rating_5', 'Research_0', 'Research_1']

# Load the trained model and the scaler used to prepare its input
try:
    with open('models/MLPmodel.pkl', 'rb') as file:
        mlp_model = pickle.load(file)

    with open('models/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    logger.info("Model and scaler loaded successfully.")
except Exception:
    logger.exception("Failed to load model or scaler.")
    st.error("The prediction model could not be loaded. Please contact the app administrator.")
    st.stop()

st.title("UCLA Admission Chance Predictor")
st.write("""
This app predicts a student's chance of being admitted to UCLA's Master's program based on
their academic profile, using a neural network (MLPClassifier) trained on historical
admission data.
""")

with st.form("user_inputs"):
    st.subheader("Applicant Profile")

    GRE_Score = st.number_input("GRE Score (out of 340)", min_value=0, max_value=340, value=310, step=1)
    TOEFL_Score = st.number_input("TOEFL Score (out of 120)", min_value=0, max_value=120, value=100, step=1)
    University_Rating = st.selectbox("Bachelor's University Rating (out of 5)", options=["1", "2", "3", "4", "5"])
    SOP = st.slider("Statement of Purpose Strength (out of 5)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
    LOR = st.slider("Letter of Recommendation Strength (out of 5)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
    CGPA = st.number_input("Undergraduate CGPA (out of 10)", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
    Research = st.selectbox("Research Experience", options=["Yes", "No"])

    submitted = st.form_submit_button("Predict Admission Chance")

if submitted:
    logger.info(
        "User query received: GRE_Score=%s, TOEFL_Score=%s, University_Rating=%s, SOP=%s, "
        "LOR=%s, CGPA=%s, Research=%s",
        GRE_Score, TOEFL_Score, University_Rating, SOP, LOR, CGPA, Research,
    )

    try:
        # Handle the dummy variables to pass to the model
        University_Rating_1 = 1 if University_Rating == "1" else 0
        University_Rating_2 = 1 if University_Rating == "2" else 0
        University_Rating_3 = 1 if University_Rating == "3" else 0
        University_Rating_4 = 1 if University_Rating == "4" else 0
        University_Rating_5 = 1 if University_Rating == "5" else 0

        Research_Yes = 1 if Research == "Yes" else 0
        Research_No = 1 if Research == "No" else 0

        # Prepare the input for prediction. This has to go in the same order as it was trained
        prediction_input = pd.DataFrame([[
            GRE_Score, TOEFL_Score, SOP, LOR, CGPA,
            University_Rating_1, University_Rating_2, University_Rating_3,
            University_Rating_4, University_Rating_5, Research_No, Research_Yes
        ]], columns=FEATURE_COLS)

        # Scale the input the same way the training data was scaled
        prediction_input_scaled = scaler.transform(prediction_input)

        # Make prediction
        new_prediction = mlp_model.predict(prediction_input_scaled)
        admit_probability = mlp_model.predict_proba(prediction_input_scaled)[0][1]
        logger.info("Prediction result: %s (admit probability: %.4f)", new_prediction[0], admit_probability)

        # Display result
        st.subheader("Prediction Result:")
        if new_prediction[0] == 1:
            st.write(f"You are likely to be admitted! Estimated admission chance: {admit_probability:.0%}")
        else:
            st.write(f"You are unlikely to be admitted. Estimated admission chance: {admit_probability:.0%}")
    except Exception:
        logger.exception("Prediction failed for the submitted query.")
        st.error("Something went wrong while making the prediction. Please check your inputs and try again.")

st.write(
    """We used a neural network (MLPClassifier) model to predict your admission chance. The chart
    below shows how the model's training loss decreased over training iterations."""
)
st.image("loss_curve.png")
