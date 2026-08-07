import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from logger import get_logger
logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_COLS = ['Age', 'Annual_Income', 'Spending_Score']

# Load the trained clustering model, the scaler used to prepare its input, and the
# customer data the model was trained on
try:
    with open(os.path.join(BASE_DIR, 'models', 'KMeansmodel.pkl'), 'rb') as file:
        kmeans_model = pickle.load(file)

    with open(os.path.join(BASE_DIR, 'models', 'scaler.pkl'), 'rb') as file:
        scaler = pickle.load(file)

    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'Processed_Mall_Customers.csv'))
    X_scaled = scaler.transform(df[FEATURE_COLS])
    df['Cluster'] = kmeans_model.predict(X_scaled)

    logger.info("Model, scaler, and customer data loaded successfully.")
except Exception:
    logger.exception("Failed to load model, scaler, or customer data.")
    st.error("The segmentation model could not be loaded.")
    st.stop()

# Describe each segment relative to the customer base's median income/spending, plus its
# average age, so the labels stay meaningful even if the data changes
income_mid = df['Annual_Income'].median()
spend_mid = df['Spending_Score'].median()
cluster_profile = df.groupby('Cluster').agg(
    Annual_Income=('Annual_Income', 'mean'),
    Spending_Score=('Spending_Score', 'mean'),
    Age=('Age', 'mean'),
)
cluster_profile['Segment'] = cluster_profile.apply(
    lambda row: "{} Income, {} Spending".format(
        "High" if row['Annual_Income'] >= income_mid else "Low",
        "High" if row['Spending_Score'] >= spend_mid else "Low",
    ),
    axis=1,
)

st.title("Mall Customer Segmentation")
st.write("""
This app assigns a shopper to one of the mall's customer segments based on their age, annual
income, and spending score, using a K-Means clustering model trained on the mall's customer
data.
""")

with st.form("user_inputs"):
    st.subheader("Customer Details")

    Age = st.number_input("Age", min_value=10, max_value=100, value=30, step=1)
    Annual_Income = st.number_input("Annual Income (in $1,000s)", min_value=0, max_value=200, value=60, step=1)
    Spending_Score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)

    submitted = st.form_submit_button("Find My Segment")

if submitted:
    logger.info(
        "User query received: Age=%s, Annual_Income=%s, Spending_Score=%s",
        Age, Annual_Income, Spending_Score,
    )

    try:
        prediction_input = pd.DataFrame([[Age, Annual_Income, Spending_Score]], columns=FEATURE_COLS)
        prediction_input_scaled = scaler.transform(prediction_input)
        cluster = kmeans_model.predict(prediction_input_scaled)[0]
        logger.info("Prediction result: Cluster=%s", cluster)

        profile = cluster_profile.loc[cluster]

        st.subheader("Prediction Result:")
        st.metric(f"Cluster {cluster}", profile['Segment'])
        st.caption(f"Segment average age: {profile['Age']:.0f}")

        # Echo back the submitted details so the user can confirm what was predicted on
        st.subheader("Submitted Customer Details:")
        submitted_details = pd.DataFrame({
            "Field": ["Age", "Annual Income (in $1,000s)", "Spending Score (1-100)"],
            "Value": [str(Age), str(Annual_Income), str(Spending_Score)],
        })
        st.table(submitted_details.set_index("Field"))

        # Redraw the segment plot with this customer's point highlighted, so the chart
        # reflects the input just submitted rather than a static pre-rendered image
        fig, ax = plt.subplots()
        sns.scatterplot(x='Annual_Income', y='Spending_Score', hue='Cluster', data=df, palette='colorblind', ax=ax)

        # Cluster centers live in scaled feature space; bring them back to original units
        # so they land on the same axes as the plotted (unscaled) Income/Spending data
        centers_orig = scaler.inverse_transform(kmeans_model.cluster_centers_)
        income_idx, spend_idx = FEATURE_COLS.index('Annual_Income'), FEATURE_COLS.index('Spending_Score')
        ax.scatter(centers_orig[:, income_idx], centers_orig[:, spend_idx],
                   c='black', s=200, alpha=0.5, marker='X', label='Centroids')
        ax.scatter(Annual_Income, Spending_Score, c='red', s=250, marker='*', label='This Customer')
        ax.set_title('Customer Segments (Income vs. Spending view)')
        ax.legend()
        st.pyplot(fig)
    except Exception:
        logger.exception("Prediction failed for the submitted query.")
        st.error("Something went wrong while finding this customer's segment. Please check your inputs and try again.")

st.write("""
The chart shows all mall customers grouped into segments by income and spending behavior.
Submit the form above to see where this customer lands, marked with a red star.
""")
