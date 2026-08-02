import streamlit as st

from app_utils import load_data

# page_title/page_icon set the browser tab; the filename "Home" sets the sidebar nav label
st.set_page_config(
    page_title="Ottawa Neighbourhood Safety",
    page_icon="🧭",
    layout="wide",
)

st.title("🧭 Ottawa Neighbourhood Safety")
st.markdown(
    """
BY: Olamilekan Elegbede, 041273681

According to multiple reports, Ottawa is considered one of the safest cities to reside/live in 2026.
Regardless of this fact, Ottawa has seen a fair share of criminal and safety incidents. For
anyone looking to settle in the country’s capital, this project will serve as guide to answer ‘Which
Ottawa neighbourhood is safe to settle in?’

The objective of this project is to use machine learning to group Ottawa neighborhoods based on
crime and safety indicators to enable decision making for new and current residents and city
officials.

Explore how Ottawa's 104 neighbourhoods group into distinct safety profiles, based on
crime, hate crime, homicide, shooting, and police-response data (2018-2025).

Use the sidebar to navigate:
- **Cluster Map** — geographic view of every neighbourhood, coloured by safety cluster
- **Cluster Info** — what makes each cluster's crime profile distinct
- **Neighbourhood Ranking** — see how one neighbourhood compares to the rest of the city
"""
)

# load_data() is cached (see app_utils.py), so this only reads the CSV once per session
df = load_data()
n_clusters = df["cluster"].nunique()

# quick headline stats for the landing page
col1, col2, col3 = st.columns(3)
col1.metric("Neighbourhoods", df["Neighbourhood"].nunique())
col2.metric("Safety Clusters", n_clusters)
col3.metric("Total Population", f"{int(df['Population'].sum()):,}")
