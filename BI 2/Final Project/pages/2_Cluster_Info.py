import altair as alt
import pandas as pd
import streamlit as st

from app_utils import (
    cluster_label_map,
    crime_cols,
    feature_zscores,
    load_data,
    rank_clusters_by_crime,
    with_composition,
)

st.set_page_config(page_title="Cluster Info", page_icon="📊", layout="wide")

st.title("📊 Cluster Info")
st.markdown("What makes each safety cluster's crime profile distinct.")

df = load_data()
cluster_rank = rank_clusters_by_crime(df)
label_map = cluster_label_map(cluster_rank)

with st.sidebar:
    st.header("Cluster")
    selected_label = st.selectbox("Choose a cluster", options=[label_map[c] for c in cluster_rank])
    label_to_cid = {v: k for k, v in label_map.items()}
    cid = label_to_cid[selected_label]

cluster_df = df[df["cluster"] == cid]
cols = crime_cols(df)

col1, col2, col3 = st.columns(3)
col1.metric("Neighbourhoods", len(cluster_df))
col2.metric("Total Population", f"{int(cluster_df['Population'].sum()):,}")
col3.metric(
    "Avg crime incidents / neighbourhood",
    f"{cluster_df[cols].sum(axis=1).mean():.0f}",
)

st.subheader("Crime Type Composition")
comp = with_composition(df)
share = comp.groupby("cluster")[["Violent", "Property", "Other"]].sum()
share = share.div(share.sum(axis=1), axis=0) * 100
share_row = share.loc[cid].reset_index()
share_row.columns = ["Category", "Share (%)"]
share_row["Label"] = share_row["Share (%)"].map(lambda v: f"{v:.1f}%")

comp_bars = (
    alt.Chart(share_row)
    .mark_bar()
    .encode(
        x=alt.X("Share (%):Q"),
        y=alt.Y("Category:N", sort=["Violent", "Property", "Other"]),
        color=alt.Color(
            "Category:N",
            scale=alt.Scale(
                domain=["Violent", "Property", "Other"],
                range=["#d73027", "#4575b4", "#91bfdb"],
            ),
            legend=None,
        ),
        tooltip=["Category", alt.Tooltip("Share (%):Q", format=".1f")],
    )
)
comp_labels = comp_bars.mark_text(align="left", dx=4).encode(
    text="Label:N",
    color=alt.value("white"),
)
st.altair_chart(comp_bars + comp_labels, use_container_width=True)

st.subheader("Distinguishing Features")
st.caption("How this cluster's average compares to the city-wide average for each crime type (z-score).")
z = feature_zscores(df)
cluster_z = z.groupby("cluster")[cols].mean().loc[cid].sort_values(ascending=False)

top_n = 6
combined = pd.concat([cluster_z.head(top_n), cluster_z.tail(top_n)]).drop_duplicates()
z_df = combined.reset_index()
z_df.columns = ["Feature", "Z-score"]

z_chart = (
    alt.Chart(z_df)
    .mark_bar()
    .encode(
        x=alt.X("Z-score:Q"),
        y=alt.Y("Feature:N", sort=z_df.sort_values("Z-score")["Feature"].tolist()),
        color=alt.condition(alt.datum["Z-score"] > 0, alt.value("#f03b20"), alt.value("#2166ac")),
        tooltip=["Feature", alt.Tooltip("Z-score:Q", format=".2f")],
    )
)
st.altair_chart(z_chart, use_container_width=True)

st.subheader("Neighbourhoods in this Cluster")
st.dataframe(
    cluster_df[["Neighbourhood", "Population"]].sort_values("Population", ascending=False),
    hide_index=True,
    use_container_width=True,
)
