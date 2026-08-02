import streamlit as st

from app_utils import (
    cluster_label_map,
    crime_cols,
    display_feature_order,
    load_data,
    rank_clusters_by_crime,
    with_composition,
    with_homicide,
)

st.set_page_config(page_title="Neighbourhood Ranking", page_icon="🏘️", layout="wide")

st.title("🏘️ Neighbourhood Ranking")
st.markdown("See how one neighbourhood compares to the rest of the city on a chosen safety metric.")

df = load_data()
cols = crime_cols(df)
comp = with_composition(df)

metrics = {
    "Total Crime per Capita": comp[cols].sum(axis=1) / comp["Population"],
    "Violent Crime per Capita": comp["Violent"] / comp["Population"],
    "Property Crime per Capita": comp["Property"] / comp["Population"],
    "Other Crime per Capita": comp["Other"] / comp["Population"],
    "Police Critical Calls per Capita": comp["Police_Critical"] / comp["Population"],
    "Police High Calls per Capita": comp["Police_High"] / comp["Population"],
    "Police Medium Calls per Capita": comp["Police_Medium"] / comp["Population"],
}
metric_df = comp[["Neighbourhood", "cluster", "Population"]].copy()
for name, series in metrics.items():
    metric_df[name] = series

cluster_rank = rank_clusters_by_crime(df)
label_map = cluster_label_map(cluster_rank)

with st.sidebar:
    st.header("Selection")
    neighbourhood = st.selectbox(
        "Neighbourhood", options=sorted(metric_df["Neighbourhood"].unique())
    )
    metric = st.selectbox("Safety metric (lower = safer)", options=list(metrics.keys()))

ranked = metric_df.sort_values(metric).reset_index(drop=True)
ranked["Rank"] = ranked.index + 1
total = len(ranked)

row = ranked[ranked["Neighbourhood"] == neighbourhood].iloc[0]
percentile = 100 * (1 - (row["Rank"] - 1) / (total - 1))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Population", f"{int(row['Population']):,}")
col2.metric(metric, f"{row[metric]:.4f}")
col3.metric("Rank (safest = 1)", f"{int(row['Rank'])} / {total}")
col4.metric("Safer than", f"{percentile:.0f}% of neighbourhoods")
st.caption(f"Cluster: {label_map[row['cluster']]}")

st.subheader("Crime Incidents (2018–2025)")
st.caption(
    "Raw incident counts for every feature used to build the clusters. "
    "Bar length is relative to the highest count for that crime type city-wide. "
    "Rank is this neighbourhood's position out of all neighbourhoods for that crime type "
    "(#1 = most incidents); its standout crime type is highlighted in red."
)

work = with_homicide(df)
feature_order = display_feature_order(df)
max_vals = work[feature_order].max()
feature_ranks = work[feature_order].rank(method="min", ascending=False).astype(int)

row_idx = work.index[work["Neighbourhood"] == neighbourhood][0]
row_values = work.loc[row_idx, feature_order]
row_ranks = feature_ranks.loc[row_idx]
standout_feature = row_ranks.idxmin()

for feat in feature_order:
    value = int(row_values[feat])
    denom = max_vals[feat] if max_vals[feat] else 1
    ratio = min(max(value / denom, 0.0), 1.0)
    rank = int(row_ranks[feat])
    is_standout = feat == standout_feature and value > 0

    c1, c2, c3, c4 = st.columns([3, 4, 1, 1])
    c1.markdown(feat)
    c2.progress(ratio)
    c3.markdown(f"<div style='text-align:right'>{value}</div>", unsafe_allow_html=True)
    rank_color = "#d73027" if is_standout else "#888"
    c4.markdown(f"<div style='text-align:right;color:{rank_color}'>#{rank}</div>", unsafe_allow_html=True)

st.subheader("Nearby in Rank")
window = 5
lo = max(0, int(row["Rank"]) - 1 - window)
hi = min(total, int(row["Rank"]) + window)
nearby = ranked.iloc[lo:hi][["Rank", "Neighbourhood", "cluster", metric]].copy()
nearby["cluster"] = nearby["cluster"].map(label_map)
nearby.rename(columns={"cluster": "Cluster"}, inplace=True)
nearby["Selected"] = nearby["Neighbourhood"].apply(lambda n: "★" if n == neighbourhood else "")
st.dataframe(
    nearby[["Selected", "Rank", "Neighbourhood", "Cluster", metric]],
    hide_index=True,
    use_container_width=True,
    column_config={
        "Neighbourhood": st.column_config.TextColumn("Neighbourhood", width="large"),
    },
)
