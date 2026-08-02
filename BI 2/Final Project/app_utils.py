"""Shared data loading and styling helpers for the Streamlit app pages."""
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.colors as mcolors
import streamlit as st

DATA_PATH = Path(__file__).parent / "data" / "processed" / "Clustered_Data.csv"

META_COLS = {
    "Neighbourhood", "cluster", "Population",
    "Police_Critical", "Police_High", "Police_Medium", "geometry",
}
VIOLENT_COLS = [
    "Assaults", "Manslaughter", "Murder 1st Dgree", "Murder 2nd Dgree",
    "Shootings", "Violations Causing Death",
    "Other Violations Involving Violence Or The Threat Of Violence",
    "Sexual Violations", "Violations Resulting In The Deprivation Of Freedom",
    "Attempting The Commission Of A Capital Crime", "Offensive Weapons",
]
PROPERTY_COLS = [
    "Theft $5000 and Under", "Theft Over $5000", "Break and Enter",
    "Auto Theft", "Bike Theft", "Mischief", "Arson",
    "Possession / Trafficking Stolen Goods",
]

# green -> yellow-green -> yellow -> orange -> red -> dark maroon, safest to highest-crime
CRIME_GRADIENT = mcolors.LinearSegmentedColormap.from_list(
    "crime_gradient", ["#006837", "#78c679", "#ffffb2", "#fd8d3c", "#f03b20", "#67000d"]
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["cluster"] = df["cluster"].astype(int)
    return df


@st.cache_data
def load_geodata() -> gpd.GeoDataFrame:
    df = load_data()
    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.GeoSeries.from_wkt(df["geometry"]),
        crs="EPSG:4326",
    )


def crime_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def other_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in crime_cols(df) if c not in VIOLENT_COLS and c not in PROPERTY_COLS]


def rank_clusters_by_crime(df: pd.DataFrame) -> list[int]:
    """Cluster ids ordered from lowest to highest average total crime."""
    cols = crime_cols(df)
    return (
        df.assign(total_crime=df[cols].sum(axis=1))
          .groupby("cluster")["total_crime"]
          .mean()
          .sort_values()
          .index.tolist()
    )


def cluster_color_map(cluster_rank: list[int]) -> dict[int, str]:
    """Map each cluster id to a hex colour along the safest -> highest-crime gradient."""
    n = len(cluster_rank)
    palette = [mcolors.to_hex(CRIME_GRADIENT(i / max(n - 1, 1))) for i in range(n)]
    return {cid: palette[i] for i, cid in enumerate(cluster_rank)}


def cluster_label_map(cluster_rank: list[int]) -> dict[int, str]:
    """e.g. {cid: 'Cluster 3 (Safest)'} / '(Highest crime)' for the endpoints."""
    labels = {}
    for i, cid in enumerate(cluster_rank):
        suffix = ""
        if i == 0:
            suffix = " (Safest)"
        elif i == len(cluster_rank) - 1:
            suffix = " (Highest crime)"
        labels[cid] = f"Cluster {cid}{suffix}"
    return labels


def with_composition(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Violent / Property / Other incident-count columns."""
    out = df.copy()
    out["Violent"] = out[VIOLENT_COLS].sum(axis=1)
    out["Property"] = out[PROPERTY_COLS].sum(axis=1)
    out["Other"] = out[other_cols(df)].sum(axis=1)
    return out


def with_homicide(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a combined 'Homicide' column (Manslaughter + Murder 1st/2nd Degree)."""
    out = df.copy()
    out["Homicide"] = out[["Manslaughter", "Murder 1st Dgree", "Murder 2nd Dgree"]].sum(axis=1)
    return out


def display_feature_order(df: pd.DataFrame) -> list[str]:
    """Crime features for per-neighbourhood display: homicide columns combined into
    one row (alphabetised with the rest), 'Other' pinned last as the catch-all bucket."""
    raw = crime_cols(df)
    named = [c for c in raw if c not in ("Manslaughter", "Murder 1st Dgree", "Murder 2nd Dgree", "Other")]
    order = sorted(named + ["Homicide"])
    if "Other" in raw:
        order.append("Other")
    return order


@st.cache_data
def feature_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score of every crime feature across all neighbourhoods (city-wide baseline)."""
    from scipy.stats import zscore

    cols = crime_cols(df)
    z = df.copy()
    z[cols] = z[cols].apply(zscore)
    return z
