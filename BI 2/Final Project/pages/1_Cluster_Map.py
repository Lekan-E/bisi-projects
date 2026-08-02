import folium
import streamlit as st
import streamlit.components.v1 as components

from app_utils import (
    cluster_color_map,
    cluster_label_map,
    load_geodata,
    rank_clusters_by_crime,
)

st.set_page_config(page_title="Cluster Map", page_icon="🗺️", layout="wide")

st.title("🗺️ Cluster Map")
st.markdown(
    "Every neighbourhood polygon, coloured on a green (safest) to dark maroon "
    "(highest crime) gradient by its assigned safety cluster."
)

gdf = load_geodata()
cluster_rank = rank_clusters_by_crime(gdf)
color_map = cluster_color_map(cluster_rank)
label_map = cluster_label_map(cluster_rank)

with st.sidebar:
    st.header("Filters")
    selected_labels = st.multiselect(
        "Clusters to show",
        options=[label_map[cid] for cid in cluster_rank],
        default=[label_map[cid] for cid in cluster_rank],
    )
    label_to_cid = {v: k for k, v in label_map.items()}
    selected_clusters = {label_to_cid[label] for label in selected_labels}

if not selected_clusters:
    st.warning("Select at least one cluster from the sidebar to display the map.")
    st.stop()

gdf_view = gdf[gdf["cluster"].isin(selected_clusters)]

m = folium.Map(
    location=[45.4215, -75.6972],
    zoom_start=11,
    tiles="http://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}",
    attr="Google Maps",
)

folium.GeoJson(
    gdf_view.__geo_interface__,
    style_function=lambda feat: {
        "fillColor": color_map.get(feat["properties"]["cluster"], "#cccccc"),
        "color": "white",
        "weight": 1,
        "fillOpacity": 0.70,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["Neighbourhood", "cluster", "Population"],
        aliases=["Neighbourhood", "Cluster", "Population"],
        sticky=True,
        style="font-size:13px;",
    ),
    popup=folium.GeoJsonPopup(
        fields=["Neighbourhood", "cluster", "Population",
                "Murder 1st Dgree", "Murder 2nd Dgree", "Shootings", "Auto Theft"],
        aliases=["Neighbourhood", "Cluster", "Population",
                 "Murder 1st Dgree", "Murder 2nd Dgree", "Shootings", "Auto Theft"],
    ),
).add_to(m)

legend_html = """
<div style="position:fixed; top:30px; right:30px; z-index:1000;
            background:white; padding:12px 16px; border-radius:8px;
            box-shadow:0 2px 6px rgba(0,0,0,0.3); font-family:Arial; font-size:13px;">
  <b>Safety Cluster</b><br>
  <span style="font-size:11px;color:#666;">ranked by avg total crime</span><br><br>
"""
for i, cid in enumerate(cluster_rank):
    rank_label = "Lowest crime" if i == 0 else ("Highest crime" if i == len(cluster_rank) - 1 else "")
    legend_html += (
        f'<i style="background:{color_map[cid]};width:14px;height:14px;'
        f'display:inline-block;margin-right:6px;border-radius:2px;border:1px solid #ccc;"></i>'
        f"Cluster {cid}"
        + (f' <span style="color:#888;font-size:11px;">— {rank_label}</span>' if rank_label else "")
        + "<br>"
    )
legend_html += "</div>"
m.get_root().html.add_child(folium.Element(legend_html))

components.html(m._repr_html_(), height=700, scrolling=False)

st.caption(f"Showing {len(gdf_view)} of {len(gdf)} neighbourhoods.")
