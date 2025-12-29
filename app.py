"""
Zoem Kaart Web Demo - Dutch Pollinator Biodiversity Explorer
Author: Mohamed Z. Hatim, Wageningen University & Research
A web-based demonstration of the Zoem Kaart QGIS plugin capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Optional, Dict, List, Tuple
import math

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Zoem Kaart - Dutch Pollinator Explorer",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dutch Red List status mapping (simplified for demo)
RED_LIST_STATUS = {
    "Critically Endangered": {"color": "#d62728", "priority": 5},
    "Endangered": {"color": "#ff7f0e", "priority": 4},
    "Vulnerable": {"color": "#ffbb00", "priority": 3},
    "Near Threatened": {"color": "#98df8a", "priority": 2},
    "Least Concern": {"color": "#2ca02c", "priority": 1},
    "Data Deficient": {"color": "#7f7f7f", "priority": 0},
    "Unknown": {"color": "#c7c7c7", "priority": 0},
}

# GBIF taxon keys for pollinator groups
TAXON_KEYS = {
    "All Pollinators": None,  # Will combine all below
    "Wild Bees (Apoidea)": 4334,  # Apoidea superfamily
    "Butterflies (Lepidoptera)": 797,  # Lepidoptera order
    "Hoverflies (Syrphidae)": 6920,  # Syrphidae family
}

# Netherlands bounding box
NL_BOUNDS = {
    "min_lat": 50.75,
    "max_lat": 53.47,
    "min_lon": 3.37,
    "max_lon": 7.21,
}

# Some example Dutch Red List species (wild bees) - in real app, load from JSON
DUTCH_RED_LIST_BEES = {
    "Bombus distinguendus": "Critically Endangered",
    "Bombus muscorum": "Endangered",
    "Bombus ruderarius": "Endangered",
    "Bombus sylvarum": "Vulnerable",
    "Bombus humilis": "Vulnerable",
    "Bombus veteranus": "Vulnerable",
    "Bombus pascuorum": "Least Concern",
    "Bombus terrestris": "Least Concern",
    "Bombus lapidarius": "Least Concern",
    "Bombus pratorum": "Least Concern",
    "Bombus hortorum": "Least Concern",
    "Apis mellifera": "Least Concern",
    "Andrena fulva": "Least Concern",
    "Andrena haemorrhoa": "Least Concern",
    "Osmia bicornis": "Least Concern",
    "Anthophora plumipes": "Least Concern",
    "Colletes daviesanus": "Least Concern",
    "Halictus rubicundus": "Near Threatened",
    "Lasioglossum malachurum": "Least Concern",
    "Megachile willughbiella": "Least Concern",
}

DUTCH_RED_LIST_BUTTERFLIES = {
    "Phengaris alcon": "Critically Endangered",
    "Melitaea cinxia": "Critically Endangered",
    "Argynnis niobe": "Endangered",
    "Pyrgus malvae": "Endangered",
    "Plebeius argus": "Vulnerable",
    "Issoria lathonia": "Vulnerable",
    "Aglais io": "Least Concern",
    "Vanessa atalanta": "Least Concern",
    "Pieris brassicae": "Least Concern",
    "Pieris rapae": "Least Concern",
    "Gonepteryx rhamni": "Least Concern",
    "Polygonia c-album": "Least Concern",
    "Pararge aegeria": "Least Concern",
    "Maniola jurtina": "Least Concern",
    "Aphantopus hyperantus": "Least Concern",
    "Polyommatus icarus": "Least Concern",
    "Celastrina argiolus": "Least Concern",
}

# Combine all Red List data
ALL_RED_LIST = {**DUTCH_RED_LIST_BEES, **DUTCH_RED_LIST_BUTTERFLIES}


# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_gbif_occurrences(
    taxon_key: Optional[int],
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    year_start: int,
    year_end: int,
    limit: int = 300
) -> pd.DataFrame:
    """Fetch occurrence data from GBIF API."""
    
    base_url = "https://api.gbif.org/v1/occurrence/search"
    
    all_records = []
    
    # If "All Pollinators", fetch each group
    if taxon_key is None:
        taxon_keys = [4334, 797, 6920]  # Bees, Butterflies, Hoverflies
    else:
        taxon_keys = [taxon_key]
    
    for tk in taxon_keys:
        params = {
            "taxonKey": tk,
            "country": "NL",
            "hasCoordinate": "true",
            "hasGeospatialIssue": "false",
            "decimalLatitude": f"{min_lat},{max_lat}",
            "decimalLongitude": f"{min_lon},{max_lon}",
            "year": f"{year_start},{year_end}",
            "limit": limit // len(taxon_keys),
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for record in data.get("results", []):
                all_records.append({
                    "species": record.get("species", "Unknown"),
                    "genus": record.get("genus", "Unknown"),
                    "family": record.get("family", "Unknown"),
                    "order": record.get("order", "Unknown"),
                    "lat": record.get("decimalLatitude"),
                    "lon": record.get("decimalLongitude"),
                    "date": record.get("eventDate", ""),
                    "year": record.get("year"),
                    "basis": record.get("basisOfRecord", ""),
                    "dataset": record.get("datasetName", ""),
                    "gbif_id": record.get("gbifID"),
                })
        except Exception as e:
            st.warning(f"Error fetching data for taxon {tk}: {e}")
    
    df = pd.DataFrame(all_records)
    
    # Remove records without coordinates or species
    if not df.empty:
        df = df.dropna(subset=["lat", "lon", "species"])
    
    return df


def enrich_with_red_list(df: pd.DataFrame) -> pd.DataFrame:
    """Add Red List status to observations."""
    if df.empty:
        return df
    
    df = df.copy()  # Avoid SettingWithCopyWarning
    df["red_list_status"] = df["species"].map(ALL_RED_LIST).fillna("Unknown")
    df["status_color"] = df["red_list_status"].map(
        lambda x: RED_LIST_STATUS.get(x, {}).get("color", "#c7c7c7")
    )
    df["conservation_priority"] = df["red_list_status"].map(
        lambda x: RED_LIST_STATUS.get(x, {}).get("priority", 0)
    )
    
    return df


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_shannon_diversity(df: pd.DataFrame) -> float:
    """Calculate Shannon diversity index."""
    if df.empty or "species" not in df.columns:
        return 0.0
    
    species_counts = df["species"].value_counts()
    total = species_counts.sum()
    
    if total == 0:
        return 0.0
    
    proportions = species_counts / total
    shannon = -sum(p * np.log(p) for p in proportions if p > 0)
    
    return round(shannon, 3)


def calculate_simpson_diversity(df: pd.DataFrame) -> float:
    """Calculate Simpson diversity index (1 - D)."""
    if df.empty or "species" not in df.columns:
        return 0.0
    
    species_counts = df["species"].value_counts()
    total = species_counts.sum()
    
    if total <= 1:
        return 0.0
    
    simpson_d = sum(n * (n - 1) for n in species_counts) / (total * (total - 1))
    
    return round(1 - simpson_d, 3)


def calculate_pielou_evenness(df: pd.DataFrame) -> float:
    """Calculate Pielou's evenness index."""
    if df.empty or "species" not in df.columns:
        return 0.0
    
    n_species = df["species"].nunique()
    if n_species <= 1:
        return 0.0
    
    shannon = calculate_shannon_diversity(df)
    max_diversity = np.log(n_species)
    
    if max_diversity == 0:
        return 0.0
    
    return round(shannon / max_diversity, 3)


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_map(df: pd.DataFrame, center_lat: float, center_lon: float, zoom: int = 8) -> folium.Map:
    """Create an interactive Folium map with observations."""
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="cartodbpositron"
    )
    
    if df.empty:
        return m
    
    # Add marker cluster
    marker_cluster = MarkerCluster(name="Observations").add_to(m)
    
    for _, row in df.iterrows():
        popup_html = f"""
        <b>{row['species']}</b><br>
        Family: {row['family']}<br>
        Red List: <span style="color:{row['status_color']}">{row['red_list_status']}</span><br>
        Date: {row.get('date', 'Unknown')[:10] if row.get('date') else 'Unknown'}<br>
        """
        
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            popup=folium.Popup(popup_html, max_width=250),
            color=row["status_color"],
            fill=True,
            fillColor=row["status_color"],
            fillOpacity=0.7,
        ).add_to(marker_cluster)
    
    # Add heatmap layer
    if len(df) > 10:
        heat_data = df[["lat", "lon"]].values.tolist()
        HeatMap(heat_data, name="Density Heatmap", show=False).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    return m


def create_red_list_chart(df: pd.DataFrame) -> go.Figure:
    """Create a pie chart of Red List status distribution."""
    
    if df.empty:
        return go.Figure()
    
    status_counts = df["red_list_status"].value_counts()
    
    colors = [RED_LIST_STATUS.get(s, {}).get("color", "#c7c7c7") for s in status_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        marker_colors=colors,
        hole=0.4,
        textinfo="percent+label",
        textposition="outside",
    )])
    
    fig.update_layout(
        title="Conservation Status Distribution",
        showlegend=False,
        margin=dict(t=50, b=20, l=20, r=20),
        height=300,
    )
    
    return fig


def create_species_bar_chart(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create a bar chart of most observed species."""
    
    if df.empty:
        return go.Figure()
    
    species_counts = df["species"].value_counts().head(top_n)
    
    # Get colors based on Red List status
    colors = []
    for species in species_counts.index:
        status = ALL_RED_LIST.get(species, "Unknown")
        colors.append(RED_LIST_STATUS.get(status, {}).get("color", "#c7c7c7"))
    
    fig = go.Figure(data=[go.Bar(
        x=species_counts.values,
        y=species_counts.index,
        orientation="h",
        marker_color=colors,
    )])
    
    fig.update_layout(
        title=f"Top {top_n} Most Observed Species",
        xaxis_title="Observations",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        margin=dict(t=50, b=40, l=200, r=20),
        height=400,
    )
    
    return fig


def create_temporal_chart(df: pd.DataFrame) -> go.Figure:
    """Create a temporal pattern chart."""
    
    if df.empty or "year" not in df.columns:
        return go.Figure()
    
    yearly_counts = df.groupby("year").size().reset_index(name="count")
    
    fig = go.Figure(data=[go.Scatter(
        x=yearly_counts["year"],
        y=yearly_counts["count"],
        mode="lines+markers",
        line=dict(color="#2c5282", width=2),
        marker=dict(size=8),
    )])
    
    fig.update_layout(
        title="Observations Over Time",
        xaxis_title="Year",
        yaxis_title="Number of Observations",
        margin=dict(t=50, b=40, l=60, r=20),
        height=250,
    )
    
    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("🐝 Zoem Kaart")
    st.markdown("### Dutch Pollinator Biodiversity Explorer")
    st.markdown(
        "*Web demonstration of the Zoem Kaart QGIS plugin — "
        "Developed by [Mohamed Z. Hatim](mailto:mohamed.hatim@wur.nl), "
        "Wageningen University & Research*"
    )
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🔍 Query Parameters")
        
        # Taxon selection
        taxon_group = st.selectbox(
            "Pollinator Group",
            options=list(TAXON_KEYS.keys()),
            index=0,
        )
        
        # Date range
        st.subheader("Time Period")
        col1, col2 = st.columns(2)
        with col1:
            year_start = st.number_input("From Year", min_value=1900, max_value=2025, value=2015)
        with col2:
            year_end = st.number_input("To Year", min_value=1900, max_value=2025, value=2024)
        
        # Area selection
        st.subheader("Geographic Area")
        area_option = st.radio(
            "Select area",
            ["All Netherlands", "Custom Bounds"],
            index=0,
        )
        
        if area_option == "Custom Bounds":
            col1, col2 = st.columns(2)
            with col1:
                min_lat = st.number_input("Min Latitude", value=51.5, min_value=50.0, max_value=54.0)
                min_lon = st.number_input("Min Longitude", value=4.0, min_value=3.0, max_value=8.0)
            with col2:
                max_lat = st.number_input("Max Latitude", value=52.5, min_value=50.0, max_value=54.0)
                max_lon = st.number_input("Max Longitude", value=6.0, min_value=3.0, max_value=8.0)
        else:
            min_lat, max_lat = NL_BOUNDS["min_lat"], NL_BOUNDS["max_lat"]
            min_lon, max_lon = NL_BOUNDS["min_lon"], NL_BOUNDS["max_lon"]
        
        # Result limit
        limit = st.slider("Max observations", min_value=50, max_value=500, value=200, step=50)
        
        st.markdown("---")
        
        # Fetch button
        fetch_button = st.button("🔎 Fetch Data", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "This is a web demonstration of the **Zoem Kaart** QGIS plugin, "
            "which provides comprehensive tools for analyzing Dutch pollinator biodiversity."
        )
        st.markdown(
            "**Full QGIS plugin features:**\n"
            "- Multi-source integration (GBIF, iNaturalist, Waarneming.nl, FLORON)\n"
            "- 8 processing algorithms\n"
            "- Species traits database\n"
            "- Nectar/pollen index\n"
            "- Export to multiple formats"
        )
        st.markdown("---")
        st.markdown("**Data source:** [GBIF](https://www.gbif.org/)")
    
    # Main content area
    if fetch_button or "data" in st.session_state:
        if fetch_button:
            with st.spinner("Fetching data from GBIF..."):
                taxon_key = TAXON_KEYS[taxon_group]
                df = fetch_gbif_occurrences(
                    taxon_key=taxon_key,
                    min_lat=min_lat,
                    max_lat=max_lat,
                    min_lon=min_lon,
                    max_lon=max_lon,
                    year_start=year_start,
                    year_end=year_end,
                    limit=limit,
                )
                df = enrich_with_red_list(df)
                st.session_state["data"] = df
        else:
            df = st.session_state["data"]
        
        if df.empty:
            st.warning("No observations found for the selected parameters. Try adjusting your filters.")
            return
        
        # Summary metrics
        st.subheader("📊 Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Observations", f"{len(df):,}")
        with col2:
            st.metric("Species", f"{df['species'].nunique():,}")
        with col3:
            st.metric("Shannon Index", calculate_shannon_diversity(df))
        with col4:
            st.metric("Simpson Index", calculate_simpson_diversity(df))
        with col5:
            threatened = df[df["red_list_status"].isin(["Critically Endangered", "Endangered", "Vulnerable"])]["species"].nunique()
            st.metric("Threatened Species", threatened)
        
        st.markdown("---")
        
        # Map and charts
        col_map, col_charts = st.columns([3, 2])
        
        with col_map:
            st.subheader("🗺️ Distribution Map")
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2
            m = create_map(df, center_lat, center_lon)
            st_folium(m, width=700, height=500)
        
        with col_charts:
            st.subheader("🔴 Conservation Status")
            st.plotly_chart(create_red_list_chart(df), use_container_width=True)
            
            st.plotly_chart(create_temporal_chart(df), use_container_width=True)
        
        # Species chart (full width)
        st.subheader("🦋 Species Observations")
        st.plotly_chart(create_species_bar_chart(df, top_n=15), use_container_width=True)
        
        # Data table
        with st.expander("📋 View Raw Data"):
            display_cols = ["species", "family", "red_list_status", "lat", "lon", "date", "year"]
            # FIX: Sort before selecting columns
            df_sorted = df.sort_values("conservation_priority", ascending=False)
            st.dataframe(
                df_sorted[display_cols],
                use_container_width=True,
            )
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="zoem_kaart_export.csv",
                mime="text/csv",
            )
    
    else:
        # Welcome message when no data loaded
        st.info("👈 Configure your query parameters in the sidebar and click **Fetch Data** to explore Dutch pollinator biodiversity.")
        
        # Show example map of Netherlands
        st.subheader("🗺️ Study Area: The Netherlands")
        m = folium.Map(
            location=[52.1, 5.3],
            zoom_start=7,
            tiles="cartodbpositron"
        )
        # Add Netherlands boundary indicator
        folium.Rectangle(
            bounds=[[NL_BOUNDS["min_lat"], NL_BOUNDS["min_lon"]], 
                    [NL_BOUNDS["max_lat"], NL_BOUNDS["max_lon"]]],
            color="#2c5282",
            fill=True,
            fillOpacity=0.1,
        ).add_to(m)
        st_folium(m, width=700, height=400)
        
        # Feature highlights
        st.markdown("---")
        st.subheader("✨ Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🐝 Multi-Taxon Support")
            st.markdown("Explore wild bees, butterflies, and hoverflies from GBIF data.")
        
        with col2:
            st.markdown("#### 📊 Diversity Metrics")
            st.markdown("Calculate Shannon, Simpson, and Pielou's evenness indices.")
        
        with col3:
            st.markdown("#### 🔴 Conservation Status")
            st.markdown("Automatic enrichment with Dutch Red List classifications.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🐝 Zoem Kaart Web Demo | "
        "Full QGIS plugin available for advanced analysis | "
        "© 2025 Mohamed Z. Hatim, Wageningen University & Research"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
