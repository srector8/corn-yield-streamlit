# -*- coding: utf-8 -*-
"""Corn Yield Dashboard

Streamlit app to explore US Corn Belt yield data over time.
"""

import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from PIL import Image


# Preprocessing
@st.cache_data
def preprocess_data(file_path, shapefile_path):
    # Load and merge CSV with shapefile, simplify geometry
    avg_df = pd.read_csv(file_path)
    shp_gdf = gpd.read_file(shapefile_path)

    # Merge on county IDs
    merged_gdf = shp_gdf.merge(avg_df, left_on='ID_2', right_on='id2')

    # Simplify geometries for faster plotting
    merged_gdf['geometry'] = merged_gdf['geometry'].simplify(0.01)

    # Extract the list of years in the dataset
    years = sorted(avg_df['year'].unique())

    return merged_gdf, years

def make_colorbar():
    # Horizontal gradient: Blue → Purple → Red
    width, height = 256, 20
    colorbar = Image.new("RGBA", (width, height))
    for x in range(width):
        r = int(255 * (x / (width - 1)))
        g = 120
        b = int(255 * (1 - x / (width - 1)))
        for y in range(height):
            colorbar.putpixel((x, y), (r, g, b, 255))
    return colorbar

def kde_1d(data, num_points=200, bandwidth=None):
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    # Silverman's rule of thumb if no bandwidth provided
    if bandwidth is None:
        bandwidth = 1.06 * data.std() * len(data) ** (-1/5)

    xs = np.linspace(data.min(), data.max(), num_points)
    density = np.zeros_like(xs)

    for x in data:
        density += np.exp(-0.5 * ((xs - x) / bandwidth) ** 2)

    density /= (len(data) * bandwidth * np.sqrt(2 * np.pi))
    return xs, density

def plot_choropleth(gdf, feature, year):
    # Filter for selected year
    gdf_year = gdf[gdf['year'] == year].copy()

    # Feature-specific scaling (important)
    min_val = gdf_year[feature].min()
    max_val = gdf_year[feature].max()

    gdf_year["__norm"] = (gdf_year[feature] - min_val) / (max_val - min_val + 1e-9)
    gdf_year["color"] = gdf_year["__norm"].apply(lambda x: [int(255 * x), 120, int(255 * (1 - x)), 180])

    gdf_year["value_disp"] = gdf_year[feature].round(4)
    
    layer = pdk.Layer(
        "GeoJsonLayer",
        gdf_year,
        opacity=0.75,
        stroked=True,
        filled=True,
        get_fill_color="color",
        get_line_color=[0, 0, 0],
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=40.0,
        longitude=-93.0,
        zoom=4,
        pitch=0,
    )
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>{NAME_2}</b><br>" + feature + ": {value_disp}",
            "style": {"backgroundColor": "white", "color": "black"}
        },
    )

    # Layout: map left, scale right
    left, right = st.columns([4, 1])

    with left:
        st.pydeck_chart(r)
        st.caption(f"Showing **{feature}** for **{year}**")

        st.write("### Color Scale")
        st.image(make_colorbar(), use_column_width=True)
        st.write(f"{min_val:.4f} ⟶ {max_val:.4f}")

    with right:
        st.write("### Values")
        st.metric("Minimum", f"{min_val:.4f}")
        st.metric("Maximum", f"{max_val:.4f}")
        st.metric("Mean", f"{gdf_year[feature].mean():.4f}")
        st.metric("Median", f"{gdf_year[feature].median():.4f}")
        st.metric("Std Dev", f"{gdf_year[feature].std():.4f}")

    st.write(f"### Distribution of {feature} in {year}")
    
    vals = gdf_year[feature].dropna()
    xs, density = kde_1d(vals)
    
    density_df = pd.DataFrame({"value": xs, "density": density})
    density_df = density_df.set_index("value") 
    
    st.area_chart(density_df)
    
    return gdf_year

def create_uncertainty_visualization(gdf_year, selected_county, feature):
    """Create uncertainty visualization for a selected county using pydeck"""
    
    # Get the county data
    county_data = gdf_year[gdf_year['NAME_2'] == selected_county].iloc[0]
    county_value = county_data[feature]
    
    # Calculate uncertainty metrics based on surrounding counties
    county_geom = county_data.geometry
    gdf_year['distance'] = gdf_year.geometry.distance(county_geom)
    
    # Consider counties within 2 degrees as "neighbors" for uncertainty calculation
    neighbors = gdf_year[gdf_year['distance'] < 2.0]
    
    if len(neighbors) > 1:
        # Calculate uncertainty metrics
        neighbor_values = neighbors[feature].dropna()
        uncertainty = neighbor_values.std()
        confidence_interval = (county_value - uncertainty, county_value + uncertainty)
        
        # Create uncertainty visualization using pydeck
        st.subheader(f"Uncertainty Visualization for {selected_county}")
        
        # Prepare data for uncertainty layer
        uncertainty_gdf = neighbors.copy()
        
        # Color coding based on distance (closer = more relevant for uncertainty)
        uncertainty_gdf['uncertainty_color'] = uncertainty_gdf['distance'].apply(
            lambda x: [0, 0, 255, int(200 * (1 - min(x/2.0, 1)))]  # Blue with opacity based on distance
        )
        
        # Selected county in red
        selected_county_gdf = gdf_year[gdf_year['NAME_2'] == selected_county].copy()
        selected_county_gdf['uncertainty_color'] = [255, 0, 0, 180]
        
        # Combine layers
        combined_gdf = pd.concat([uncertainty_gdf, selected_county_gdf])
        
        # Create pydeck layers
        uncertainty_layer = pdk.Layer(
            "GeoJsonLayer",
            combined_gdf,
            opacity=0.7,
            stroked=True,
            filled=True,
            get_fill_color="uncertainty_color",
            get_line_color=[0, 0, 0, 100],
            get_line_width=2,
            pickable=True,
            tooltip={
                "html": """
                <b>{NAME_2}</b><br>
                Value: {""" + feature + """:.4f}<br>
                Distance: {distance:.2f}°
                """,
                "style": {"backgroundColor": "white", "color": "black"}
            }
        )
        
        # Center on selected county
        county_center = county_geom.centroid
        view_state = pdk.ViewState(
            latitude=county_center.y,
            longitude=county_center.x,
            zoom=7,
            pitch=0,
        )
        
        uncertainty_deck = pdk.Deck(
            layers=[uncertainty_layer],
            initial_view_state=view_state,
            tooltip=True
        )
        
        st.pydeck_chart(uncertainty_deck)
        
        # Display uncertainty metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("County Value", f"{county_value:.4f}")
        with col2:
            st.metric("Uncertainty (Std Dev)", f"{uncertainty:.4f}")
        with col3:
            st.metric("Neighbors Considered", len(neighbors))
        with col4:
            st.metric("Confidence Range", 
                     f"[{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
        
        # Create distribution chart using streamlit
        st.write("### Value Distribution in Surrounding Counties")
        if len(neighbor_values) > 0:
            # Create histogram data
            hist_values, bin_edges = np.histogram(neighbor_values, bins=15)
            hist_df = pd.DataFrame({
                'bin_center': (bin_edges[:-1] + bin_edges[1:]) / 2,
                'count': hist_values
            })
            
            # Add vertical line for selected county value
            st.area_chart(hist_df.set_index('bin_center'))
            st.caption(f"Red vertical line shows {selected_county}'s value: {county_value:.4f}")
        
        # Interactive uncertainty explanation
        st.write("### Uncertainty Interpretation")
        st.info(f"""
        **What this uncertainty means:**
        - The actual value for {selected_county} could reasonably range from **{confidence_interval[0]:.4f}** to **{confidence_interval[1]:.4f}**
        - This uncertainty is calculated based on variation in {len(neighbors)} surrounding counties
        - 🔴 **Red**: Selected county
        - 🔵 **Blue**: Surrounding counties (darker blue = closer/more relevant for uncertainty)
        - Higher uncertainty suggests the value may be less reliable for decision-making
        """)
        
    else:
        st.warning(f"Insufficient data around {selected_county} to calculate meaningful uncertainty.")

def create_interactive_uncertainty_map(gdf_year, selected_county, feature):
    """Create an interactive map showing uncertainty around the selected county"""
    
    county_data = gdf_year[gdf_year['NAME_2'] == selected_county].iloc[0]
    county_geom = county_data.geometry
    
    # Calculate distances and uncertainties
    gdf_year['distance'] = gdf_year.geometry.distance(county_geom)
    gdf_year['uncertainty_weight'] = 1 / (gdf_year['distance'] + 0.1)  # Avoid division by zero
    
    # Create a folium map centered on the selected county
    county_center = county_geom.centroid
    m = folium.Map(location=[county_center.y, county_center.x], zoom_start=8)
    
    # Add the selected county with highlight
    folium.GeoJson(
        county_geom.__geo_interface__,
        style_function=lambda x: {
            'fillColor': 'red',
            'color': 'red',
            'weight': 3,
            'fillOpacity': 0.3
        },
        tooltip=f"Selected: {selected_county}"
    ).add_to(m)
    
    # Add neighboring counties with uncertainty encoding
    for idx, row in gdf_year.iterrows():
        if row['NAME_2'] != selected_county and row['distance'] < 2.0:
            # Encode uncertainty using color intensity
            uncertainty_level = min(row['uncertainty_weight'] * 10, 1.0)
            fill_color = f'rgba(0, 0, 255, {uncertainty_level})'
            
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, color=fill_color: {
                    'fillColor': color,
                    'color': 'blue',
                    'weight': 1,
                    'fillOpacity': 0.5
                },
                tooltip=f"{row['NAME_2']}: {row[feature]:.2f}"
            ).add_to(m)
    
    return m

# Streamlit App
def main():
    st.set_page_config(layout="wide")
    st.title("US Corn Belt Yield Dashboard")

    # File Inputs
    file_path = "all_feature_data_avg.csv"
    shapefile_path = "CornBeltCounty.shp"

    if file_path and shapefile_path:
        merged_gdf, years = preprocess_data(file_path, shapefile_path)

        # Sidebar Controls
        st.sidebar.header("Controls")
        year = st.sidebar.slider("Select Year", min_value=min(years), max_value=max(years), value=max(years))

        cols_to_use = [
            'yield', 'tmmx', 'rmax', 'vs', 'sph', 'srad',
            'vpd', 'rmin', 'pr', 'tmmn', 'th'
        ]
        feature = st.sidebar.selectbox("Select Feature", cols_to_use, index=0)

        # Plot
        st.header(f"{feature.title()} Map - {year}")
        gdf_year = plot_choropleth(merged_gdf, feature, year)
        
        # County Selection for Uncertainty Analysis
        st.write("---")
        st.header("County-Specific Uncertainty Analysis")
        
        # Get list of counties for the selected year
        counties = sorted(gdf_year['NAME_2'].unique())
        
        # Create a selectbox with search functionality
        selected_county = st.selectbox(
            "Select a county to analyze uncertainty:",
            options=counties,
            index=0,
            help="Choose a county to see detailed uncertainty analysis"
        )
        
        if selected_county:
            # Create tabs for different uncertainty visualizations
            tab1, tab2 = st.tabs(["Uncertainty Metrics & Map", "Interactive Uncertainty Context"])
            
            with tab1:
                create_uncertainty_visualization(gdf_year, selected_county, feature)
            
            with tab2:
                st.subheader(f"Spatial Uncertainty Context for {selected_county}")
                uncertainty_map = create_interactive_uncertainty_map(gdf_year, selected_county, feature)
                st_folium(uncertainty_map, width=800, height=500)
                
                st.caption("""
                **Map Interpretation:**
                - 🔴 Red: Selected county
                - 🔵 Blue: Surrounding counties (darker = closer/more relevant for uncertainty)
                - The spatial distribution helps understand how geographic context affects uncertainty
                """)
        
    else:
        st.info("Please upload both the CSV data and shapefile to continue.")


# Run App
if __name__ == "__main__":
    main()
