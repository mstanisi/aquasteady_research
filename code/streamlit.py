transformed_path = '/Users/markos98/aquasteady_research/data/transformed/'
spatial_path = "/Users/markos98/aquasteady_research/data/spatial/"
visuals_path = "/Users/markos98/aquasteady_research/visuals/"

import pandas as pd
import plotly.express as px
import streamlit as st
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from PIL import Image

# ====================
# STYLING IMPROVEMENTS
# ====================
st.set_page_config(layout="wide")

# Custom CSS for white background and readable text
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: white !important;
    }
    
    /* Plotly chart styling */
    .js-plotly-plot .plotly {
        background-color: white !important;
    }
    
    /* Text elements */
    h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stExpander {
        color: #333333 !important;
    }
    
    /* Expandable sections */
    .stExpander {
        background-color: white;
        border: 1px solid #e0e0e0 !important;
    }
    
    /* Images with white background */
    img {
        background-color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
**Currently in the development stage**, lab tests for the efficacy of AquaSteady's seaweed-based hydrogels are highly promising. This graph shows how a sapling planted in AquaSteady withstood a drought period and grew to be 27% larger than a reference, non-AquaSteady sapling.
""")

# ====================
# VISUALIZATIONS
# ====================

# Display your pre-saved image with improved styling
def display_image_with_white_bg(image_path, caption):
    img = Image.open(image_path)
    
    # If image has transparency, create white background
    if img.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])  # Paste using alpha channel as mask
        img = background
        
    st.image(img, caption=caption)

# Sapling image
display_image_with_white_bg(visuals_path + "sapling.png", "")

with st.expander("About this graph"):
    st.markdown("""
    "AquaSteady nets and discs have proven effective on saplings according to a study done on orange trees in Brazil. Soon after the transplantation they were hit by 45 days of drought but the saplings with AquaSteady were doing well and now one year after the planting they grew 27% more than the reference saplings."
    """)

st.markdown("""
    **AquaSteady will soon be market-ready**, but the question remains whether farmers will be on the market for it. Using 20 years of data from the USDA Census of Agriculture, I investigated what factors are preventing American farmers from reaching their water conservation goals. 
Because all these factors rose over time, the data was multi-correlated and required **machine learning techniques** to uncover the true meaning behind them. 
""")

# Other images
display_image_with_white_bg(visuals_path + "ridge_regression.png", "")

with st.expander("About this graph"):
    st.markdown("""
    This machine learning method uses farming regions as its target variable and tests whether the given barriers, known as features, accurately guess the region of the row that they belong to. 
    """)
    
display_image_with_white_bg(visuals_path + "randomforest.png", "")

with st.expander("About this graph"):
    st.markdown("""
    This machine learning method is similar to the previous one, but uses the barriers themselves as its target. It uses a decision tree structure to determine how well a given region is at predicting its own standing among the barriers. 
    """)

display_image_with_white_bg(visuals_path + "watershed_map.png", 
                          "Water Conservation Challenge Importance by Watershed")

with st.expander("About this map"):
    st.markdown("""
    **Key Regions:**
    - California (HUC 18)
    - Arkansas-White-Red (HUC 11) 
    - Missouri (HUC 10)
    
    Colors show relative importance scores from Random Forest analysis.
    """)

st.markdown("""
    Now that the major impediments to water conservation have been determined, as well what regions are most affected by these trends, let's take a look at how these factors plot over time. 
""")

# ====================
# INTERACTIVE CHARTS
# ====================

@st.cache_data
def load_finance_data():
    finance_states = pd.read_csv(transformed_path + 'finance_states.csv')
    finance_regions = pd.read_csv(transformed_path + 'finance_regions.csv')
    return finance_states, finance_regions
    
finance_states, finance_regions = load_finance_data()

# Configure Plotly to use white backgrounds by default
plotly_template = {
    'layout': {
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {'color': '#333333'},
        'xaxis': {'gridcolor': '#f0f0f0'},
        'yaxis': {'gridcolor': '#f0f0f0'}
    }
}

st.header("Farmers reporting that they 'cannot finance improvements' to their irrigation setups grew steadily.")
top_regions = finance_regions.groupby('AREA')['Acres Irrigated'].max().nlargest(5).index
fig_line = px.line(
    finance_regions[finance_regions['AREA'].isin(top_regions)],
    x='year',
    y='Acres Irrigated',
    color='AREA',
    template=plotly_template
)
st.plotly_chart(fig_line)

@st.cache_data
def load_yield_data():
    yield_states = pd.read_csv(transformed_path + 'yield_states.csv')
    yield_regions = pd.read_csv(transformed_path + 'yield_regions.csv')
    return yield_states, yield_regions
    
yield_states, yield_regions = load_yield_data()

st.header("Farmers reporting that 'risk of reduced yield or poorer crop quality' prevented them from making irrigation improvements also rose over time.")
top_regions = yield_regions.groupby('AREA')['Acres Irrigated'].max().nlargest(5).index
fig_line = px.line(
    yield_regions[yield_regions['AREA'].isin(top_regions)],
    x='year',
    y='Acres Irrigated',
    color='AREA',
    template=plotly_template
)
st.plotly_chart(fig_line)

st.markdown("""
    We can now, with confidence, say that the markets most worth targeting are in farmland belonging to three major watersheds: **California (region 18), Missouri (region 10), and Arkansas-White-Red (region 11)**. Within these regions, the marketing itself should focus on **AquaSteady as a financially stable solution for irrigation and as a boon for crop yield and crop quality**. 

I will be returning to this project to determine what agricultural zones within the aforementioned regions should be targeted for field research, as well as how farmers in these zones deal with droughts. 
""")
