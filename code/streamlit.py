transformed_path = '/Users/markos98/aquasteady_research/data/transformed/'

import pandas as pd
import plotly.express as px
import streamlit as st
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# Sidebar
add_selectbox = st.sidebar.selectbox(
    "Select Data",
    ("Overview", "Risk of Reduced Yield", "Cannot Finance Improvements")
)

# Overview section
if add_selectbox == "Overview":
    
    st.header("Overview Dashboard")

    # Load saved data
    @st.cache_data  # Cache for performance
    def load_data():
        return pd.read_csv(transformed_path + 'random_forest.csv')

    df = load_data()

    st.title('Feature Importance Visualization')

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 24))  # Height adjusted as in your original

    # Get sorted data
    sorted_idx = np.argsort(df['importance'])
    features = df['feature'].values
    importances = df['importance'].values

    # Create horizontal bars
    bars = ax.barh(
        range(len(sorted_idx)),
        importances[sorted_idx],
        color='royalblue',
        height=0.8
    )

    # Customize labels
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([features[i] for i in sorted_idx], fontsize=10)
    ax.set_title('Feature Importances', pad=20, fontsize=14)
    ax.set_xlabel('Relative Importance', fontsize=12)
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.001,
            bar.get_y() + bar.get_height()/2,
            f'{width:.3f}',
            va='center',
            fontsize=8
        )

    plt.tight_layout()

    # Display in Streamlit
    st.pyplot(fig)

    # Optional: Show raw data
    if st.checkbox('Show raw data'):
        st.dataframe(df.sort_values('importance', ascending=False))    

# Cannot finance improvements section
elif add_selectbox == "Cannot Finance Improvements":
    st.header("Financing Issues Dashboard")
    
    @st.cache_data
    def load_finance_data():
        finance_states = pd.read_csv(transformed_path + 'finance_states.csv')
        finance_regions = pd.read_csv(transformed_path + 'finance_regions.csv')
        return finance_states, finance_regions
    
    finance_states, finance_regions = load_finance_data()

    # Map
    st.header("Financing Issues by State (2023)")
    fig_map = px.choropleth(
        finance_states,
        locations='state_code',
        locationmode="USA-states",
        color='Acres Irrigated',
        scope="usa",
        color_continuous_scale="sunset"
    )
    st.plotly_chart(fig_map)

    # Bar chart
    st.header("Top 5 States")
    top_states = finance_states.groupby('AREA')['Acres Irrigated'].max().nlargest(5).index
    fig_bar = px.bar(
        finance_states[finance_states['AREA'].isin(top_states)],
        x='AREA',
        y='Acres Irrigated',
        color='AREA'
    )
    st.plotly_chart(fig_bar)

    # Line plot
    st.header("Top 5 Regions Over Time")
    top_regions = finance_regions.groupby('AREA')['Acres Irrigated'].max().nlargest(5).index
    fig_line = px.line(
        finance_regions[finance_regions['AREA'].isin(top_regions)],
        x='year',
        y='Acres Irrigated',
        color='AREA'
    )
    st.plotly_chart(fig_line)

# Risk of reduced yield section
elif add_selectbox == "Risk of Reduced Yield":
    st.header("Yield Risk Dashboard")
    
    @st.cache_data
    def load_yield_data():
        yield_states = pd.read_csv(transformed_path + 'yield_states.csv')
        yield_regions = pd.read_csv(transformed_path + 'yield_regions.csv')
        return yield_states, yield_regions
    
    yield_states, yield_regions = load_yield_data()

    # Map
    st.header("Yield Issues by State (2023)")
    fig_map = px.choropleth(
        yield_states,
        locations='state_code',
        locationmode="USA-states",
        color='Acres Irrigated',
        scope="usa",
        color_continuous_scale="sunset"
    )
    st.plotly_chart(fig_map)

    # Bar chart
    st.header("Top 5 States")
    top_states = yield_states.groupby('AREA')['Acres Irrigated'].max().nlargest(5).index
    fig_bar = px.bar(
        yield_states[yield_states['AREA'].isin(top_states)],
        x='AREA',
        y='Acres Irrigated',
        color='AREA'
    )
    st.plotly_chart(fig_bar)

    # Line plot
    st.header("Top 5 Regions Over Time")
    top_regions = yield_regions.groupby('AREA')['Acres Irrigated'].max().nlargest(5).index
    fig_line = px.line(
        yield_regions[yield_regions['AREA'].isin(top_regions)],
        x='year',
        y='Acres Irrigated',
        color='AREA'
    )
    st.plotly_chart(fig_line)