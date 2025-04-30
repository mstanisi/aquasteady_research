transformed_path = '/Users/markos98/aquasteady_research/data/transformed/'

import pandas as pd
import plotly.express as px
import streamlit as st

# Load data
@st.cache_data
def load_data():
    finance = pd.read_csv(transformed_path + 'finance.csv')
    finance['year'] = finance['year'].astype(int)
    return finance

finance = load_data()
available_years = sorted(finance['year'].unique())

# Sidebar
selected_year = st.sidebar.selectbox(
    "Select Year:",
    available_years,
    index=len(available_years)-1
)

# Filter data
filtered_df = finance[finance['year'] == selected_year]

# Map
st.header(f"Irrigation Financing Issues ({selected_year})")
fig_map = px.choropleth(
    filtered_df,
    locations='state_code',
    locationmode="USA-states",
    color='Acres Irrigated',
    scope="usa",
    color_continuous_scale="Viridis"
)
st.plotly_chart(fig_map)

# Line plot
st.header("Top 5 States Over Time")
top_states = finance.groupby('AREA')['Acres Irrigated'].max().nlargest(5).index
fig_line = px.line(
    finance[finance['AREA'].isin(top_states)],
    x='year',
    y='Acres Irrigated',
    color='AREA'
)
st.plotly_chart(fig_line)