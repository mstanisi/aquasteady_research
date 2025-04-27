import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

transformed_path = '/Users/markos98/aquasteady_research/data/transformed/'

# ========== DATA LOADING ==========
try:
    finance = pd.read_csv(transformed_path + 'finance.csv')
    finance['year'] = finance['year'].astype(int)
    available_years = sorted(finance['year'].unique())
except Exception as e:
    print(f"Data loading error: {e}")
    raise

# ========== APP INITIALIZATION ==========
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

# ========== LAYOUT COMPONENTS ==========
sidebar = dbc.Col(
    children=[
        html.H4("Filters", className="mt-3"),
        html.Hr(),
        html.Label("Select Year:", className="mt-2"),
        dcc.Dropdown(
            id='year-selector',
            options=[{'label': str(year), 'value': year} for year in available_years],
            value=available_years[-1],
            clearable=False
        ),
        html.Hr(),
        html.P(
            "Explore irrigation financing trends",
            className="text-muted small"
        )
    ],
    md=3,
    style={'background-color': '#f8f9fa', 'padding': '20px'}
)

main_content = dbc.Col(
    children=[
        dbc.Row(dbc.Col(dcc.Graph(id='us-map'))),
        dbc.Row(dbc.Col(dcc.Graph(id='line-plot'))),
        dbc.Row([
            dbc.Col(html.H4("Data Table", className="mt-4"), width=12),
            dbc.Col(
                dash_table.DataTable(
                    id='data-table',
                    columns=[{"name": i, "id": i} for i in ['AREA', 'year', 'Acres Irrigated']],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '8px'},
                    style_header={
                        'backgroundColor': '#f8f9fa',
                        'fontWeight': 'bold'
                    }
                )
            )
        ])
    ],
    md=9
)

# ========== APP LAYOUT ==========
app.layout = dbc.Container(
    children=[
        dbc.Row(
            dbc.Col(
                html.H1("U.S. Irrigation Finance Dashboard", 
                       className="text-center my-4"),
                width=12
            )
        ),
        dbc.Row([sidebar, main_content])
    ],
    fluid=True
)

# ========== CALLBACKS ==========
@app.callback(
    [Output('us-map', 'figure'),
     Output('line-plot', 'figure'),
     Output('data-table', 'data')],
    [Input('year-selector', 'value')]
)
def update_dashboard(selected_year):
    try:
        filtered_df = finance[finance['year'] == selected_year]
        
        map_fig = px.choropleth(
            filtered_df,
            locations='AREA',
            locationmode="USA-states",
            color='Acres Irrigated',
            scope="usa",
            color_continuous_scale="Viridis",
            title=f"Irrigation Financing Issues ({selected_year})"
        ).update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        
        top_states = finance.groupby('AREA')['Acres Irrigated'].max().nlargest(5).index
        line_fig = px.line(
            finance[finance['AREA'].isin(top_states)],
            x='year',
            y='Acres Irrigated',
            color='AREA',
            title="Top 5 States Over Time"
        ).update_layout(hovermode="x unified")
        
        table_data = filtered_df[['AREA', 'year', 'Acres Irrigated']].to_dict('records')
        
        return map_fig, line_fig, table_data
    except Exception as e:
        print(f"Callback error: {e}")
        return {}, {}, []

# ========== RUN APP ==========
if __name__ == '__main__':
    app.run(debug=True)  # Changed from app.run_server()