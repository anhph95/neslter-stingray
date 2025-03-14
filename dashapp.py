import dash
from dash import dcc, html, ctx, Output, Input, State, no_update
import plotly.express as px
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from pathlib import Path

def cli():
    import argparse
    parser = argparse.ArgumentParser(description='Stingray Dashboard')
    parser.add_argument('--host', type=str, help='Host IP address for the Dash app, default is 0.0.0.0', default='0.0.0.0')
    parser.add_argument('--port', type=int, help='Port number for the Dash app, default is 8050', default=8050)
    return parser.parse_args()

args = cli()
args.port = str(args.port)

# Dash App Initialization
app = dash.Dash(__name__)

# Define working directory and subdirectories
work_dir = Path('/dash_data') if Path('/dash_data').is_dir() else Path('dash_data')
data_dir, misc_dir = work_dir / 'data', work_dir / 'misc'

# Function to scan for CSV files
def get_csv_files():
    return sorted(f.stem for f in data_dir.glob("*.csv")) if data_dir.exists() else []

# # Get list of available CSV files (without extensions)
# csv_files = sorted(f.stem for f in data_dir.glob("*.csv")) if data_dir.exists() else []

# Units for each variable
sled_units = {
    'temperature': '°C',
    'conductivity': 'S m⁻¹',
    'pressure': 'dbar',
    'depth': 'm',
    'salinity': 'psu',
    'density': 'kg m⁻³',
    'latitude': '°',
    'longitude': '°',
    'nitrate': 'µM',
    'par': 'µmol photons m⁻² s⁻¹',
    'chlorophyll': 'µg l⁻¹',
    'backscattering': 'm⁻¹ sr⁻¹',
    'oxygen_concentration': 'µM',
    'oxygen_saturation': '%',
    'pitch': '°',
    'roll': '°',
    'heading': '°',
    'altitude': 'm',
}

def load_data(file_name):
    '''Load CSV file into a DataFrame and preprocess it.'''
    df = pd.read_csv(f'{data_dir}/{file_name}.csv', low_memory=False)
    if not pd.api.types.is_datetime64_any_dtype(df['times']):
        df['times'] = pd.to_datetime(df['times'], errors='coerce', cache=True)
    for col in ['media', 'frame', 'media_path', 'link']:
        df[col] = df.get(col, np.nan)
    return df

# Initial dataset
csv_files = get_csv_files()
if csv_files:
    df = load_data(csv_files[-1])
else:
    df = pd.DataFrame()

# Define variables for dropdowns
meta_vars = ['timestamp', 'times', 'matdate', 'latitude', 'longitude', 'depth', 'media', 'media_path', 'frame', 'id', 'link']
sensor_vars = [col for col in df.columns if '_std' not in col and col not in meta_vars] if not df.empty else []
colormaps = [name for name in px.colors.sequential.__dict__ if not name.startswith('_')]

# Load bathymetry and station data if files exist
load_csv = lambda file: pd.read_csv(file, dtype=str, encoding="utf-8") if file.exists() else None
stations, bathy = map(load_csv, [misc_dir / 'NESLTER_station_list.csv', misc_dir / 'NESLTER_transect_bathymetry.csv'])

# App Layout
app.layout = html.Div([
    html.Div([
        html.Span('NES-LTER Stingray Dashboard', 
                style={'color': 'white', 'fontSize': '24px', 'fontWeight': 'bold', 'padding': '10px'}),

        html.Span('Written by: Anh H. Pham | anh.pham@whoi.edu', 
                style={'font-size': '14px', 'color': 'white', 'padding': '10px'}),
    ], style={
        'display': 'flex',
        'justify-content': 'space-between',  # Pushes title left & author right
        'align-items': 'center',  # Aligns text vertically
        'backgroundColor': '#3D4451',  # Dark background
        'width': '100%',  # Ensures full width
        'border-radius': '5px',  # Rounded corners for a clean look
        'box-shadow': '0px 2px 5px rgba(0,0,0,0.1)'  # Light shadow for depth
    }),
    
    html.Div([
        # Row 1: Cruise Selection, Date Range, Start & End Time, Download Button
        html.Div([
            html.Label('Cruise:', style={'font-weight': 'bold', 'font-size': '14px'}),
            dcc.Dropdown(id='csv-selector',
                        options=[{'label': f, 'value': f} for f in csv_files],
                        value=csv_files[-1] if csv_files else None, clearable=False,
                        style={'width': '140px', 'font-size': '12px'}),
            html.Button('Refresh', id='refresh-button', n_clicks=0,
                        style={
                            'font-size': '12px',  # Slightly larger text
                            'padding': '5px 5px',  # More padding for better visibility
                            'background-color': '#007BFF',  # Bright blue to stand out
                            'color': 'white',  # White text for contrast
                            'border': 'none',  # Remove default border
                            'border-radius': '5px',  # Rounded corners
                            'cursor': 'pointer',  # Pointer cursor to indicate it's clickable
                            'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.2)',  # Light shadow to make it pop
                            'margin-left': '10px',
                            'transition': '0.3s'  # Smooth transition for hover effect
                        },),

            html.Label('Start Date:', style={'font-size': '12px'}),
            dcc.Input(
                id='start-date', 
                type='text', 
                placeholder='YYYY-MM-DD', 
                debounce=True,
                style={'width': '80px', 'font-size': '12px'}
            ),

            html.Label('Start Time:', style={'font-size': '12px'}),
            dcc.Input(
                id='start-time', 
                type='text', 
                placeholder='HH:MM:SS', 
                debounce=True,
                style={'width': '80px', 'font-size': '12px'}
            ),

            html.Label('End Date:', style={'font-size': '12px'}),
            dcc.Input(
                id='end-date', 
                type='text', 
                placeholder='YYYY-MM-DD', 
                debounce=True,
                style={'width': '80px', 'font-size': '12px'}
            ),

            html.Label('End Time:', style={'font-size': '12px'}),
            dcc.Input(
                id='end-time', 
                type='text', 
                placeholder='HH:MM:SS', 
                debounce=True,
                style={'width': '80px', 'font-size': '12px'}
            ),

            html.Button('Download CSV', id='download-button',
                        style={
                            'font-size': '12px',  # Slightly larger text
                            'padding': '5px 5px',  # More padding for better visibility
                            'background-color': '#007BFF',  # Bright blue to stand out
                            'color': 'white',  # White text for contrast
                            'border': 'none',  # Remove default border
                            'border-radius': '5px',  # Rounded corners
                            'cursor': 'pointer',  # Pointer cursor to indicate it's clickable
                            'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.2)',  # Light shadow to make it pop
                            'margin-left': '10px',
                            'transition': '0.3s'  # Smooth transition for hover effect
                        },),
            dcc.Download(id='download-dataframe-csv'),
        ], style={'display': 'flex', 'align-items': 'center', 'gap': '8px', 'flex-wrap': 'wrap'}),

        html.Br(),

        # Row 2: X-Axis, Min/Max Depth, Color Variable, Min/Max Color
        html.Div([
            html.Label('X-Axis:', style={'font-size': '12px'}),
            dcc.Dropdown(id='x-axis-variable', value='latitude', clearable=False,
                        options=[{'label': lbl, 'value': val} for lbl, val in [('Latitude', 'latitude'), ('Time', 'times')]],
                        style={'width': '100px', 'font-size': '12px'}),

            html.Label('Min Depth:', style={'font-size': '12px'}),
            dcc.Input(id='y-min', type='number', value=-100, debounce=True,
                    style={'width': '60px', 'font-size': '12px'}),

            html.Label('Max Depth:', style={'font-size': '12px'}),
            dcc.Input(id='y-max', type='number', value=1000, debounce=True,
                    style={'width': '60px', 'font-size': '12px'}),

            html.Label('Color Variable:', style={'font-size': '12px'}),
            dcc.Dropdown(id='color-variable', value='temperature', clearable=False,
                        options=[{'label': var.capitalize(), 'value': var} for var in sensor_vars],
                        style={'width': '100px', 'font-size': '12px'}),
            
            dcc.Checklist(id='bathymetry', options=[{'label': 'Bathymetry', 'value': 'True'}], value=['True'],
                        style={'font-size': '12px'}),

            dcc.Checklist(id='station', options=[{'label': 'Stations', 'value': 'True'}], value=['True'],
                        style={'font-size': '12px'}),
        ], style={'display': 'flex', 'align-items': 'center', 'gap': '8px', 'flex-wrap': 'wrap'}),

        html.Br(),

        # Row 3: Colormap, Opacity, Figure Size, Bathymetry & Station Checklists
        html.Div([
            html.Label('Colormap:', style={'font-size': '12px'}),
            dcc.Dropdown(id='color-map', options=[{'label': cmap, 'value': cmap} for cmap in colormaps], value='Jet',
                        clearable=False, style={'width': '100px', 'font-size': '12px'}),
            
            html.Label('Color Min:', style={'font-size': '12px'}),
            dcc.Input(id='vmin-input', type='number', step=0.01, debounce=True,
                    style={'width': '60px', 'font-size': '12px'}),

            html.Label('Color Max:', style={'font-size': '12px'}),
            dcc.Input(id='vmax-input', type='number', step=0.01, debounce=True,
                    style={'width': '60px', 'font-size': '12px'}),

            html.Label('Filter Min:', style={'font-size': '12px'}),
            dcc.Input(id='filter-min', type='number', debounce=True,
                    style={'width': '60px', 'font-size': '12px'}),
            
            html.Label('Filter Max:', style={'font-size': '12px'}),
            dcc.Input(id='filter-max', type='number', debounce=True,
                    style={'width': '60px', 'font-size': '12px'}),

            html.Label('Filter Opacity:', style={'font-size': '12px'}),
            dcc.Input(id='hidden-opacity', type='number', value=0.01, step=0.01, debounce=True,
                    style={'width': '60px', 'font-size': '12px'}),

            html.Label('Fig Width:', style={'font-size': '12px'}),
            dcc.Input(id='fig-width', type='number', value=None, debounce=True,
                    style={'width': '70px', 'font-size': '12px'}),

            html.Label('Fig Height:', style={'font-size': '12px'}),
            dcc.Input(id='fig-height', type='number', value=None, debounce=True,
                    style={'width': '70px', 'font-size': '12px'}),
        ], style={'display': 'flex', 'align-items': 'center', 'gap': '8px', 'flex-wrap': 'wrap'}),

    ], style={
        'backgroundColor': '#F0F0F0',  # Light grey background for all controls
        'padding': '10px',
        'border-radius': '5px',  # Rounded corners for a clean look
        'box-shadow': '0px 2px 5px rgba(0,0,0,0.1)'  # Light shadow for depth
    }),

    html.Div([
        dcc.Graph(id='scatter-plot', style={'flex': '7', 'align-self': 'stretch'}),  
        html.Div(
            id='click-output',
            style={
                'flex': '3',
                'padding': '10px 10px',
                'height': '500px',
                'overflowX': 'auto',
                'text-align': 'left',
                'backgroundColor': 'white',

            }
        )
    ], style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'stretch', 'justify-content': 'flex-end'})
])

# Callback to update file list and maintain valid selection
@app.callback(
    Output("csv-selector", "options"),
    Output("csv-selector", "value"),  # Ensure dropdown selection stays valid
    Input("refresh-button", "n_clicks"),
    prevent_initial_call=True  # Prevents running on app startup
)
def update_file_list(n_clicks):
    files = get_csv_files()  # Refresh file list
    options = [{'label': f, 'value': f} for f in files]
    new_value = files[-1] if files else None
    return options, new_value

@app.callback(
    [
        Output('start-date', 'value'),
        Output('end-date', 'value'),
        Output('start-time', 'value'),
        Output('end-time', 'value'),
    ],
    Input('csv-selector', 'value'),
)
def update_dropdowns(csv_file):
    '''Update date range and time fields based on the selected CSV file.'''
    if not csv_file:
        return no_update, no_update, no_update, no_update

    df = load_data(csv_file)
    
    if 'times' in df.columns and not df['times'].empty:
        return df['times'].min().strftime('%Y-%m-%d'), df['times'].max().strftime('%Y-%m-%d'), df['times'].min().strftime('%H:%M:%S'), df['times'].max().strftime('%H:%M:%S')

    return no_update, no_update, no_update, no_update

@app.callback(
    Output('download-dataframe-csv', 'data'),
    Input('download-button', 'n_clicks'),
    State('csv-selector', 'value'),
    prevent_initial_call=True
)
def download_csv(n_clicks, csv_file):
    '''Download the selected CSV file.'''
    if ctx.triggered_id == 'download-button' and csv_file:
        df = load_data(csv_file)
        return dcc.send_data_frame(df.to_csv, filename=f'{csv_file}.csv', index=False)
    return no_update

@app.callback(
    Output('scatter-plot', 'figure'),
    [
        Input('csv-selector', 'value'),
        Input('start-date', 'value'),
        Input('start-time', 'value'),
        Input('end-date', 'value'),
        Input('end-time', 'value'),
        Input('x-axis-variable', 'value'),
        Input('color-variable', 'value'),
        Input('color-map', 'value'),
        Input('vmin-input', 'value'),
        Input('vmax-input', 'value'),
        Input('y-min', 'value'),
        Input('y-max', 'value'),
        Input('filter-min', 'value'),
        Input('filter-max', 'value'),
        Input('hidden-opacity', 'value'),
        Input('fig-width', 'value'),
        Input('fig-height', 'value'),
        Input('bathymetry', 'value'),
        Input('station', 'value')
    ]
)
def update_plot(csv_file, start_date, start_time, end_date,  end_time, x_axis, color_var, color_map, vmin, vmax,
                zmin, zmax, filter_min, filter_max, hidden_opacity, figure_width, figure_height, bathymetry, station):
    '''Update scatter plot based on user inputs.'''
    
    # Return empty figure if no CSV file is selected
    if not csv_file:
        return px.scatter()

    df = load_data(csv_file)
    # Filter out invalid latitude values
    df.loc[(df['latitude'] < 39.5) | (df['latitude'] > 41.5), 'latitude'] = np.nan

    # Date time filtering
    start_date = pd.to_datetime(f"{start_date} {start_time}" if start_time else start_date, errors='coerce')
    end_date = pd.to_datetime(f"{end_date} {end_time}" if end_time else end_date, errors='coerce')
    df = df[(df['times'] >= start_date) & (df['times'] <= end_date)]   

    # Compute min and max values for color_var if not provided
    if vmin is None or vmax is None:
        quantiles = df[color_var].quantile([0.05, 0.95])
        vmin = quantiles[0.05] if vmin is None else vmin
        vmax = quantiles[0.95] if vmax is None else vmax

    # Construct hover_data and custom_data dynamically
    hover_data = { 'depth': ':.2f', 'latitude': ':.2f', 'longitude': ':.2f',  color_var: ':.2f'}
    custom_data = ['media','frame', 'times', 'latitude', 'longitude', 'link'] + sensor_vars

    # Create scatter plot
    fig = px.scatter(
        df,
        x=x_axis,
        y='depth',
        color=color_var,
        color_continuous_scale=color_map,
        range_color=[vmin, vmax],
        # hover_name='media',
        hover_data=hover_data,
        custom_data=custom_data
    )

    # Apply opacity adjustments based on color_var threshold
    opacity_values = np.where((df[color_var] < filter_min) | (df[color_var] > filter_max) | pd.isna(df[color_var]), hidden_opacity, 1)
    fig.update_traces(marker=dict(opacity=opacity_values))
       
    # Update figure layout
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        coloraxis_colorbar=dict(
            title=dict(
                text=f'<br><br>&nbsp;{color_var.replace("_", " ").capitalize()} ({sled_units.get(color_var, "")})',
                side='right',  # Keeps the title vertical
            ),
            ticks='outside'
        ),
        xaxis=dict(
            title=x_axis.capitalize(),
            autorange='reversed' if x_axis == 'latitude' else True,
            showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)',
            linecolor='black', linewidth=1, mirror=True,
            ticks='outside', tickwidth=1, tickcolor='black'
        ),
        yaxis=dict(
            title='Depth (m)',
            range=[zmax, zmin],
            showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)',
            linecolor='black', linewidth=1, mirror=True,
            ticks='outside', tickwidth=1, tickcolor='black'
        ),
        font=dict(color='black'),
        width = figure_width if figure_width else None,
        height = figure_height if figure_height else None
    )
    
    # Add bathymetry trace if selected
    if 'True' in bathymetry and x_axis == 'latitude':
        fig.add_trace(go.Scatter(
            x=bathy['latitude'],
            y=bathy['bottom_depth_meters'],
            mode='lines',
            line=dict(color='black', width=1),
            name='Bathymetry',
            showlegend=False
        ))

    # Add station markers if selected
    if 'True' in station and x_axis == 'latitude':
        for _, row in stations.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['latitude'], row['latitude']],
                y=[-5, -20],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[row['latitude']],
                y=[-25],
                text=[row['station']],
                mode='text',
                textposition='top center',
                textfont=dict(color='black', size=14),
                showlegend=False
            ))

    return fig

@app.callback(
    Output('click-output', 'children'),
    Input('scatter-plot', 'clickData'),
)
def display_click_data(clickData):
    '''Displays details of a clicked point on the scatter plot, showing all relevant variables.'''
    
    if not clickData or 'points' not in clickData or not clickData['points']:
        return 'Click on a point to see full details.'

    try:
        point = clickData['points'][0]
        custom_data = point.get('customdata', [])
        media = custom_data[0] if len(custom_data) > 0 and pd.notna(custom_data[0]) else 'N/A'
        frame = custom_data[1] if len(custom_data) > 1 and pd.notna(custom_data[1]) else 'N/A'
        raw_time = custom_data[2] if len(custom_data) > 2 and pd.notna(custom_data[2]) else None
        latitude = custom_data[3] if len(custom_data) > 4 and pd.notna(custom_data[3]) else None
        longitude = custom_data[4] if len(custom_data) > 5 and pd.notna(custom_data[4]) else None
        media_link = custom_data[5] if len(custom_data) > 6 and pd.notna(custom_data[5]) else None
        depth = point.get('y', 'N/A')

        # Format time safely
        formatted_time = pd.to_datetime(raw_time).strftime('%Y-%m-%d %H:%M:%S') if raw_time else 'N/A'

        # Prepare variable details dynamically
        variable_details = []
        for i, var in enumerate(sensor_vars, start=6):  # Start at index 5 since first vars are fixed
            value = custom_data[i] if i < len(custom_data) else None  # Ensure index exists

            # Adjust formatting based on magnitude
            if pd.isna(value):  
                display_value = 'N/A'
            elif isinstance(value, (int, float)):
                if abs(value) >= 1e6:  # Large numbers in scientific notation
                    display_value = f'{value:.2e}'
                elif abs(value) < 1 and value != 0:  # Small floats with 4 decimals
                    display_value = f'{value:.4f}'
                else:  # Standard format with comma separator
                    display_value = f'{value:,.2f}'
            else:  # Catch any unexpected types
                display_value = str(value)

            variable_details.append(
                html.Div([
                    html.Span(f'📈 {var.capitalize().replace("_"," ")} ({sled_units.get(var,"")}):', style={'flex': '7', 'text-align': 'left'}),
                    html.Span(display_value, style={'flex': '3', 'text-align': 'right'})
                ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'})
            )

        return html.Div([
            html.Div([
                html.Span('📽️ Media:', style={'flex': '3', 'text-align': 'left'}),
                html.A(media, href=media_link, target='_blank', style={'flex': '7', 'text-align': 'right'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),
            
            html.Div([
                html.Span('🎞️ Frame:', style={'flex': '1', 'text-align': 'left'}),
                html.Span(frame, style={'flex': '1', 'text-align': 'right'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),

            html.Div([
                html.Span('⏳ Time:', style={'flex': '1', 'text-align': 'left'}),
                html.Span(formatted_time, style={'flex': '1', 'text-align': 'right'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),

            html.Div([
                html.Span('🌍 Latitude:', style={'flex': '1', 'text-align': 'left'}),
                html.Span(f'{latitude:.2f}°' if latitude is not None else 'N/A', style={'flex': '1', 'text-align': 'right'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),

            html.Div([
                html.Span('🌍 Longitude:', style={'flex': '1', 'text-align': 'left'}),
                html.Span(f'{longitude:.2f}°' if longitude is not None else 'N/A', style={'flex': '1', 'text-align': 'right'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),

            html.Div([
                html.Span('🌊 Depth:', style={'flex': '1', 'text-align': 'left'}),
                html.Span(f'{depth:.2f} m' if depth != 'N/A' else 'N/A', style={'flex': '1', 'text-align': 'right'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),

            html.Hr(),

            *variable_details  # Unpacking list of dynamically generated variable details
        ])

    except Exception as e:
        return f'⚠️ Error processing click data: {str(e)}'

if __name__ == '__main__':
    app.run_server(host=args.host, port=args.port)
