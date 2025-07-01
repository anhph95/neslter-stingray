import dash
from dash import dcc, html, ctx, Output, Input, State, no_update
import plotly.express as px
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from pathlib import Path

# Utility function to calculate seawater density using the IES80 equation
def ies80(s, t, p=0):
    """
    Computes the density of seawater using the International Equation of State (IES80).
    
    Parameters:
    s : array_like
        Salinity (PSU)
    t : array_like
        Temperature (°C)
    p : array_like, optional
        Pressure (bars), default is 0
    
    Returns:
    rho : array_like
        Density of seawater (kg/m³)
    """
    # Coefficients for density calculation
    r0_coef = [999.842594, 6.793952e-2, -9.09529e-3, 1.001685e-4, -1.120083e-6,
                6.536332e-9, 8.24493e-1, -4.0899e-3, 7.6438e-5, -8.2467e-7,
                5.3875e-9, -5.72466e-3, 1.0227e-4, -1.6546e-6, 4.8314e-4]
    
    r0 = np.polyval(r0_coef[:6][::-1], t) + \
         np.polyval(r0_coef[6:11][::-1], t) * s + \
         np.polyval(r0_coef[11:14][::-1], t) * s**1.5 + \
         r0_coef[14] * s**2
    
    if np.any(p):
        # Coefficients for compressibility
        K_coef = [19652.21, 148.4206, -2.327105, 1.360447e-2, -5.155288e-5, 3.239908,
                  1.43713e-3, 1.16092e-4, -5.77905e-7, 8.50935e-5,
                  -6.12293e-6, 5.2787e-8, 54.6746, -0.603459, 1.09987e-2,
                  -6.1670e-5, 7.944e-2, 1.6483e-2, -5.3009e-4, 2.2838e-3,
                  -1.0981e-5, -1.6078e-6, 1.91075e-4, -9.9348e-7,
                  2.0816e-8, 9.1697e-10]
        
        K = np.polyval(K_coef[:5][::-1], t) + \
            np.polyval(K_coef[5:9][::-1], t) * p + \
            np.polyval(K_coef[9:12][::-1], t) * p**2 + \
            np.polyval(K_coef[12:16][::-1], t) * s + \
            np.polyval(K_coef[16:19][::-1], t) * s**1.5 + \
            np.polyval(K_coef[19:22][::-1], t) * p * s + \
            K_coef[22] * p * s**1.5 + \
            np.polyval(K_coef[23:26][::-1], t) * p**2 * s
        
        rho = r0 / (1 - p / K)
    else:
        rho = r0
    
    return rho

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
    'sound_velocity': 'm s⁻¹',
}

# Function to scan for CSV files
def get_csv_files():
    return sorted(f.stem for f in data_dir.glob("*.csv")) if data_dir.exists() else []

# ========== Command-line interface ==========
def cli():
    import argparse
    parser = argparse.ArgumentParser(description='Stingray Dashboard')
    parser.add_argument('--host', type=str, help='Host IP address for the Dash app, default is 0.0.0.0', default='0.0.0.0')
    parser.add_argument('--port', type=int, help='Port number for the Dash app, default is 8050', default=8050)
    return parser.parse_args()

# Command-line arguments
args = cli()
args.port = str(args.port)

# ========== Dash App Initialization ==========
app = dash.Dash(__name__)

# Number of processes for the server
n_process = min(os.cpu_count()-1, 8) 

# Define working directory and subdirectories
work_dir = Path('/dash_data') if Path('/dash_data').is_dir() else Path('dash_data')
data_dir, misc_dir = work_dir / 'data', work_dir / 'misc'

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
stations['latitude'] = pd.to_numeric(stations['latitude'], errors='coerce')

# App Layout
app.layout = html.Div([
    html.Div([
        html.Span('NES-LTER Stingray Dashboard', 
            style={'color': 'white', 'fontSize': '24px', 'fontWeight': 'bold', 'padding': '10px'}),
        html.Span('Written by: Anh H. Pham | anh.pham@whoi.edu', 
            style={'font-size': '14px', 'color': 'white', 'padding': '10px'}),
    ], style={
        'display': 'flex',
        'justify-content': 'space-between',
        'align-items': 'center',
        'backgroundColor': '#3D4451',
        'width': '100%',
        'border-radius': '5px',
        'box-shadow': '0px 2px 5px rgba(0,0,0,0.1)'
    }),

    html.Div([
        # Row 1
        html.Div([
            html.Label('Cruise:', style={'font-weight': 'bold', 'font-size': '14px'}),
            dcc.Dropdown(id='csv-selector',
                options=[{'label': f, 'value': f} for f in csv_files],
                value=csv_files[-1] if csv_files else None, clearable=False,
                style={'width': '140px', 'font-size': '12px'}),
            html.Button('Refresh', id='refresh-button', n_clicks=0, style={
                'font-size': '12px', 'padding': '5px 5px', 'background-color': '#007BFF',
                'color': 'white', 'border': 'none', 'border-radius': '5px', 'cursor': 'pointer',
                'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.2)', 'margin-left': '10px', 'transition': '0.3s'
            }),

            html.Label('Start Date:', style={'font-size': '12px'}),
            dcc.Input(id='start-date', type='text', placeholder='YYYY-MM-DD', debounce=True,
                style={'width': '80px', 'font-size': '12px'}),
            html.Label('Start Time:', style={'font-size': '12px'}),
            dcc.Input(id='start-time', type='text', placeholder='HH:MM:SS', debounce=True,
                style={'width': '80px', 'font-size': '12px'}),
            html.Label('End Date:', style={'font-size': '12px'}),
            dcc.Input(id='end-date', type='text', placeholder='YYYY-MM-DD', debounce=True,
                style={'width': '80px', 'font-size': '12px'}),
            html.Label('End Time:', style={'font-size': '12px'}),
            dcc.Input(id='end-time', type='text', placeholder='HH:MM:SS', debounce=True,
                style={'width': '80px', 'font-size': '12px'}),

            html.Button('Download CSV', id='download-button', style={
                'font-size': '12px', 'padding': '5px 5px', 'background-color': '#007BFF',
                'color': 'white', 'border': 'none', 'border-radius': '5px', 'cursor': 'pointer',
                'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.2)', 'margin-left': '10px', 'transition': '0.3s'
            }),
            dcc.Download(id='download-dataframe-csv'),
        ], style={'display': 'flex', 'align-items': 'center', 'gap': '8px', 'flex-wrap': 'wrap'}),

        html.Br(),

        # Row 2
        html.Div([
            html.Label('X-Axis:', style={'font-size': '12px'}),
            dcc.Dropdown(id='x-axis-variable', value='latitude', clearable=False,
                options=[{'label': lbl, 'value': val} for lbl, val in [('Latitude', 'latitude'), ('Latitude_bin', 'latitude_bin'), ('Time', 'times')]],
                style={'width': '100px', 'font-size': '12px'}),
            html.Label('Min Depth:', style={'font-size': '12px'}),
            dcc.Input(id='z-min', type='number', value=0, debounce=True,
                style={'width': '60px', 'font-size': '12px'}),
            html.Label('Max Depth:', style={'font-size': '12px'}),
            dcc.Input(id='z-max', type='number', value=1000, debounce=True,
                style={'width': '60px', 'font-size': '12px'}),
            html.Label('Min Latitude:', style={'font-size': '12px'}),
            dcc.Input(id='lat-min', type='number', value=39.5, debounce=True,
                style={'width': '60px', 'font-size': '12px'}),
            html.Label('Max Latitude:', style={'font-size': '12px'}),
            dcc.Input(id='lat-max', type='number', value=41.5, debounce=True,
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

        # Row 3
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
            dcc.Input(id='hidden-opacity', type='number', value=0.05, step=0.01, debounce=True,
                style={'width': '60px', 'font-size': '12px'}),
            html.Label('Fig Width:', style={'font-size': '12px'}),
            dcc.Input(id='fig-width', type='number', value=None, debounce=True,
                style={'width': '70px', 'font-size': '12px'}),
            html.Label('Fig Height:', style={'font-size': '12px'}),
            dcc.Input(id='fig-height', type='number', value=None, debounce=True,
                style={'width': '70px', 'font-size': '12px'}),
            dcc.Store(id='user-range-change', data=False),
        ], style={'display': 'flex', 'align-items': 'center', 'gap': '8px', 'flex-wrap': 'wrap'}),

    ], style={
        'backgroundColor': '#F0F0F0',
        'padding': '10px',
        'border-radius': '5px',
        'box-shadow': '0px 2px 5px rgba(0,0,0,0.1)'
    }),

    # Section Plot (full-width) with Store
    html.Div([
        dcc.Graph(id='section-plot', style={'width': '100%', 'height': '250px'}),
        dcc.Store(id='filtered-selection-range')
    ], style={
        'backgroundColor': '#F0F0F0',
        'padding': '10px',
        'borderRadius': '5px',
        'boxShadow': '0px 2px 5px rgba(0,0,0,0.1)',
        'marginBottom': '20px'
    }),

    # Scatter Plot with Sidebar (Click Output + Store)
    html.Div([
        # Scatter Plot (takes most space)
        dcc.Graph(id='scatter-plot', style={'flex': '7', 'height': '100%'}),

        # Click Output Panel
        html.Div([
            html.Div(id='click-output', style={
                'padding': '10px',
                'overflowX': 'auto',
                'textAlign': 'left',
                'backgroundColor': 'white',
                'height': '100%'
            }),
            dcc.Store(id='available-sensor-vars')
        ], style={'flex': '3', 'display': 'flex', 'flexDirection': 'column'})
    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'alignItems': 'stretch',
        'height': '500px',
        'marginBottom': '20px'
    }),

    # TS Plot and Profile Plot (side-by-side)
    html.Div([
        dcc.Graph(id='ts-plot', style={
            'width': '550px',
            'height': '500px',
            'padding': '10px'
        }),
        dcc.Graph(id='profile-plot', style={
            'width': '500px',
            'height': '550px',
            'padding': '10px'
        })
    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'justifyContent': 'left',
        'alignItems': 'left',
        'gap': '20px'
    })
])

# --- Callback to populate CSV selector and ensure valid default selection ---
@app.callback(
    Output("csv-selector", "options"),
    Output("csv-selector", "value"),
    Input("refresh-button", "n_clicks"),
    prevent_initial_call=True
)
def update_file_list(n_clicks):
    files = get_csv_files()
    options = [{'label': f, 'value': f} for f in files]
    new_value = files[-1] if files else None
    return options, new_value

# --- Callback: Update date and time range based on selected CSV file or section plot ---
@app.callback(
    [
        Output('start-date', 'value'),
        Output('end-date', 'value'),
        Output('start-time', 'value'),
        Output('end-time', 'value')
    ],
    [
        Input('csv-selector', 'value'),
        Input('section-plot', 'selectedData')
    ],
    [
        State('start-date', 'value'),
        State('end-date', 'value'),
        State('start-time', 'value'),
        State('end-time', 'value')
    ],
)
def update_time_range(csv_file, selected_data, start_date, end_date, start_time, end_time):
    triggered = ctx.triggered_id

    if triggered == 'csv-selector':
        if not csv_file:
            return None, None, None, None

        df = load_data(csv_file)
        if 'times' in df.columns and not df['times'].empty:
            min_time = df['times'].min()
            max_time = df['times'].max()
            return (
                min_time.strftime('%Y-%m-%d'),
                max_time.strftime('%Y-%m-%d'),
                min_time.strftime('%H:%M:%S'),
                max_time.strftime('%H:%M:%S')
            )
        else:
            return None, None, None, None

    elif triggered == 'section-plot' and selected_data and 'range' in selected_data and 'x' in selected_data['range']:
        try:
            x0, x1 = sorted(selected_data['range']['x'])
            start = pd.to_datetime(x0)
            end = pd.to_datetime(x1)
            return (
                start.strftime('%Y-%m-%d'),
                end.strftime('%Y-%m-%d'),
                start.strftime('%H:%M:%S'),
                end.strftime('%H:%M:%S')
            )
        except Exception:
            return start_date, end_date, start_time, end_time

    return start_date, end_date, start_time, end_time

# --- Callback: Update color variable options based on selected CSV file ---
@app.callback(
    [
        Output('color-variable', 'options'),
        Output('color-variable', 'value')
    ],
    [
        Input('csv-selector', 'value')
    ],
    [
        State('color-variable', 'value')
    ],
    prevent_initial_call=True
)
def update_color_variable_options(csv_file, current_color):
    if not csv_file:
        return [], None

    df = load_data(csv_file)
    sensor_vars = [col for col in df.columns if '_std' not in col and col not in meta_vars]
    color_options = [{'label': var.capitalize(), 'value': var} for var in sensor_vars]
    if current_color in sensor_vars:

        return color_options, current_color
    else:
        return color_options, (sensor_vars[0] if sensor_vars else None)

@app.callback(
    [
        Output('vmin-input', 'value'),
        Output('vmax-input', 'value')
    ],
    Input('color-variable', 'value'),
    prevent_initial_call=True
)
def reset_vmin_vmax(color_var):
    return None, None

# --- Callback: Section plot for time-latitude selection ---
@app.callback(
    Output('section-plot', 'figure'),
    Input('csv-selector', 'value')
)

def draw_section_plot(csv_file):
    if not csv_file:
        return px.scatter()

    df = load_data(csv_file)

    # Basic time-latitude scatter plot
    fig = px.scatter(df, x='times', y='latitude')
    fig.update_traces(
        mode='markers',
        selected=dict(marker=dict(color='red')),
        unselected=dict(marker=dict(color='blue'))
    )

    fig.update_layout(
        dragmode='select',
        selectdirection='h',
        xaxis=dict(
            title='Time',
            rangeslider=dict(visible=False),
            showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)',
            showline=True, linecolor='black', mirror=True
        ),
        yaxis=dict(
            title='Latitude',
            autorange=True,
            fixedrange=True,
            showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)',
            showline=True, linecolor='black', mirror=True
        ),
        paper_bgcolor='#F0F0F0',
        plot_bgcolor='white'
    )
    return fig

# --- Callback: Store selection range from scatter-plot for TS plot filtering ---
@app.callback(
    Output('filtered-selection-range', 'data'),
    Input('scatter-plot', 'selectedData'),
    prevent_initial_call=True
)
def store_selection_range(selectedData):
    if not selectedData or 'points' not in selectedData or not selectedData['points']:
        return None

    xs = [point['x'] for point in selectedData['points']]
    ys = [point['y'] for point in selectedData['points']]
    return {'x0': min(xs), 'x1': max(xs), 'y0': min(ys), 'y1': max(ys)}

# --- Callback: Track user range changes ---
@app.callback(
    Output('user-range-change', 'data'),
    [
        Input('lat-min', 'value'),
        Input('lat-max', 'value'),
        Input('z-min', 'value'),
        Input('z-max', 'value')
    ],
    [
        State('lat-min', 'value'),
        State('lat-max', 'value'),
        State('z-min', 'value'),
        State('z-max', 'value'),
        State('user-range-change', 'data')
    ],
    prevent_initial_call=True
)
def track_range_change(latmin, latmax, ymin, ymax,
                       prev_latmin, prev_latmax, prev_ymin, prev_ymax,
                       was_changed):
    # Compare with previous values
    changed = any([
        latmin != prev_latmin,
        latmax != prev_latmax,
        ymin != prev_ymin,
        ymax != prev_ymax
    ])
    return True if changed else was_changed

# # --- Callback: Download CSV file ---
# @app.callback(
#     Output('download-dataframe-csv', 'data'),
#     Input('download-button', 'n_clicks'),
#     State('csv-selector', 'value'),
#     prevent_initial_call=True
# )
# def download_csv(n_clicks, csv_file):
#     '''Download the selected CSV file.'''
#     if ctx.triggered_id == 'download-button' and csv_file:
#         df = load_data(csv_file)
#         return dcc.send_data_frame(df.to_csv, filename=f'{csv_file}.csv', index=False)
#     return no_update

@app.callback(
    [
        Output('scatter-plot', 'figure'),
        Output('available-sensor-vars', 'data')
    ],
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
        Input('z-min', 'value'),
        Input('z-max', 'value'),
        Input('lat-min', 'value'),
        Input('lat-max', 'value'),
        Input('filter-min', 'value'),
        Input('filter-max', 'value'),
        Input('hidden-opacity', 'value'),
        Input('fig-width', 'value'),
        Input('fig-height', 'value'),
        Input('bathymetry', 'value'),
        Input('station', 'value'),
        Input('user-range-change', 'data')
    ]
)
def update_plot(csv_file, start_date, start_time, end_date,  end_time,
                x_axis, color_var, color_map, vmin, vmax,
                zmin, zmax, latmin, latmax, filter_min, filter_max,
                hidden_opacity, fig_width, fig_height,
                bathymetry, station, user_changed_range):

    if not csv_file:
        return px.scatter(), []

    # Load and filter data
    df = load_data(csv_file)
    sensor_vars = [col for col in df.columns if '_std' not in col and col not in meta_vars]
    
    # Check x_axis variable
    if x_axis not in df.columns:
        x_axis = 'latitude'

    if filter_min is not None and filter_max is not None:
        df = df[df[color_var].between(filter_min, filter_max) & df[color_var].notna()]

    # Determine color range
    if vmin is None or vmax is None:
        q = df[color_var].quantile([0.05, 0.95])
        vmin = q[0.05] if vmin is None else vmin
        vmax = q[0.95] if vmax is None else vmax

    # Filter by time range
    # if time_range and 'x0' in time_range and 'x1' in time_range:
    #     x0, x1 = sorted([pd.to_datetime(time_range['x0']), pd.to_datetime(time_range['x1'])])
    #     df = df[df['times'].between(x0, x1)]
    
    # Date time filtering
    start_dt = pd.to_datetime(f"{start_date} {start_time}" if start_time else start_date, errors='coerce')
    end_dt = pd.to_datetime(f"{end_date} {end_time}" if end_time else end_date, errors='coerce')
    if pd.notnull(start_dt) and pd.notnull(end_dt):
        df = df[(df['times'] >= start_dt) & (df['times'] <= end_dt)] 
        
    # Create scatter plot
    fig = px.scatter(
        df,
        x=x_axis,
        y='depth',
        color=color_var,
        color_continuous_scale=color_map,
        range_color=[vmin, vmax],
        hover_data={'depth': ':.2f', 'latitude': ':.2f', 'longitude': ':.2f', color_var: ':.2f'},
        custom_data=['media', 'frame', 'times', 'latitude', 'longitude', 'link'] + sensor_vars
    )

    fig.update_traces(
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=hidden_opacity))
    )

    # Optional overlays
    if 'True' in bathymetry and (x_axis == 'latitude' or x_axis == 'latitude_bin'):
        fig.add_trace(go.Scatter(
            x=bathy['latitude'],
            y=bathy['bottom_depth_meters'],
            mode='lines',
            line=dict(color='black', width=1),
            name='Bathymetry',
            showlegend=False
        ))

    if 'True' in station and stations is not None and (x_axis == 'latitude' or x_axis == 'latitude_bin'):
        fig.update_layout(
            annotations=[
                dict(x=lat, y=1.05, xref='x', yref='paper', text=label,
                     showarrow=False, font=dict(size=12), align='center')
                for lat, label in zip(stations['latitude'], stations['station'])
            ],
            shapes=[
                dict(type='line', x0=lat, x1=lat, y0=1, y1=1.01, xref='x', yref='paper',
                     line=dict(color='black', width=1))
                for lat in stations['latitude']
            ]
        )

    # X-axis range logic (latitude has priority)
    if (x_axis == 'latitude' or x_axis == 'latitude_bin') and latmin is not None and latmax is not None:
        x_range = [latmax, latmin]
    else:
        x_range = [df['times'].min(), df['times'].max()] if x_axis == 'times' else None

    # Y-axis range logic (depth)
    if zmin is not None and zmax is not None:
        y_range = [zmax, zmin - 10]  # reversed axis for depth
    else:
        y_range = [df['depth'].max(), df['depth'].min()]
    
    fig.update_layout(
        dragmode="zoom",
        uirevision=None if user_changed_range else 'scatter-plot-static',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
        width=fig_width or None,
        height=fig_height or None,
        xaxis=dict(
            title=x_axis.capitalize(),
            range=x_range,
            showgrid=True, gridcolor='rgba(0,0,0,0.1)',
            showline=True, linecolor='black', mirror=True,
            ticks='outside', tickwidth=1, tickcolor='black',
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            title='Depth (m)',
            range=y_range,
            showgrid=True, gridcolor='rgba(0,0,0,0.1)',
            showline=True, linecolor='black', mirror=True,
            ticks='outside', tickwidth=1, tickcolor='black'
        ),
        coloraxis_colorbar=dict(
            title=dict(
                text=f'<br><br>&nbsp;{color_var.replace("_", " ").capitalize()} ({sled_units.get(color_var, "")})',
                side='right'
            )
        )
    )

    return fig, sensor_vars

@app.callback(
    Output('ts-plot', 'figure'),
    [
        Input('csv-selector', 'value'),
        Input('start-date', 'value'),
        Input('start-time', 'value'),
        Input('end-date', 'value'),
        Input('end-time', 'value'),
        Input('color-variable', 'value'),
        Input('color-map', 'value'),
        Input('vmin-input', 'value'),
        Input('vmax-input', 'value'),
        Input('filter-min', 'value'),
        Input('filter-max', 'value'),
        Input('hidden-opacity', 'value'),
        Input('fig-width', 'value'),
        Input('fig-height', 'value'),
        Input('filtered-selection-range', 'data')
    ]
)
def update_ts_plot(csv_file, start_date, start_time, end_date,  end_time,
                   color_var, color_map, vmin, vmax,
                   filter_min, filter_max, hidden_opacity,
                   fig_width, fig_height, select_range):
    
    if not csv_file:
        return px.scatter()

    df = load_data(csv_file)
    df = df.dropna(subset=['temperature', 'salinity'])

    sensor_vars = [col for col in df.columns if '_std' not in col and col not in meta_vars]

    # Grid for density contours
    tmin, tmax = df['temperature'].quantile([0.01, 0.99]).round().astype(int)
    smin, smax = df['salinity'].quantile([0.01, 0.99]).round().astype(int)
    tmin -= 2
    tmax += 2
    smin -= 2
    smax += 2

    T, S = np.meshgrid(np.arange(tmin, tmax, 0.5), np.arange(smin, smax, 0.5), indexing='ij')
    D = ies80(S, T, 0) - 1000

    # Apply value filter
    if filter_min is not None and filter_max is not None:
        df = df[df[color_var].between(filter_min, filter_max) & df[color_var].notna()]

    # Set color range dynamically if not provided
    if vmin is None or vmax is None:
        q = df[color_var].quantile([0.05, 0.95])
        vmin = q[0.05] if vmin is None else vmin
        vmax = q[0.95] if vmax is None else vmax

    # Date time filtering
    start_dt = pd.to_datetime(f"{start_date} {start_time}" if start_time else start_date, errors='coerce')
    end_dt = pd.to_datetime(f"{end_date} {end_time}" if end_time else end_date, errors='coerce')
    if pd.notnull(start_dt) and pd.notnull(end_dt):
        df = df[(df['times'] >= start_dt) & (df['times'] <= end_dt)] 

    # Prepare opacity masking based on zoom selection
    opacity_values = np.ones(len(df))
    if select_range and all(k in select_range for k in ['x0', 'x1', 'y0', 'y1']):
        lat0, lat1 = sorted([select_range['x0'], select_range['x1']])
        d0, d1 = sorted([select_range['y0'], select_range['y1']])
        mask = df['latitude'].between(lat0, lat1) & df['depth'].between(d0, d1)
        opacity_values = np.where(mask, 1, hidden_opacity/2)

    # Create main scatter plot
    fig = px.scatter(
        df,
        x='salinity',
        y='temperature',
        color=color_var,
        color_continuous_scale=color_map,
        range_color=[vmin, vmax],
        hover_data={
            'depth': ':.2f', 'latitude': ':.2f', 'longitude': ':.2f', color_var: ':.2f'
        },
        custom_data=['media', 'frame', 'times', 'latitude', 'longitude', 'link'] + sensor_vars
    )

    fig.update_traces(marker=dict(opacity=opacity_values))

    # Overlay simple black density contour lines
    fig.add_trace(go.Contour(
        z=D,
        x=np.arange(smin, smax, 0.5),
        y=np.arange(tmin, tmax, 0.5),
        colorscale=[[0, 'black'], [1, 'black']],  # force all lines to black
        contours=dict(
            coloring='lines',
            showlabels=True,
            labelfont=dict(size=10, color='black'),
        ),
        line=dict(color='black', width=1),
        hoverinfo='skip',
        showscale=False
    ))

    # Final layout config
    fig.update_layout(
        dragmode="zoom",
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
        width=fig_width or None,
        height=fig_width or None,  # ensure square shape
        yaxis=dict(
            title='Temperature (°C)',
            range=[tmin, tmax],
            showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)',
            showline=True, linecolor='black',
            mirror=True, ticks='outside', tickwidth=1, tickcolor='black'
        ),
        xaxis=dict(
            title='Salinity (psu)',
            range=[smin, smax],
            showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)',
            showline=True, linecolor='black',
            mirror=True, ticks='outside', tickwidth=1, tickcolor='black'
        ),
        coloraxis_colorbar=dict(
            title=dict(
                text=f'<br><br>&nbsp;{color_var.replace("_", " ").capitalize()} ({sled_units.get(color_var, "")})',
                side='right'
            )
        )
    )

    return fig

@app.callback(
    Output('profile-plot', 'figure'),
    [
        Input('csv-selector', 'value'),
        Input('start-date', 'value'),
        Input('start-time', 'value'),
        Input('end-date', 'value'),
        Input('end-time', 'value'),
        Input('color-variable', 'value'),
        Input('color-map', 'value'),
        Input('vmin-input', 'value'),
        Input('vmax-input', 'value'),
        Input('z-min', 'value'),
        Input('z-max', 'value'),
        Input('filter-min', 'value'),
        Input('filter-max', 'value'),
        Input('hidden-opacity', 'value'),
        Input('fig-width', 'value'),
        Input('fig-height', 'value'),
        Input('filtered-selection-range', 'data')
    ]
)
def update_profile_plot(csv_file, start_date, start_time, end_date,  end_time,
                        color_var, color_map, vmin, vmax, zmin, zmax,
                        filter_min, filter_max, hidden_opacity,
                        fig_width, fig_height, select_range):

    if not csv_file:
        return px.scatter()

    df = load_data(csv_file)

    # Apply value filter
    if filter_min is not None and filter_max is not None:
        df = df[df[color_var].between(filter_min, filter_max) & df[color_var].notna()]

    # Set color range dynamically if not provided
    if vmin is None or vmax is None:
        q = df[color_var].quantile([0.05, 0.95])
        vmin = q[0.05] if vmin is None else vmin
        vmax = q[0.95] if vmax is None else vmax

    # Date time filtering
    start_dt = pd.to_datetime(f"{start_date} {start_time}" if start_time else start_date, errors='coerce')
    end_dt = pd.to_datetime(f"{end_date} {end_time}" if end_time else end_date, errors='coerce')
    if pd.notnull(start_dt) and pd.notnull(end_dt):
        df = df[(df['times'] >= start_dt) & (df['times'] <= end_dt)] 

    if select_range and all(k in select_range for k in ['x0', 'x1', 'y0', 'y1']):
        lat0, lat1 = sorted([select_range['x0'], select_range['x1']])
        d0, d1 = sorted([select_range['y0'], select_range['y1']])
        mask = df['latitude'].between(lat0, lat1) & df['depth'].between(d0, d1)
        df = df[mask].reset_index(drop=True)
    
    if 'depth_bin' not in df.columns:
        step = 1
        df['depth_bin'] = np.floor((df['depth'] + step / 2) / step) * step
        
    main_summary = df.groupby('depth_bin')[color_var].agg(
        median=lambda x: x.median(),
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).reset_index()
    
    fig = go.Figure()

    # Line plot for median
    fig.add_trace(go.Scatter(
        x=main_summary['median'],
        y=main_summary['depth_bin'],
        mode='lines+markers',  # dots + line = "O-dot"
        line=dict(color='blue'),
        marker=dict(symbol='circle', size=6, color='blue'),
        showlegend=False
    ))
    
    # Shaded area between q25 and q75
    fig.add_trace(go.Scatter(
        x=main_summary['q25'],
        y=main_summary['depth_bin'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=main_summary['q75'],
        y=main_summary['depth_bin'],
        fill='tonextx',
        mode='lines',
        fillcolor='rgba(0, 0, 255, 0.3)',  # tab:blue with alpha
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        dragmode="zoom",
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
        width=fig_width or None,
        height=fig_width or None,  # ensure square shape
    )
    
    # Reverse y-axis if depth increases downward
    fig.update_yaxes(
        range=[zmax, zmin-10], title='Depth (m)',
        showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)',
        showline=True, linecolor='black', linewidth=1,
        mirror=True, ticks='outside', tickwidth=1, tickcolor='black'
        )
    
    fig.update_xaxes(
        title=f'{color_var.replace("_", " ").capitalize()} ({sled_units.get(color_var, "")})',
        showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)',
        showline=True, linecolor='black', linewidth=1,
        mirror=True, ticks='outside', tickwidth=1, tickcolor='black'
    )
    return fig

@app.callback(
    Output('click-output', 'children'),
    Input('scatter-plot', 'clickData'),
    State('available-sensor-vars', 'data')
)
def display_click_data(clickData, sensor_vars):
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
        for i, var in enumerate(sensor_vars or [], start=6):  # Start at index 5 since first vars are fixed
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
    app.run_server(host=args.host, port=args.port, processes=n_process, threaded=False, debug=False)