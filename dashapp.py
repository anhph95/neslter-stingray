import os
from pathlib import Path
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, ctx, Output, Input, State, Patch, no_update
import plotly.express as px
import plotly.graph_objects as go

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

# ========== Dash App Initialization ==========
app = dash.Dash(__name__)

# Number of processes for the server
n_process = min(os.cpu_count()-1, 8) 

# Define working directory and subdirectories
work_dir = Path('/dash_data') if Path('/dash_data').is_dir() else Path('dash_data')
data_dir, misc_dir = work_dir / 'data', work_dir / 'misc'

# Global cache for one dataset at a time
DATA_CACHE = {}
CURRENT_FILE = None

def load_data(file_name, sub_sample=1):
    """
    Load CSV file into a DataFrame, preprocess it, and filter invalid coordinates.
    Keeps only one file cached in memory at a time.
    """
    global DATA_CACHE, CURRENT_FILE

    # Return cached if same file already loaded
    if file_name == CURRENT_FILE and file_name in DATA_CACHE:
        df = DATA_CACHE[file_name]
    else:
        # Clear cache and load new file
        DATA_CACHE.clear()
        CURRENT_FILE = file_name

        df = pd.read_csv(f"{data_dir}/{file_name}.csv", low_memory=False)

        # Ensure 'times' column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['times']):
            df['times'] = pd.to_datetime(df['times'], errors='coerce', cache=True)

        # Add missing columns
        for col in ['media', 'frame', 'media_path', 'id', 'link', 'media_2', 'frame_2', 'media_path_2', 'id_2', 'link_2']:
            if col not in df.columns:
                df[col] = np.nan

        # Filter invalid coordinates
        if 'longitude' in df.columns and 'latitude' in df.columns:
            df = df[
                (df['longitude'].between(-180, 180, inclusive="both")) &
                (df['latitude'].between(-180, 180, inclusive="both"))
            ]

        # Save in cache
        DATA_CACHE[file_name] = df

    # Apply subsampling (only if > 0)
    if sub_sample and sub_sample > 0:
        return df.iloc[::sub_sample, :].reset_index(drop=True)
    else:
        return df.reset_index(drop=True)

# Initial dataset
csv_files = get_csv_files()
if csv_files:
    df = load_data(csv_files[-1])
else:
    df = pd.DataFrame()

# Define variables for dropdowns
meta_vars = ['timestamp', 'times', 'matdate', 'latitude', 'longitude', 'depth', 'media', 'media_path', 'frame', 'id', 'link', 'media_2', 'media_path_2', 'frame_2', 'id_2', 'link_2']
sensor_vars = [col for col in df.columns if '_std' not in col and col not in meta_vars] if not df.empty else []
colormaps = [name for name in px.colors.sequential.__dict__ if not name.startswith('_')]

# Load bathymetry and station data if files exist
load_csv = lambda file: pd.read_csv(file, dtype=str, encoding="utf-8") if file.exists() else None
stations, bathy = map(load_csv, [misc_dir / 'NESLTER_station_list.csv', misc_dir / 'NESLTER_transect_bathymetry.csv'])
stations['latitude'] = pd.to_numeric(stations['latitude'], errors='coerce')
bathy['latitude'] = pd.to_numeric(bathy['latitude'], errors='coerce')

# ========================
# App Layout
# ========================

app.layout = html.Div([

    # ===== TOP HEADER =====
    html.Div([
        # Left side: WHOI Logo
        html.Div([
            html.A(
                html.Img(
                    src='/assets/WHOI_OneLineLogo_WhiteType_RGB.png',
                    style={'height': '30px'}
                ),
                href="https://www.whoi.edu/",
                target="_blank"
            )
        ], style={
            'flex': '1',
            'display': 'flex',
            'justifyContent': 'flex-start',
            'alignItems': 'center'
        }),

        # Right side: LTER Network Logo
        html.Div([
            html.A(
                html.Img(
                    src='/assets/lter-network.png',
                    style={'height': '25px'}
                ),
                href="https://lternet.edu/",
                target="_blank"
            )
        ], style={
            'flex': '1',
            'display': 'flex',
            'justifyContent': 'flex-end',
            'alignItems': 'center'
        })
    ], style={
        'display': 'flex',
        'alignItems': 'center',
        'backgroundColor': "#183554",
        'padding': '5px 16px',
        'borderTopLeftRadius': '5px',
        'borderTopRightRadius': '5px',
        'box-shadow': '0px 2px 5px rgba(0,0,0,0.15)',
        'maxWidth': '99%',
        'margin': 'auto',
    }),


    # ===== SUB HEADER =====
    html.Div([
        # Left side: Title
        html.Div([
            html.Span(
                'STINGRAY DASHBOARD',
                style={
                    'color': "#183554",
                    'fontSize': '32px',
                    'fontWeight': 'bold'
                }
            )
        ], style={
            'flex': '1',
            'display': 'flex',
            'justifyContent': 'flex-start',
            'alignItems': 'center'
        }),

        # Right side: NES-LTER logo
        html.Div([
            html.A(
                html.Img(
                    src='/assets/NES-LTER-horizontal.png',
                    style={'height': '40px'}
                ),
                href="https://nes-lter.whoi.edu/",
                target="_blank"
            )
        ], style={
            'flex': '1',
            'display': 'flex',
            'justifyContent': 'flex-end',
            'alignItems': 'center'
        })
    ], style={
        'display': 'flex',
        'alignItems': 'center',
        'backgroundColor': 'white',
        'padding': '8px 16px',
        'borderBottomLeftRadius': '5px',
        'borderBottomRightRadius': '5px',
        'box-shadow': '0px 2px 5px rgba(0,0,0,0.1)',
        'maxWidth': '99%',
        'margin': 'auto',
        'marginBottom': '10px'
    }),

    # ===== MAIN BODY =====
    html.Div([

        # --- LEFT CONTROLS ---
        html.Div([

            # === Row 1: CSV selector → opacity ===
            html.Div([
                # Cruise Selector
                html.Div([
                    html.Label('Cruise:', style={'font-weight': 'bold', 'font-size': '20px'}),
                    dcc.Dropdown(
                        id='csv_selector',
                        options=[{'label': f, 'value': f} for f in csv_files],
                        value=csv_files[-1] if csv_files else None,
                        clearable=False,
                        style={'width': '100%', 'font-size': '15px'}
                    )
                ], style={'margin-bottom': '8px'}),

                # Refresh Button
                html.Button(
                    'Refresh',
                    id='refresh-button',
                    style={
                        'margin': '6px 0 10px 0',
                        'font-size': '15px',
                        'padding': '6px',
                        'background-color': '#007BFF',
                        'color': 'white',
                        'border': 'none',
                        'border-radius': '5px',
                        'cursor': 'pointer',
                        'marginRight': '10px'
                    }
                ),
                
                # # Download Button
                # html.Button(
                #     'Download',
                #     id='download-btn',
                #     style={
                #         'margin': '6px 0 10px 0',
                #         'font-size': '15px',
                #         'padding': '6px',
                #         'background-color': '#28A745',
                #         'color': 'white',
                #         'border': 'none',
                #         'border-radius': '5px',
                #         'cursor': 'pointer'
                #     }
                # ),                

                # Sub-sampling
                html.Div([
                    html.Label('Sub-sampling:'),
                    dcc.Input(
                        id='sub_sample', type='number', value=3,
                        style={'width': '100%', 'font-size': '15px'}
                    )
                ], style={'margin-bottom': '6px'}),

                # Opacity
                html.Div([
                    html.Label('Opacity:'),
                    dcc.Input(
                        id='hidden_opacity', type='number', value=0.05,
                        style={'width': '100%', 'font-size': '15px'}
                    )
                ], style={'margin-bottom': '6px'}),
            ], style={
                'aspectRatio': '4 / 1',   # matches section-plot
                'marginBottom': '10px',
                'backgroundColor': '#F0F0F0',
                'padding': '10px',
                'border-radius': '5px',
                'box-shadow': '0px 2px 5px rgba(0,0,0,0.1)'
            }),

            # === Row 2: Transect controls ===
            html.Div([
                html.Div([
                    html.Label('TRANSECT PLOT:', style={'font-weight': 'bold', 'font-size': '20px'})
                ], style={'margin-bottom': '8px'}),

                # Checklists
                dcc.Checklist(
                    id='bathymetry',
                    options=[{'label': 'Bathymetry', 'value': 'True'}],
                    value=['True'],
                    style={'font-size': '15px', 'margin-bottom': '6px'}
                ),
                dcc.Checklist(
                    id='station',
                    options=[{'label': 'Stations', 'value': 'True'}],
                    value=['True'],
                    style={'font-size': '15px', 'margin-bottom': '8px'}
                ),

                # Axis / Depth / Coordinates
                html.Div([
                    html.Label('X-Axis:'),
                    dcc.Dropdown(
                        id='x_axis_variable',
                        options=[{'label': lbl, 'value': val} for lbl, val in [
                            ('Latitude','latitude'), ('Longitude','longitude'), ('Time','times')]],
                        value='latitude',
                        style={'width': '100%', 'font-size': '15px'}
                    )
                ], style={'margin-bottom': '8px'}),

                # Color Variable
                html.Div([
                    html.Label('Color Variable:'),
                    dcc.Dropdown(
                        id='color_variable',
                        options=[{'label': var.capitalize(), 'value': var} for var in sensor_vars],
                        value='temperature',
                        style={'width': '100%', 'font-size': '15px'}
                    )
                ], style={'margin-bottom': '8px'}),

                # Colormap
                html.Div([
                    html.Label('Colormap:'),
                    dcc.Dropdown(
                        id='color_map',
                        options=[{'label': cmap, 'value': cmap} for cmap in colormaps],
                        value='Jet',
                        style={'width': '100%', 'font-size': '15px'}
                    )
                ], style={'margin-bottom': '8px'}),

                # Depth range
                html.Div([
                    html.Label('Min Depth:'),
                    dcc.Input(
                        id='z_min', type='number', value=0,
                        style={'width': '100%', 'font-size': '15px'}
                    )
                ], style={'margin-bottom': '6px'}),
                html.Div([
                    html.Label('Max Depth:'),
                    dcc.Input(
                        id='z_max', type='number', value=200,
                        style={'width': '100%', 'font-size': '15px'}
                    )
                ], style={'margin-bottom': '6px'}),

                # Coordinate range
                html.Div([
                    html.Label('Min Coordinate:'),
                    dcc.Input(
                        id='coord_min', type='number', value=None,
                        style={'width': '100%', 'font-size': '15px'}
                    )
                ], style={'margin-bottom': '6px'}),
                html.Div([
                    html.Label('Max Coordinate:'),
                    dcc.Input(
                        id='coord_max', type='number', value=None,
                        style={'width': '100%', 'font-size': '15px'}
                    )
                ], style={'margin-bottom': '8px'}),

                # Color range
                html.Div([
                    html.Label('Color Min:'),
                    dcc.Input(
                        id='v_min', type='number',
                        style={'width': '100%', 'font-size': '15px'}
                    )
                ], style={'margin-bottom': '6px'}),
                html.Div([
                    html.Label('Color Max:'),
                    dcc.Input(
                        id='v_max', type='number',
                        style={'width': '100%', 'font-size': '15px'}
                    )
                ], style={'margin-bottom': '6px'}),
            ], style={
                'aspectRatio': '2.5 / 1',   # matches scatter_plot
                'marginBottom': '10px',
                'backgroundColor': '#F0F0F0',
                'padding': '10px',
                'border-radius': '5px',
                'box-shadow': '0px 2px 5px rgba(0,0,0,0.1)'
            }),
            
            dcc.Store(id='user_range_change', data=False),  # track if user changed ranges

            # === Row 3: TS controls ===
            html.Div([
                html.Div([
                    html.Label('T-S PLOT:', style={'font-weight': 'bold', 'font-size': '20px'})
                ], style={'margin-bottom': '8px'}),

                # Color Variable
                html.Div([
                    html.Label('Color Variable:'),
                    dcc.Dropdown(
                        id='ts_color_variable',
                        options=[{'label': var.capitalize(), 'value': var} for var in sensor_vars if var not in ['temperature', 'salinity']],
                        value='chlorophyll',
                        style={'width': '100%', 'font-size': '15px'}
                    )
                ], style={'margin-bottom': '8px'}),

                # Colormap
                html.Div([
                    html.Label('Colormap:'),
                    dcc.Dropdown(
                        id='ts_color_map',
                        options=[{'label': cmap, 'value': cmap} for cmap in colormaps],
                        value='Viridis',
                        style={'width': '100%', 'font-size': '15px'}
                    )
                ], style={'margin-bottom': '8px'}),

                # Color range
                html.Div([
                    html.Label('Color Min:'),
                    dcc.Input(
                        id='ts_v_min', type='number',
                        style={'width': '100%', 'font-size': '15px'}
                    )
                ], style={'margin-bottom': '6px'}),
                html.Div([
                    html.Label('Color Max:'),
                    dcc.Input(
                        id='ts_v_max', type='number',
                        style={'width': '100%', 'font-size': '15px'}
                    )
                ], style={'margin-bottom': '6px'}),
            ], style={
                'aspectRatio': '2.5 / 1',   # matches ts+profile row
                'backgroundColor': '#F0F0F0',
                'padding': '10px',
                'border-radius': '5px',
                'box-shadow': '0px 2px 5px rgba(0,0,0,0.1)'
            })

        ], style={'flex': '2', 'padding': '8px', 'minWidth': '200px', 'maxWidth': '240px'}),

        # --- MIDDLE PLOTS ---
        html.Div([
            html.Div([dcc.Graph(id='section-plot')], style={'aspectRatio': '4 / 1', 'marginBottom': '10px', 'minHeight': '200px', 'width': '100%'}),
            dcc.Store(id='scatter_selected_data'),
            html.Div([dcc.Graph(id='scatter_plot')], style={'aspectRatio': '2.5 / 1', 'marginBottom': '10px', 'minHeight': '350px', 'width': '100%'}),
            html.Div([
                dcc.Graph(
                    id='ts_plot',
                    style={
                        'flex': '1 1 400px',
                        'margin': '5px',
                        'minWidth': '300px',
                        'minHeight': '350px',
                        'height': '100%',
                        'overflow': 'hidden'
                    }
                ),
                dcc.Graph(
                    id='profile_plot',
                    style={
                        'flex': '1 1 400px',
                        'margin': '5px',
                        'minWidth': '300px',
                        'minHeight': '350px',
                        'height': '100%',
                        'overflow': 'hidden'
                    }
                )
            ], style={
                'display': 'flex',
                'flexWrap': 'wrap',           # ✅ allows stacking on small screens
                'justifyContent': 'space-between',
                'alignItems': 'stretch',
                'width': '100%',
                'boxSizing': 'border-box',
                'overflow': 'hidden',
                'minHeight': '350px'
            })
        ], style={'flex': '6', 'padding': '8px', 'minWidth': '520px', 'maxWidth': 'calc(100% - 400px)'}),  # Adjusted maxWidth

        # --- RIGHT OUTPUT ---
        html.Div([
            # Row 1: datetime inputs
            html.Div([
                html.Label('Start Date:'), dcc.Input(id='start_date', type='text', placeholder='YYYY-MM-DD',
                    style={'width': '100%', 'font-size': '15px'}),
                html.Label('Start Time:'), dcc.Input(id='start_time', type='text', placeholder='HH:MM:SS',
                    style={'width': '100%', 'font-size': '15px'}),
                html.Label('End Date:'), dcc.Input(id='end_date', type='text', placeholder='YYYY-MM-DD',
                    style={'width': '100%', 'font-size': '15px'}),
                html.Label('End Time:'), dcc.Input(id='end_time', type='text', placeholder='HH:MM:SS',
                    style={'width': '100%', 'font-size': '15px'}),
            ], style={
                'aspectRatio': '4 / 1',
                'marginBottom': '10px',
                'backgroundColor': '#F0F0F0',
                'padding': '10px',
                'border-radius': '5px',
                'box-shadow': '0px 2px 5px rgba(0,0,0,0.1)'
            }),

            # Row 2: click output
            html.Div(id='click-output', style={
                'aspectRatio': '2.5 / 1',
                'marginBottom': '10px',
                'backgroundColor': 'white',
                'padding': '6px',
                'border-radius': '5px',
                'box-shadow': '0px 2px 5px rgba(0,0,0,0.1)'
            }),

            # Row 3: spacer to align
            html.Div([], style={
                'aspectRatio': '2.5 / 1',
            }),
            dcc.Store(id='available-sensor-vars')
        ], style={'flex': '2', 'padding': '8px', 'minWidth': '200px', 'maxWidth': '240px'})

    ], style={'display': 'flex', 'flexDirection': 'row'}),


    # ===== FOOTER (credits) =====
    html.Div([

        # Left side: Names
        html.Span(
            'Developed by: Anh Pham, Sidney Batchelder, Heidi Sosik',
            style={
                'color': 'white',
                'fontSize': '14px',
                'marginLeft': '8px'
            }
        ),

        # Right side: Email
        html.Span(
            'anh.pham@whoi.edu',
            style={
                'color': 'white',
                'fontSize': '14px',
                'marginRight': '8px'
            }
        )

    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',   # names left, email right
        'alignItems': 'center',
        'backgroundColor': "#183554",
        'padding': '6px 12px',
        'border-radius': '5px',
        'box-shadow': '0px -2px 5px rgba(0,0,0,0.1)',
        'maxWidth': '99%',
        'margin': '20px auto 0 auto'
    })

], style={
    'padding': '5px',
    'maxWidth': '99%',
    'margin': 'auto',
    'fontFamily': 'Arial'
})

# --- Callback to populate CSV selector and ensure valid default selection ---
@app.callback(
    Output("csv_selector", "options"),
    Output("csv_selector", "value"),
    Input("refresh-button", "n_clicks"),
    prevent_initial_call=True
)
def update_file_list(n_clicks):
    global DATA_CACHE, CURRENT_FILE
    DATA_CACHE.clear()
    CURRENT_FILE = None

    files = get_csv_files()
    options = [{'label': f, 'value': f} for f in files]
    new_value = files[-1] if files else None
    return options, new_value

# --- Callback: Update date and time range based on selected CSV file or section plot ---
@app.callback(
    [
        Output('start_date', 'value'),
        Output('end_date', 'value'),
        Output('start_time', 'value'),
        Output('end_time', 'value')
    ],
    [
        Input('csv_selector', 'value'),
        Input('section-plot', 'selectedData')
    ],
    [
        State('start_date', 'value'),
        State('end_date', 'value'),
        State('start_time', 'value'),
        State('end_time', 'value')
    ],
)
def update_time_range(csv_file, selected_data, start_date, end_date, start_time, end_time):
    triggered = ctx.triggered_id

    # --- CSV file changed ---
    if triggered == 'csv_selector':
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

    # --- Section plot triggered ---
    elif triggered == 'section-plot':
        df = load_data(csv_file)

        # If there IS a selection
        if selected_data and 'range' in selected_data and 'x' in selected_data['range']:
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

        # If NO selection (double-click reset) → fallback to dataset min/max
        elif 'times' in df.columns and not df['times'].empty:
            min_time = df['times'].min()
            max_time = df['times'].max()
            return (
                min_time.strftime('%Y-%m-%d'),
                max_time.strftime('%Y-%m-%d'),
                min_time.strftime('%H:%M:%S'),
                max_time.strftime('%H:%M:%S')
            )

    # --- Default: return whatever was there ---
    return start_date, end_date, start_time, end_time

# --- Callback: Update color variable options based on selected CSV file ---
@app.callback(
    [
        Output('color_variable', 'options'),
        Output('color_variable', 'value'),
        Output('ts_color_variable', 'options'),
        Output('ts_color_variable', 'value'),
    ],
    [
        Input('csv_selector', 'value')
    ],
    [
        State('color_variable', 'value'),
        State('ts_color_variable', 'value')
    ],
    prevent_initial_call=True
)

def update_color_variable_options(csv_file, current_color, current_ts_color):
    if not csv_file:
        return [], None, [], None

    df = load_data(csv_file)
    sensor_vars = [col for col in df.columns if '_std' not in col and col not in meta_vars]
    color_options = [{'label': var.capitalize(), 'value': var} for var in sensor_vars]
    if current_color in sensor_vars:
        return color_options, current_color, color_options, current_ts_color
    else:
        return color_options, (sensor_vars[0] if sensor_vars else None), color_options, (sensor_vars[0] if sensor_vars else None)

@app.callback(
    [
        Output('v_min', 'value'),
        Output('v_max', 'value')
    ],
    Input('color_variable', 'value'),
    prevent_initial_call=True
)

def reset_vmin_vmax(color_var):
    return None, None

@app.callback(
    [
        Output('ts_v_min', 'value'),
        Output('ts_v_max', 'value')
    ],
    Input('ts_color_variable', 'value'),
    prevent_initial_call=True
)

def reset_ts_vmin_vmax(color_var):
    return None, None

# --- Callback: Section plot for time-latitude selection ---
@app.callback(
    Output('section-plot', 'figure'),
    Input('csv_selector', 'value'),
    Input('sub_sample', 'value')
)

def draw_section_plot(csv_file, sub_sample):
    if not csv_file:
        return px.scatter()

    df = load_data(csv_file, sub_sample=sub_sample)

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
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    return fig

# # --- Callback: Store selection range from TS plot for scatter plot filtering ---
# @app.callback(
#     Output('ts_selected_data', 'data'),
#     Input('ts_plot', 'selectedData')
# )
# def store_ts_selection_indices(selectedData):
#     if not selectedData or "points" not in selectedData:
#         return None
    
#     # collect selected indices instead of x/y ranges
#     selected_ids = [p["customdata"][0] for p in selectedData["points"]]
#     return {"selected_ids": selected_ids}

# --- Callback: Mirror selection between scatter and TS plots ---
@app.callback(
    Output("scatter_plot", "figure", allow_duplicate=True),
    Output("ts_plot", "figure", allow_duplicate=True),
    Input("scatter_plot", "selectedData"),
    Input("ts_plot", "selectedData"),
    State("scatter_plot", "figure"),
    State("ts_plot", "figure"),
    prevent_initial_call=True
)
def mirror_selection(scatter_sel, ts_sel, fig_scatter, fig_ts):
    trigger = ctx.triggered_id

    # --- Determine selected point IDs ---
    selected_ids = None
    if trigger == "scatter_plot" and scatter_sel and scatter_sel.get("points"):
        selected_ids = [p.get("customdata", [None])[0] for p in scatter_sel["points"] if p.get("customdata")]
    elif trigger == "ts_plot" and ts_sel and ts_sel.get("points"):
        selected_ids = [p.get("customdata", [None])[0] for p in ts_sel["points"] if p.get("customdata")]
    else:
        # 👇 no selection → treat as "select all"
        selected_ids = None

    # --- Patch scatter ---
    patched_scatter = Patch()
    if "data" in fig_scatter:
        for i, _ in enumerate(fig_scatter["data"]):
            patched_scatter["data"][i]["selectedpoints"] = selected_ids

    # --- Patch TS ---
    patched_ts = Patch()
    if "data" in fig_ts:
        for i, _ in enumerate(fig_ts["data"]):
            patched_ts["data"][i]["selectedpoints"] = selected_ids

    return patched_scatter, patched_ts


# --- Callback: Store selection range from scatter_plot for TS plot filtering ---
@app.callback(
    Output('scatter_selected_data', 'data'),
    Input('scatter_plot', 'selectedData'),
    prevent_initial_call=True
)
def store_scatter_selection_indices(selectedData):
    if not selectedData or "points" not in selectedData:
        return None
    
    # collect selected indices instead of x/y ranges
    selected_ids = [p["customdata"][0] for p in selectedData["points"]]
    return {"selected_ids": selected_ids}

# --- Callback: Track user range changes ---
@app.callback(
    Output('user_range_change', 'data'),
    [
        Input('coord_min', 'value'),
        Input('coord_max', 'value'),
        Input('z_min', 'value'),
        Input('z_max', 'value')
    ],
    [
        State('coord_min', 'value'),
        State('coord_max', 'value'),
        State('z_min', 'value'),
        State('z_max', 'value'),
        State('user_range_change', 'data')
    ],
    prevent_initial_call=True
)
def track_range_change(coord_min, coord_max, ymin, ymax,
                       prev_coord_min, prev_coord_max, prev_ymin, prev_ymax,
                       was_changed):
    # Compare with previous values
    changed = any([
        coord_min != prev_coord_min,
        coord_max != prev_coord_max,
        ymin != prev_ymin,
        ymax != prev_ymax
    ])
    return True if changed else was_changed

# # --- Callback: Download CSV file ---
# @app.callback(
#     Output('download_dataframe_csv', 'data'),
#     Input('download-button', 'n_clicks'),
#     State('csv_selector', 'value'),
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
        Output('scatter_plot', 'figure'),
        Output('available-sensor-vars', 'data')
    ],
    [
        Input('csv_selector', 'value'),
        Input('sub_sample', 'value'),
        Input('start_date', 'value'),
        Input('start_time', 'value'),
        Input('end_date', 'value'),
        Input('end_time', 'value'),
        Input('x_axis_variable', 'value'),
        Input('color_variable', 'value'),
        Input('color_map', 'value'),
        Input('v_min', 'value'),
        Input('v_max', 'value'),
        Input('z_min', 'value'),
        Input('z_max', 'value'),
        Input('coord_min', 'value'),
        Input('coord_max', 'value'),
        # Input('filter_min', 'value'),
        # Input('filter_max', 'value'),
        Input('hidden_opacity', 'value'),
        Input('bathymetry', 'value'),
        Input('station', 'value'),
        Input('user_range_change', 'data'),
    ]
)

def update_plot(csv_file, sub_sample, start_date, start_time, end_date,  end_time,
                x_axis, color_var, color_map, vmin, vmax,
                zmin, zmax, coord_min, coord_max, 
                # filter_min, filter_max,
                hidden_opacity, bathymetry, station, user_changed_range):

    # --- Load Data ---
    if not csv_file:
        return go.Figure().add_annotation(text="⚠️ No CSV found", x=0.5, y=0.5, showarrow=False), []
    df = load_data(csv_file, sub_sample=sub_sample)
    sensor_vars = [col for col in df.columns if '_std' not in col and col not in meta_vars]

    
    # --- Filter Data ---
    # Date time
    start_dt = pd.to_datetime(f"{start_date} {start_time}" if start_time else start_date, errors='coerce')
    end_dt = pd.to_datetime(f"{end_date} {end_time}" if end_time else end_date, errors='coerce')
    if pd.notnull(start_dt) and pd.notnull(end_dt):
        df = df[(df['times'] >= start_dt) & (df['times'] <= end_dt)].reset_index(drop=True)
        
    # # Color
    # if filter_min is not None and filter_max is not None:
    #     df = df[(df[color_var].between(filter_min, filter_max)) & (df[color_var].notna())]
    # else:
    #     df = df[df[color_var].notna()]

    if df.empty:
        return go.Figure().add_annotation(
            text=f"⚠️ No {color_var} data available", 
            x=0.5, y=0.5, xref="paper", yref="paper", 
            showarrow=False, font=dict(size=16, color="red")
        ), []
    
    df['point_id'] = df.index
    
    # --- Plotting ---
    # Determine color range
    if vmin is None or vmax is None:
        q = df[color_var].quantile([0.05, 0.95])
        vmin = q[0.05] if vmin is None else vmin
        vmax = q[0.95] if vmax is None else vmax
        
    # Create scatter plot
    fig = px.scatter(
        df,
        x=x_axis,
        y='depth',
        color=color_var,
        color_continuous_scale=color_map,
        range_color=[vmin, vmax],
        hover_data={'depth': ':.2f', 'latitude': ':.2f', 'longitude': ':.2f', color_var: ':.2f'},
        custom_data=['point_id', 'media', 'frame', 'media_2', 'frame_2', 'times', 'latitude', 'longitude', 'link', 'link_2'] + sensor_vars
    )
    
    fig.update_traces(
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=hidden_opacity))
    )

    # X-axis range + formatting logic
    if x_axis == 'latitude':
        if coord_min is not None and coord_max is not None:
            x_range = [coord_max, coord_min]   # reversed so lat priority
        else:
            x_range = [df['latitude'].max() + 0.1, df['latitude'].min() - 0.1]

        # Generate ticks with N/S
        ticks = np.linspace(x_range[1], x_range[0], 6)  # 6 ticks
        tickvals = ticks
        ticktext = [
            f"{abs(v):.1f}°{'N' if v >= 0 else 'S'}"
            for v in ticks
        ]

    elif x_axis == 'longitude':
        if coord_min is not None and coord_max is not None:
            x_range = [coord_min, coord_max]
        else:
            x_range = [df['longitude'].min() - 0.1, df['longitude'].max() + 0.1]

        # Generate ticks with E/W
        ticks = np.linspace(x_range[0], x_range[1], 6)
        tickvals = ticks
        ticktext = [
            f"{abs(v):.1f}°{'E' if v >= 0 else 'W'}"
            for v in ticks
        ]

    elif x_axis == 'times':
        if coord_min is not None and coord_max is not None:
            x_range = [coord_min, coord_max]   # assuming datetime
        else:
            x_range = [df['times'].min(), df['times'].max()]

        tickvals = None
        ticktext = None

    else:
        x_range = None
        tickvals = None
        ticktext = None

    # Y-axis range logic (depth)
    if zmin is not None and zmax is not None:
        y_range = [zmax, zmin - 10]  # reversed axis for depth
    else:
        y_range = [df['depth'].max(), df['depth'].min()]
    
    fig.update_layout(
        dragmode="zoom",
        # uirevision=None if user_changed_range else 'scatter_plot-static',
        uirevision=None if user_changed_range else 'keep',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
        xaxis=dict(
            title=x_axis.capitalize(),
            range=x_range,
            tickvals=tickvals,
            ticktext=ticktext,
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
    
    # --- Optional overlays ---
    # Bathymetry
    if 'True' in bathymetry and bathy is not None and x_axis == 'latitude':
        bathy_mask = (bathy['latitude'] <= df['latitude'].max() + 0.1) & (bathy['latitude'] >= df['latitude'].min() - 0.1)
        fig.add_trace(
            go.Scatter(
                x=bathy['latitude'][bathy_mask],
                y=bathy['bottom_depth_meters'][bathy_mask],
                mode='lines',
                line=dict(color='black', width=1),
                name='Bathymetry',
                showlegend=False
            )
        )

    # Stations
    if 'True' in station and stations is not None and x_axis == 'latitude':
        fig.update_layout(
            annotations=[
                dict(x=lat, y=1.075, xref='x', yref='paper', text=label,
                     showarrow=False, font=dict(size=12), align='center')
                for lat, label in zip(stations['latitude'], stations['station'])
            ],
            shapes=[
                dict(type='line', x0=lat, x1=lat, y0=1, y1=1.015, xref='x', yref='paper',
                     line=dict(color='black', width=1))
                for lat in stations['latitude']
            ]
        )
    return fig, sensor_vars

@app.callback(
    Output('ts_plot', 'figure'),
    [
        Input('csv_selector', 'value'),
        Input('sub_sample', 'value'),
        Input('start_date', 'value'),
        Input('start_time', 'value'),
        Input('end_date', 'value'),
        Input('end_time', 'value'),
        Input('ts_color_variable', 'value'),
        Input('ts_color_map', 'value'),
        Input('ts_v_min', 'value'),
        Input('ts_v_max', 'value'),
        # Input('ts_filter_min', 'value'),
        # Input('ts_filter_max', 'value'),
        Input('hidden_opacity', 'value'),
        # Input('scatter_selected_data', 'data')
    ]
)
def update_ts_plot(csv_file, sub_sample, start_date, start_time, end_date,  end_time,
                   color_var, color_map, vmin, vmax,
                #    filter_min, filter_max, 
                   hidden_opacity):
    
    # --- Load Data ---
    if not csv_file:
        return go.Figure().add_annotation(text="⚠️ No CSV found", x=0.5, y=0.5, showarrow=False), []
    df = load_data(csv_file, sub_sample=sub_sample)
    sensor_vars = [col for col in df.columns if '_std' not in col and col not in meta_vars]
    
    # --- Filter Data ---
    # Date time
    start_dt = pd.to_datetime(f"{start_date} {start_time}" if start_time else start_date, errors='coerce')
    end_dt = pd.to_datetime(f"{end_date} {end_time}" if end_time else end_date, errors='coerce')
    if pd.notnull(start_dt) and pd.notnull(end_dt):
        df = df[(df['times'] >= start_dt) & (df['times'] <= end_dt)].reset_index(drop=True)



    # # Color
    # if filter_min is not None and filter_max is not None:
    #     df = df[(df[color_var].between(filter_min, filter_max)) & (df[color_var].notna())]
    # else:
    #     df = df[df[color_var].notna()]
    
    # Error message
    if df.empty:
        return go.Figure().add_annotation(
            text=f"⚠️ No {color_var} data available", 
            x=0.5, y=0.5, xref="paper", yref="paper", 
            showarrow=False, font=dict(size=16, color="red")
        )
    
    df['point_id'] = df.index
    
    # --- TS Contour ---
    # Grid for density contours
    tmin, tmax = df['temperature'].quantile([0.01, 0.99]).round().astype(int)
    smin, smax = df['salinity'].quantile([0.01, 0.99]).round().astype(int)
    tmin -= 2
    tmax += 2
    smin -= 2
    smax += 2

    T, S = np.meshgrid(np.arange(tmin, tmax, 0.5), np.arange(smin, smax, 0.5), indexing='ij')
    D = ies80(S, T, 0) - 1000
    
    # --- Plotting ---
    # Set color range dynamically if not provided
    if vmin is None or vmax is None:
        q = df[color_var].quantile([0.05, 0.95])
        vmin = q[0.05] if vmin is None else vmin
        vmax = q[0.95] if vmax is None else vmax
    
    # Create main scatter plot
    fig = px.scatter(
        df,
        x='salinity',
        y='temperature',
        color=color_var,
        color_continuous_scale=color_map,
        range_color=[vmin, vmax],
        hover_data={
            'temperature': ':.2f', 'salinity': ':.2f', 'depth': ':.2f', 'latitude': ':.2f', 'longitude': ':.2f', color_var: ':.2f'
        },
        custom_data=['point_id', 'media', 'frame', 'media_2', 'frame_2', 'times', 'latitude', 'longitude', 'link', 'link_2'] + sensor_vars
    )
    
    fig.update_traces(
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=hidden_opacity))
    )
    
    # Overlay simple black density contour lines
    fig.add_trace(
        go.Contour(
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
        )
    )

    # Final layout config
    fig.update_layout(
        dragmode="zoom",
        uirevision='keep',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
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
                side='top'   # title above
            ),
            orientation='h',   # horizontal bar
            x=0.5,             # center horizontally
            xanchor='center',
            y=1.075,             # push above plot
            len=0.75,          # shrink to 75% width
            thickness=15,
            ticks='outside',
            ticklabelposition="outside top"  # <-- this moves tick labels above
        ),
    )
    return fig

@app.callback(
    Output('profile_plot', 'figure'),
    [
        Input('csv_selector', 'value'),
        Input('sub_sample', 'value'),
        Input('start_date', 'value'),
        Input('start_time', 'value'),
        Input('end_date', 'value'),
        Input('end_time', 'value'),
        Input('color_variable', 'value'),
        Input('v_min', 'value'),
        Input('v_max', 'value'),
        Input('z_min', 'value'),
        Input('z_max', 'value'),
        # Input('filter_min', 'value'),
        # Input('filter_max', 'value'),
        Input('scatter_selected_data', 'data')
    ]
)
def update_profile_plot(csv_file, sub_sample, start_date, start_time, end_date,  end_time,
                        color_var, vmin, vmax, zmin, zmax,
                        # filter_min, filter_max, 
                        selected_data):

    # --- Load Data ---
    if not csv_file:
        return go.Figure().add_annotation(text="⚠️ No CSV found", x=0.5, y=0.5, showarrow=False), []
    df = load_data(csv_file, sub_sample=sub_sample)
    # sensor_vars = [col for col in df.columns if '_std' not in col and col not in meta_vars]
    
    # --- Filter Data ---
    # Date time
    start_dt = pd.to_datetime(f"{start_date} {start_time}" if start_time else start_date, errors='coerce')
    end_dt = pd.to_datetime(f"{end_date} {end_time}" if end_time else end_date, errors='coerce')
    if pd.notnull(start_dt) and pd.notnull(end_dt):
        df = df[(df['times'] >= start_dt) & (df['times'] <= end_dt)].reset_index(drop=True)
    
    df['point_id'] = df.index    
    
    # # Color
    # if filter_min is not None and filter_max is not None:
    #     df = df[(df[color_var].between(filter_min, filter_max)) & (df[color_var].notna())]
    # else:
    #     df = df[df[color_var].notna()]

    if df.empty:
        return go.Figure().add_annotation(
            text=f"⚠️ No {color_var} data available", 
            x=0.5, y=0.5, xref="paper", yref="paper", 
            showarrow=False, font=dict(size=16, color="red")
        )
    
    if selected_data and "selected_ids" in selected_data:
        df = df[df["point_id"].isin(selected_data["selected_ids"])].reset_index(drop=True)
        
    # --- Depth binning ---
    # Bin depth
    step = 1
    df['depth'] = np.floor((df['depth'] + step / 2) / step) * step
    
    main_summary = df.groupby('depth')[color_var].agg(
        median=lambda x: x.median(),
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).reset_index()
    
    # --- Plotting ---
    # Set color range dynamically if not provided
    if vmin is None or vmax is None:
        q = df[color_var].quantile([0.05, 0.95])
        vmin = q[0.05] if vmin is None else vmin
        vmax = q[0.95] if vmax is None else vmax

    # Line plot for median
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=main_summary['median'],
            y=main_summary['depth'],
            mode='lines+markers',  # dots + line = "O-dot"
            line=dict(color='blue'),
            marker=dict(symbol='circle', size=6, color='blue'),
            showlegend=False,
            hoverinfo='text',
            hovertemplate = (
                "<b>Depth:</b> %{y:.2f} m<br>" +
                "<b>" + color_var.capitalize() + ":</b> %{x:.2f} " + sled_units.get(color_var, "") + "<br>" +
                "<extra></extra>"
            )
        )
    )
    
    # Shaded area between q25 and q75
    fig.add_trace(
        go.Scatter(
            x=main_summary['q25'],
            y=main_summary['depth'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=main_summary['q75'],
            y=main_summary['depth'],
            fill='tonextx',
            mode='lines',
            fillcolor='rgba(0, 0, 255, 0.3)',  # tab:blue with alpha
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        )
    )

    fig.update_layout(
        dragmode="zoom",
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
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
    Input('scatter_plot', 'clickData'),
    State('available-sensor-vars', 'data')
)
def display_click_data(clickData, sensor_vars):
    '''Displays details of a clicked point on the scatter plot, showing all relevant variables.'''
    
    if not clickData or 'points' not in clickData or not clickData['points']:
        return 'Click on a point to see full details.'
    
    try:
        point = clickData['points'][0]
        custom_data = point.get('customdata', [])
        media       = custom_data[1] if len(custom_data) > 1 and pd.notna(custom_data[1]) else 'N/A'
        frame       = custom_data[2] if len(custom_data) > 2 and pd.notna(custom_data[2]) else 'N/A'
        media_2     = custom_data[3] if len(custom_data) > 3 and pd.notna(custom_data[3]) else 'N/A'
        frame_2     = custom_data[4] if len(custom_data) > 4 and pd.notna(custom_data[4]) else 'N/A'
        raw_time    = custom_data[5] if len(custom_data) > 5 and pd.notna(custom_data[5]) else None
        latitude    = custom_data[6] if len(custom_data) > 6 and pd.notna(custom_data[6]) else None
        longitude   = custom_data[7] if len(custom_data) > 7 and pd.notna(custom_data[7]) else None
        media_link  = custom_data[8] if len(custom_data) > 8 and pd.notna(custom_data[8]) else None
        media_link_2= custom_data[9] if len(custom_data) > 9 and pd.notna(custom_data[9]) else None
        depth = point.get('y', 'N/A')

        # Format time safely
        formatted_time = pd.to_datetime(raw_time).strftime('%Y-%m-%d %H:%M:%S') if raw_time else 'N/A'

        # Prepare variable details dynamically
        variable_details = []
        for i, var in enumerate(sensor_vars or [], start=10):  # Start at index 10 since first vars are fixed
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
                html.Span('📽️ ISIIS 1:', style={'flex': '3', 'text-align': 'left'}),
                html.A(media, href=media_link, target='_blank', style={'flex': '7', 'text-align': 'right'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),
            
            # html.Div([
            #     html.Span('🎞️ Frame 1:', style={'flex': '1', 'text-align': 'left'}),
            #     html.Span(frame, style={'flex': '1', 'text-align': 'right'})
            # ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),

            html.Div([
                html.Span('📽️ ISIIS 2:', style={'flex': '3', 'text-align': 'left'}),
                html.A(media_2, href=media_link_2, target='_blank', style={'flex': '7', 'text-align': 'right'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),
            
            # html.Div([
            #     html.Span('🎞️ Frame 2:', style={'flex': '1', 'text-align': 'left'}),
            #     html.Span(frame_2, style={'flex': '1', 'text-align': 'right'})
            # ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),
                        
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
    # Command-line arguments
    args = cli()
    args.port = str(args.port)
    app.run(host=args.host, port=args.port, processes=n_process, threaded=False, debug=False)
else:
    # WSGI-compatible Flask server (eg gunicorn)
    application = app.server
