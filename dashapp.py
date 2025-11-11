import os
import time
from pathlib import Path
from functools import lru_cache
from urllib.parse import urlencode, urlparse, parse_qs
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, ctx, Output, Input, State, Patch, no_update

# ============================================
# 🔹 Global Constants
# ============================================
DEFAULT_SUBSAMPLE = 3
MAX_WORKERS = min(os.cpu_count() - 1, 8)

WORK_DIR = Path("/dash_data") if Path("/dash_data").is_dir() else Path("dash_data")
DATA_DIR = WORK_DIR / "data"
MISC_DIR = WORK_DIR / "misc"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MISC_DIR.mkdir(parents=True, exist_ok=True)

# Cache structures
DATA_CACHE = {}
CURRENT_FILE = None
FILE_TIMESTAMP = {}

# ============================================
# 🔹 Units dictionary
# ============================================
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

# ============================================
# 🔹 IES80 Seawater Density Function
# ============================================
def ies80(s, t, p=0):
    """Compute seawater density using the IES80 equation."""
    s, t, p = map(np.asarray, (s, t, p))

    r0_coef = [
        999.842594, 6.793952e-2, -9.09529e-3, 1.001685e-4, -1.120083e-6,
        6.536332e-9, 8.24493e-1, -4.0899e-3, 7.6438e-5, -8.2467e-7,
        5.3875e-9, -5.72466e-3, 1.0227e-4, -1.6546e-6, 4.8314e-4
    ]

    r0 = (
        np.polyval(r0_coef[:6][::-1], t)
        + np.polyval(r0_coef[6:11][::-1], t) * s
        + np.polyval(r0_coef[11:14][::-1], t) * s**1.5
        + r0_coef[14] * s**2
    )

    if np.any(p):
        K_coef = [
            19652.21, 148.4206, -2.327105, 1.360447e-2, -5.155288e-5, 3.239908,
            1.43713e-3, 1.16092e-4, -5.77905e-7, 8.50935e-5, -6.12293e-6,
            5.2787e-8, 54.6746, -0.603459, 1.09987e-2, -6.1670e-5, 7.944e-2,
            1.6483e-2, -5.3009e-4, 2.2838e-3, -1.0981e-5, -1.6078e-6,
            1.91075e-4, -9.9348e-7, 2.0816e-8, 9.1697e-10
        ]
        K = (
            np.polyval(K_coef[:5][::-1], t)
            + np.polyval(K_coef[5:9][::-1], t) * p
            + np.polyval(K_coef[9:12][::-1], t) * p**2
            + np.polyval(K_coef[12:16][::-1], t) * s
            + np.polyval(K_coef[16:19][::-1], t) * s**1.5
            + np.polyval(K_coef[19:22][::-1], t) * p * s
            + K_coef[22] * p * s**1.5
            + np.polyval(K_coef[23:26][::-1], t) * p**2 * s
        )
        rho = r0 / (1 - p / K)
    else:
        rho = r0
    return rho

# ============================================
# 🔹 File Utilities
# ============================================
def get_csv_files() -> list[str]:
    """Return list of available CSV filenames (stem only)."""
    if not DATA_DIR.exists():
        return []
    return sorted(f.stem for f in DATA_DIR.glob("*.csv") if f.is_file())

@lru_cache(maxsize=4)
def load_csv(path: Path) -> pd.DataFrame | None:
    """Read a small auxiliary CSV file (stations, bathymetry)."""
    return pd.read_csv(path, dtype=str, encoding="utf-8") if path.exists() else None

# ============================================
# 🔹 Main Data Loader with Smart Caching
# ============================================
def load_data(file_name: str, sub_sample: int = 1) -> pd.DataFrame:
    """Load, preprocess, and optionally sub-sample a dataset."""
    global DATA_CACHE, CURRENT_FILE, FILE_TIMESTAMP
    csv_path = DATA_DIR / f"{file_name}.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    mtime = csv_path.stat().st_mtime
    # Use cache only if file unchanged
    if (file_name == CURRENT_FILE 
        and file_name in DATA_CACHE 
        and FILE_TIMESTAMP.get(file_name) == mtime):
        return DATA_CACHE[file_name].iloc[::sub_sample, :].reset_index(drop=True)

    # Load fresh
    df = pd.read_csv(csv_path, low_memory=False)

    # ✅ Ensure stable, persistent point_id across reloads and callbacks
    if "point_id" not in df.columns:
        df["point_id"] = np.arange(len(df))

    FILE_TIMESTAMP[file_name] = mtime
    CURRENT_FILE = file_name
    DATA_CACHE.clear()

    # Clean and standardize
    if "times" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["times"]):
        df["times"] = pd.to_datetime(df["times"], errors="coerce", cache=True)

    # Ensure media-related columns
    media_cols = [
        "media", "frame", "media_path", "id", "link",
        "media_2", "frame_2", "media_path_2", "id_2", "link_2"
    ]
    for col in media_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Clean coordinates
    if {"longitude", "latitude"}.issubset(df.columns):
        df = df[
            df["longitude"].between(-180, 180, inclusive="both") &
            df["latitude"].between(-90, 90, inclusive="both")
        ]

    # Convert numeric types
    for col in ["depth", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    DATA_CACHE[file_name] = df
    return df.iloc[::sub_sample, :].reset_index(drop=True)

# ============================================
# 🔹 CLI Interface
# ============================================
def cli():
    import argparse
    parser = argparse.ArgumentParser(description="Stingray Dashboard Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP for Dash app")
    parser.add_argument("--port", type=int, default=8050, help="Port for Dash app")
    return parser.parse_args()

# ============================================
# 🔹 Initialize Dash and Load Defaults
# ============================================
app = dash.Dash(__name__)

csv_files = get_csv_files()
df = load_data(csv_files[-1]) if csv_files else pd.DataFrame()

meta_vars = [
    "timestamp", "times", "matdate",
    "latitude", "longitude", "depth",
    "media", "media_path", "frame", "id", "link",
    "media_2", "media_path_2", "frame_2", "id_2", "link_2"
]

sensor_vars = [
    col for col in df.columns if "_std" not in col and col not in meta_vars
] if not df.empty else []

colormaps = [
    name for name in px.colors.sequential.__dict__ if not name.startswith("_")
]

stations = load_csv(MISC_DIR / "NESLTER_station_list.csv")
bathy = load_csv(MISC_DIR / "NESLTER_transect_bathymetry.csv")

if stations is not None:
    stations["latitude"] = pd.to_numeric(stations["latitude"], errors="coerce")

if bathy is not None:
    bathy["latitude"] = pd.to_numeric(bathy["latitude"], errors="coerce")
    bathy["bottom_depth_meters"] = pd.to_numeric(bathy["bottom_depth_meters"], errors="coerce")

# ========================
# App Layout (Updated)
# ========================

app.layout = html.Div([

    # --- URL Sync ---
    dcc.Location(id='url', refresh=False),

    # --- Auto file scanner ---
    dcc.Interval(id="file-scan-interval", interval=60 * 1000, n_intervals=0),
    dcc.Store(id="file-snapshot", data={}),

    # ===== TOP HEADER =====
    html.Div([
        html.A(
            html.Img(src='/assets/WHOI_OneLineLogo_WhiteType_RGB.png',
                     style={'height': '30px'}, title='WHOI Homepage'),
            href="https://www.whoi.edu/", target="_blank"
        ),
        html.A(
            html.Img(src='/assets/lter-network.png',
                     style={'height': '25px'}, title='LTER Network Homepage'),
            href="https://lternet.edu/", target="_blank"
        )
    ], className='header-top flex-row'),

    # ===== SUB HEADER =====
    html.Div([
        html.Span('STINGRAY DASHBOARD', className='header-title'),
        html.A(
            html.Img(src='/assets/NES-LTER-horizontal.png',
                     style={'height': '40px'}, title='NES-LTER Homepage'),
            href="https://nes-lter.whoi.edu/", target="_blank"
        )
    ], className='header-sub flex-row'),

    # ===== MAIN BODY =====
    html.Div([

        # --- LEFT CONTROLS ---
        html.Div([

            # Row 1: File + cruise axis controls
            html.Div([
                html.Label('Dataset:', style={'font-size': '20px'}),
                dcc.Dropdown(
                    id='csv_selector',
                    options=[{'label': f, 'value': f} for f in csv_files],
                    value=csv_files[-1] if csv_files else None,
                    clearable=False
                ),
                html.Button('Refresh list', id='refresh-button'),

                html.Div([
                    html.Label('Cruise track X-axis:'),
                    dcc.Dropdown(
                        id='cruise_track_xaxis',
                        options=[{'label': f.capitalize(), 'value': f} for f in ['times', 'latitude', 'longitude', 'depth']],
                        value='times'
                    )
                ], style={'marginTop': '6px'}),

                html.Div([
                    html.Label('Cruise track Y-axis:'),
                    dcc.Dropdown(
                        id='cruise_track_yaxis',
                        options=[{'label': f.capitalize(), 'value': f} for f in ['times', 'latitude', 'longitude', 'depth']],
                        value='latitude'
                    )
                ])
            ], className='panel'),

            # Row 2: Transect controls
            html.Div([
                html.Label('TRANSECT PLOT:', className='section-label'),

                dcc.Checklist(
                    id='bathymetry',
                    options=[{'label': 'Bathymetry', 'value': 'True'}],
                    value=['True']
                ),
                dcc.Checklist(
                    id='station',
                    options=[{'label': 'Stations', 'value': 'True'}],
                    value=['True']
                ),

                html.Div([
                    html.Label('X-Axis:'),
                    dcc.Dropdown(
                        id='x_axis_variable',
                        options=[{'label': var.capitalize(), 'value': var}
                                 for var in ['latitude', 'longitude', 'times']],
                        value='latitude'
                    )
                ]),
                html.Div([
                    html.Label('Color Variable:'),
                    dcc.Dropdown(
                        id='color_variable',
                        options=[{'label': var.capitalize(), 'value': var} for var in sensor_vars],
                        value='temperature'
                    )
                ]),
                html.Div([
                    html.Label('Colormap:'),
                    dcc.Dropdown(
                        id='color_map',
                        options=[{'label': cmap, 'value': cmap} for cmap in colormaps],
                        value='Jet'
                    )
                ]),
                html.Div([
                    html.Label('Min Depth:'),
                    dcc.Input(id='z_min', type='number', value=0)
                ]),
                html.Div([
                    html.Label('Max Depth:'),
                    dcc.Input(id='z_max', type='number', value=200)
                ]),
                html.Div([
                    html.Label('Min Coordinate:'),
                    dcc.Input(id='coord_min', type='number', value=None)
                ]),
                html.Div([
                    html.Label('Max Coordinate:'),
                    dcc.Input(id='coord_max', type='number', value=None)
                ]),
                html.Div([
                    html.Label('Color Min:'),
                    dcc.Input(id='v_min', type='number')
                ]),
                html.Div([
                    html.Label('Color Max:'),
                    dcc.Input(id='v_max', type='number')
                ])
            ], className='panel'),

            dcc.Store(id='user_range_change', data=False),

            # Row 3: TS controls
            html.Div([
                html.Label('T-S PLOT:', className='section-label'),
                html.Div([
                    html.Label('Color Variable:'),
                    dcc.Dropdown(
                        id='ts_color_variable',
                        options=[{'label': var.capitalize(), 'value': var}
                                 for var in sensor_vars if var not in ['temperature', 'salinity']],
                        value='chlorophyll'
                    )
                ]),
                html.Div([
                    html.Label('Colormap:'),
                    dcc.Dropdown(
                        id='ts_color_map',
                        options=[{'label': cmap, 'value': cmap} for cmap in colormaps],
                        value='Viridis'
                    )
                ]),
                html.Div([
                    html.Label('Color Min:'),
                    dcc.Input(id='ts_v_min', type='number')
                ]),
                html.Div([
                    html.Label('Color Max:'),
                    dcc.Input(id='ts_v_max', type='number')
                ])
            ], className='panel'),

            # Row 4: Other options (moved to bottom)
            html.Div([
                html.Label('OTHER OPTIONS:', className='section-label'),
                html.Div([
                    html.Label('Sub-sampling:'),
                    dcc.Input(id='sub_sample', type='number', value=3)
                ]),
                html.Div([
                    html.Label('Opacity:'),
                    dcc.Input(id='hidden_opacity', type='number', value=0.05)
                ])
            ], className='panel')

        ], className='left-panel'),

        # --- MIDDLE PLOTS ---
        html.Div([
            html.Div([dcc.Graph(id='cruise_track')], className='cruise-track-graph'),
            dcc.Store(id='cruise_track_selected_data'),
            dcc.Store(id='main_plot_selected_data'),
            html.Div([dcc.Graph(id='main_plot')], className='main-graph'),
            html.Div([dcc.Graph(id='ts_plot')], className='ts-graph'),
            html.Div([dcc.Graph(id='profile_plot')], className='profile-graph'),
        ], className='middle-panel'),

        # --- RIGHT PANEL (Selected Data Info only) ---
        html.Div([
            html.Label('Details:', className='section-label', style={'font-size': '16px'}),
            html.Div(id='click-output', className='card', style={
                'font-size': '13px',
                'line-height': '1.4em'
            }),
            dcc.Store(id='available-sensor-vars')
        ], className='right-panel', style={
            'flex': '3.5',        # wider
            'minWidth': '320px',
            'maxWidth': '480px'
        })

    ], className='flex-row'),

    # ===== FOOTER =====
    html.Div([
        html.Span('Developed by: Anh Pham, Sidney Batchelder, Heidi Sosik'),
        html.Span('anh.pham@whoi.edu')
    ], className='footer')
])

# ============================================================
# === URL Synchronization & Dataset Management Callbacks ===
# ============================================================

# Define which UI parameters should be mirrored in the URL
URL_SYNCED_PARAMS = [
    {"key": "dataset",   "id": "csv_selector",   "default": None,          "type": "string"},
    {"key": "variable",  "id": "color_variable", "default": "temperature", "type": "string"},
    {"key": "colormap",  "id": "color_map",      "default": "Jet",         "type": "string"},
    {"key": "zmin",      "id": "z_min",          "default": 0,             "type": "float"},
    {"key": "zmax",      "id": "z_max",          "default": 200,           "type": "float"},
    {"key": "opacity",   "id": "hidden_opacity", "default": 0.05,          "type": "float"},
    {"key": "subsample", "id": "sub_sample",     "default": 3,             "type": "int"},
]


# --- Callback: Update URL query string based on current UI state ---
@app.callback(
    Output("url", "search"),
    [Input(p["id"], "value") for p in URL_SYNCED_PARAMS],
    prevent_initial_call=True
)
def update_url(*values):
    """Synchronize key UI parameters with the browser URL."""
    params = {
        p["key"]: val for p, val in zip(URL_SYNCED_PARAMS, values)
        if val not in [None, "", []]
    }
    return f"?{urlencode(params)}" if params else ""


# --- Callback: Restore UI state from URL query string ---
@app.callback(
    [
        Output(p["id"], "value", allow_duplicate=True)
        for p in URL_SYNCED_PARAMS
    ],
    Input("url", "search"),
    Input("csv_selector", "options"),  # wait until dataset options loaded
    prevent_initial_call="initial_duplicate",
)
def restore_from_url(search, csv_options):
    """Restore UI element values from URL parameters."""
    csv_files = [opt["value"] for opt in (csv_options or [])]
    defaults = {
        p["key"]: (
            csv_files[-1] if p["key"] == "dataset" and csv_files else p["default"]
        )
        for p in URL_SYNCED_PARAMS
    }

    if not search:
        return [defaults[p["key"]] for p in URL_SYNCED_PARAMS]

    params = parse_qs(urlparse(search).query)

    results = []
    for p in URL_SYNCED_PARAMS:
        key, typ = p["key"], p["type"]
        val = params.get(key, [defaults[key]])[0]

        # --- type casting ---
        if typ in ("int", "float"):
            try:
                val = int(val) if typ == "int" else float(val)
            except (ValueError, TypeError):
                val = defaults[key]

        # --- ensure dataset exists ---
        if key == "dataset" and val not in csv_files:
            val = defaults[key]

        results.append(val)

    return results


# --- Callback: Refresh available CSV file list ---
@app.callback(
    Output("csv_selector", "options"),
    Output("csv_selector", "value"),
    Input("refresh-button", "n_clicks"),
    prevent_initial_call=True
)
def update_file_list(n_clicks):
    """Refresh the list of available CSV datasets."""
    global DATA_CACHE, CURRENT_FILE
    DATA_CACHE.clear()
    CURRENT_FILE = None

    files = get_csv_files()
    options = [{'label': f, 'value': f} for f in files]
    new_value = files[-1] if files else None
    return options, new_value


# ============================================================
# === Color Variable and Range Management ===
# ============================================================

# --- Callback: Update color-variable dropdowns when CSV changes ---
@app.callback(
    [
        Output('color_variable', 'options'),
        Output('color_variable', 'value'),
        Output('ts_color_variable', 'options'),
        Output('ts_color_variable', 'value'),
    ],
    Input('csv_selector', 'value'),
    State('color_variable', 'value'),
    State('ts_color_variable', 'value'),
    prevent_initial_call=True
)
def update_color_variable_options(csv_file, current_color, current_ts_color):
    """Update available sensor/color variable lists when dataset changes."""
    if not csv_file:
        return [], None, [], None

    df = load_data(csv_file)
    sensor_vars = [col for col in df.columns if '_std' not in col and col not in meta_vars]
    color_options = [{'label': var.capitalize(), 'value': var} for var in sensor_vars]

    if current_color in sensor_vars:
        return color_options, current_color, color_options, current_ts_color
    else:
        default = sensor_vars[0] if sensor_vars else None
        return color_options, default, color_options, default


# --- Callback: Reset main plot color limits when color variable changes ---
@app.callback(
    Output('v_min', 'value'),
    Output('v_max', 'value'),
    Input('color_variable', 'value'),
    prevent_initial_call=True
)
def reset_vmin_vmax(color_var):
    """Reset main plot color scale limits."""
    return None, None


# --- Callback: Reset time-series color limits when color variable changes ---
@app.callback(
    Output('ts_v_min', 'value'),
    Output('ts_v_max', 'value'),
    Input('ts_color_variable', 'value'),
    prevent_initial_call=True
)
def reset_ts_vmin_vmax(color_var):
    """Reset time-series color scale limits."""
    return None, None


# ============================================================
# === Cruise Track Plot (Latitude vs. Time or Longitude) ===
# ============================================================

@app.callback(
    Output('cruise_track', 'figure'),
    Input('csv_selector', 'value'),
    Input('cruise_track_xaxis', 'value'),
    Input('cruise_track_yaxis', 'value'),
    Input('sub_sample', 'value'),
)
def draw_cruise_track(csv_file, xaxis, yaxis, sub_sample):
    """Draw the cruise track scatter plot (latitude vs. selected X-axis)."""
    if not csv_file:
        return px.scatter()

    df = load_data(csv_file, sub_sample=sub_sample)
    #df['point_id'] = df.index

    # --- Base scatter plot ---
    fig = px.scatter(df, x=xaxis, y=yaxis, custom_data=['point_id'])
    fig.update_traces(
        mode='markers',
        selected=dict(marker=dict(color='red')),
        unselected=dict(marker=dict(color='blue'))
    )

    # --- Layout setup ---
    fig.update_layout(
        dragmode='select',
        selectdirection='any',  # allow box/lasso in any direction
        newselection_mode='immediate',
        clickmode='event+select',
        xaxis=dict(
            title=xaxis.capitalize(),
            rangeslider=dict(visible=False),
            showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)',
            showline=True, linecolor='black', mirror=True
        ),
        yaxis=dict(
            title=yaxis.capitalize(),
            autorange=True,
            fixedrange=False,
            showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)',
            showline=True, linecolor='black', mirror=True
        ),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    # --- Enforce equal degree scaling for lat/lon plots ---
    if xaxis in ['longitude', 'latitude']:
        fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))

    return fig


# ============================================================
# === Selections & Range Change Tracking ===
# ============================================================

# --- Callback: Store selected point IDs from the cruise track plot ---
@app.callback(
    Output("cruise_track_selected_data", "data"),
    Input("cruise_track", "selectedData"),
    State("csv_selector", "value"),
    prevent_initial_call=True
)
def store_cruise_track_selection(selectedData, csv_file):
    """Store selected point IDs from the cruise track plot.
    If selection is cleared (double-click), return all points instead of empty.
    """
    df = load_data(csv_file)
    #df['point_id'] = df.index

    if not selectedData or "points" not in selectedData or not selectedData["points"]:
        selected_ids = df["point_id"].tolist()
    else:
        selected_ids = [p["customdata"][0] for p in selectedData["points"] if p.get("customdata")]

    # Include timestamp so dependent callbacks always update
    return {"selected_ids": selected_ids, "time": time.time()}


# --- Callback: Mirror selections between main scatter and time-series plots ---
@app.callback(
    Output("main_plot", "figure", allow_duplicate=True),
    Output("ts_plot", "figure", allow_duplicate=True),
    Input("main_plot", "selectedData"),
    Input("ts_plot", "selectedData"),
    State("main_plot", "figure"),
    State("ts_plot", "figure"),
    prevent_initial_call=True
)
def mirror_selection(scatter_sel, ts_sel, fig_scatter, fig_ts):
    """Synchronize selections between scatter and time-series plots."""
    trigger = ctx.triggered_id
    selected_ids = None

    if trigger == "main_plot" and scatter_sel and scatter_sel.get("points"):
        selected_ids = [p.get("customdata", [None])[0] for p in scatter_sel["points"] if p.get("customdata")]
    elif trigger == "ts_plot" and ts_sel and ts_sel.get("points"):
        selected_ids = [p.get("customdata", [None])[0] for p in ts_sel["points"] if p.get("customdata")]

    # None means “select all”
    patched_scatter = Patch()
    patched_ts = Patch()

    if "data" in fig_scatter:
        for i, _ in enumerate(fig_scatter["data"]):
            patched_scatter["data"][i]["selectedpoints"] = selected_ids

    if "data" in fig_ts:
        for i, _ in enumerate(fig_ts["data"]):
            patched_ts["data"][i]["selectedpoints"] = selected_ids

    return patched_scatter, patched_ts


# --- Callback: Store selected IDs from main plot (for cross-plot filtering) ---
@app.callback(
    Output('main_plot_selected_data', 'data'),
    Input('main_plot', 'selectedData'),
    prevent_initial_call=True
)
def store_scatter_selection_indices(selectedData):
    """Store IDs of selected points in main scatter plot."""
    if not selectedData or "points" not in selectedData:
        return None
    selected_ids = [p["customdata"][0] for p in selectedData["points"]]
    return {"selected_ids": selected_ids}


# --- Callback: Track user coordinate/range edits ---
@app.callback(
    Output('user_range_change', 'data'),
    Input('coord_min', 'value'),
    Input('coord_max', 'value'),
    Input('z_min', 'value'),
    Input('z_max', 'value'),
    State('coord_min', 'value'),
    State('coord_max', 'value'),
    State('z_min', 'value'),
    State('z_max', 'value'),
    State('user_range_change', 'data'),
    prevent_initial_call=True
)
def track_range_change(coord_min, coord_max, ymin, ymax,
                       prev_coord_min, prev_coord_max, prev_ymin, prev_ymax,
                       was_changed):
    """Track if the user manually changed coordinate or depth limits."""
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

# ============================================================
# === Main Plot Update: Depth vs. Variable / Coordinate ===
# ============================================================

@app.callback(
    [
        Output('main_plot', 'figure'),
        Output('available-sensor-vars', 'data')
    ],
    [
        Input('csv_selector', 'value'),
        Input('sub_sample', 'value'),
        Input('x_axis_variable', 'value'),
        Input('color_variable', 'value'),
        Input('color_map', 'value'),
        Input('v_min', 'value'),
        Input('v_max', 'value'),
        Input('z_min', 'value'),
        Input('z_max', 'value'),
        Input('coord_min', 'value'),
        Input('coord_max', 'value'),
        Input('hidden_opacity', 'value'),
        Input('bathymetry', 'value'),
        Input('station', 'value'),
        Input('user_range_change', 'data'),
        Input("cruise_track_selected_data", "modified_timestamp"),
    ],
    State('cruise_track_selected_data', 'data'),
)
def update_main_plot(csv_file, sub_sample,
                     x_axis, color_var, color_map, vmin, vmax,
                     zmin, zmax, coord_min, coord_max,
                     hidden_opacity, bathymetry, station,
                     user_changed_range, cruise_track_flag, cruise_track_selection):
    """
    Update the main scatter plot (depth vs. chosen variable or coordinate).

    Triggered by:
        - Any change in dataset, color, or axis configuration
        - Range adjustments
        - Cruise track selection changes (via modified_timestamp)

    Notes:
        - Cruise track selection is applied FIRST (priority filter)
        - Automatically adapts color scale, ticks, and layout
        - Handles empty data gracefully
    """

    # --------------------------------------------------------
    # 1️⃣ Load Data
    # --------------------------------------------------------
    if not csv_file:
        return (
            go.Figure().add_annotation(
                text="⚠️ No CSV found", x=0.5, y=0.5, showarrow=False
            ),
            []
        )

    df = load_data(csv_file, sub_sample=sub_sample)
    #df['point_id'] = df.index
    sensor_vars = [col for col in df.columns if '_std' not in col and col not in meta_vars]

    # --------------------------------------------------------
    # 2️⃣ Apply Cruise Track Selection (if any)
    # --------------------------------------------------------
    if cruise_track_selection and "selected_ids" in cruise_track_selection:
        selected_ids = cruise_track_selection["selected_ids"]
        df = df[df['point_id'].isin(selected_ids)].reset_index(drop=True)

    if df.empty:
        return (
            go.Figure().add_annotation(
                text=f"⚠️ No {color_var} data available",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=16, color="red")
            ),
            []
        )

    # --------------------------------------------------------
    # 3️⃣ Configure Color Scale
    # --------------------------------------------------------
    if vmin is None or vmax is None:
        q = df[color_var].quantile([0.05, 0.95])
        vmin = q[0.05] if vmin is None else vmin
        vmax = q[0.95] if vmax is None else vmax

    # --------------------------------------------------------
    # 4️⃣ Create Scatter Plot
    # --------------------------------------------------------
    fig = px.scatter(
        df,
        x=x_axis,
        y='depth',
        color=color_var,
        color_continuous_scale=color_map,
        range_color=[vmin, vmax],
        hover_data={
            'depth': ':.2f',
            'latitude': ':.2f',
            'longitude': ':.2f',
            color_var: ':.2f'
        },
        custom_data=['point_id', 'media', 'frame', 'media_2', 'frame_2',
                     'times', 'latitude', 'longitude', 'link', 'link_2'] + sensor_vars
    )

    fig.update_traces(
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=hidden_opacity))
    )

    # --------------------------------------------------------
    # 5️⃣ Axis Ranges & Tick Formatting
    # --------------------------------------------------------
    if x_axis == 'latitude':
        # Latitude axis — reversed so north (max) is left
        x_range = [coord_max, coord_min] if (coord_min and coord_max) else \
                  [df['latitude'].max() + 0.1, df['latitude'].min() - 0.1]
        ticks = np.linspace(x_range[1], x_range[0], 6)
        tickvals = ticks
        ticktext = [f"{abs(v):.1f}°{'N' if v >= 0 else 'S'}" for v in ticks]

    elif x_axis == 'longitude':
        x_range = [coord_min, coord_max] if (coord_min and coord_max) else \
                  [df['longitude'].min() - 0.1, df['longitude'].max() + 0.1]
        ticks = np.linspace(x_range[0], x_range[1], 6)
        tickvals = ticks
        ticktext = [f"{abs(v):.1f}°{'E' if v >= 0 else 'W'}" for v in ticks]

    elif x_axis == 'times':
        x_range = [coord_min, coord_max] if (coord_min and coord_max) else \
                  [df['times'].min(), df['times'].max()]
        tickvals, ticktext = None, None

    else:
        x_range, tickvals, ticktext = None, None, None

    # Depth axis (reversed)
    y_range = [zmax, zmin - 10] if (zmin is not None and zmax is not None) else \
              [df['depth'].max(), df['depth'].min()]

    # --------------------------------------------------------
    # 6️⃣ Layout & Appearance
    # --------------------------------------------------------
    fig.update_layout(
        dragmode="zoom",
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
                text=f'<br><br>&nbsp;{color_var.replace("_", " ").capitalize()} '
                     f'({sled_units.get(color_var, "")})',
                side='bottom'
            ),
            orientation='h',
            x=0.5, xanchor='center',
            y=-1, len=0.75, thickness=15,
            ticks='outside',
            ticklabelposition='outside bottom'
        )
    )

    # --------------------------------------------------------
    # 7️⃣ Optional Overlays (Bathymetry & Stations)
    # --------------------------------------------------------
    if 'True' in bathymetry and bathy is not None and x_axis == 'latitude':
        bathy_mask = (
            (bathy['latitude'] <= df['latitude'].max() + 0.1) &
            (bathy['latitude'] >= df['latitude'].min() - 0.1)
        )
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

    if 'True' in station and stations is not None and x_axis == 'latitude':
        fig.update_layout(
            annotations=[
                dict(
                    x=lat, y=1.25, xref='x', yref='paper',
                    text=label, showarrow=False,
                    font=dict(size=12), align='center'
                )
                for lat, label in zip(stations['latitude'], stations['station'])
            ],
            shapes=[
                dict(
                    type='line', x0=lat, x1=lat, y0=1, y1=1.05,
                    xref='x', yref='paper',
                    line=dict(color='black', width=1)
                )
                for lat in stations['latitude']
            ]
        )

    return fig, sensor_vars

# ============================================================
# === Time–Salinity (T–S) Diagram Update ===
# ============================================================

@app.callback(
    Output('ts_plot', 'figure'),
    [
        Input('csv_selector', 'value'),
        Input('sub_sample', 'value'),
        Input('ts_color_variable', 'value'),
        Input('ts_color_map', 'value'),
        Input('ts_v_min', 'value'),
        Input('ts_v_max', 'value'),
        Input('hidden_opacity', 'value'),
        Input("cruise_track_selected_data", "modified_timestamp"),
    ],
    State('cruise_track_selected_data', 'data'),
)
def update_ts_plot(csv_file, sub_sample,
                   color_var, color_map, vmin, vmax,
                   hidden_opacity, cruise_track_flag, cruise_track_selection):
    """
    Update the T–S (Temperature–Salinity) diagram.

    Triggered by:
        - Dataset or variable changes
        - Subsampling or color scale updates
        - Cruise track selection changes

    Notes:
        - Cruise track selection is always applied FIRST
        - Shows σθ density contours (IES80)
        - Automatically rescales color range
    """

    # --------------------------------------------------------
    # 1️⃣ Load Data
    # --------------------------------------------------------
    if not csv_file:
        return go.Figure().add_annotation(
            text="⚠️ No CSV found", x=0.5, y=0.5, showarrow=False
        )

    df = load_data(csv_file, sub_sample=sub_sample)
    #df['point_id'] = df.index
    sensor_vars = [col for col in df.columns if '_std' not in col and col not in meta_vars]

    # --------------------------------------------------------
    # 2️⃣ Apply Cruise Track Selection
    # --------------------------------------------------------
    if cruise_track_selection and "selected_ids" in cruise_track_selection:
        selected_ids = cruise_track_selection["selected_ids"]
        df = df[df['point_id'].isin(selected_ids)].reset_index(drop=True)

    # --------------------------------------------------------
    # 3️⃣ Handle Empty Data
    # --------------------------------------------------------
    if df.empty:
        return go.Figure().add_annotation(
            text=f"⚠️ No {color_var} data available",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=16, color="red")
        )

    # --------------------------------------------------------
    # 4️⃣ Compute Density Contour Grid (σθ)
    # --------------------------------------------------------
    tmin, tmax = df['temperature'].quantile([0.01, 0.99]).round().astype(int)
    smin, smax = df['salinity'].quantile([0.01, 0.99]).round().astype(int)

    # Add buffer for contour coverage
    tmin, tmax = tmin - 2, tmax + 2
    smin, smax = smin - 2, smax + 2

    T, S = np.meshgrid(
        np.arange(tmin, tmax, 0.5),
        np.arange(smin, smax, 0.5),
        indexing='ij'
    )
    D = ies80(S, T, 0) - 1000  # potential density anomaly (σθ)

    # --------------------------------------------------------
    # 5️⃣ Configure Color Range
    # --------------------------------------------------------
    if vmin is None or vmax is None:
        q = df[color_var].quantile([0.05, 0.95])
        vmin = q[0.05] if vmin is None else vmin
        vmax = q[0.95] if vmax is None else vmax

    # --------------------------------------------------------
    # 6️⃣ Create T–S Scatter Plot
    # --------------------------------------------------------
    fig = px.scatter(
        df,
        x='salinity',
        y='temperature',
        color=color_var,
        color_continuous_scale=color_map,
        range_color=[vmin, vmax],
        hover_data={
            'temperature': ':.2f',
            'salinity': ':.2f',
            'depth': ':.2f',
            'latitude': ':.2f',
            'longitude': ':.2f',
            color_var: ':.2f'
        },
        custom_data=['point_id', 'media', 'frame', 'media_2', 'frame_2',
                     'times', 'latitude', 'longitude', 'link', 'link_2'] + sensor_vars
    )

    # Selected vs unselected opacity
    fig.update_traces(
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=hidden_opacity))
    )

    # --------------------------------------------------------
    # 7️⃣ Overlay σθ Contours
    # --------------------------------------------------------
    fig.add_trace(
        go.Contour(
            z=D,
            x=np.arange(smin, smax, 0.5),
            y=np.arange(tmin, tmax, 0.5),
            colorscale=[[0, 'black'], [1, 'black']],  # all contours in black
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

    # --------------------------------------------------------
    # 8️⃣ Final Layout & Styling
    # --------------------------------------------------------
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
                text=f'<br><br>&nbsp;{color_var.replace("_", " ").capitalize()} '
                     f'({sled_units.get(color_var, "")})',
                side='top'
            ),
            orientation='h',  # horizontal colorbar
            x=0.5, xanchor='center',  # centered horizontally
            y=1.075, len=0.75, thickness=15,
            ticks='outside',
            ticklabelposition="outside top"
        ),
    )

    return fig

# ============================================================
# === Vertical Profile Plot Update (Depth vs. Variable) ===
# ============================================================

@app.callback(
    Output('profile_plot', 'figure'),
    [
        Input('csv_selector', 'value'),
        Input('sub_sample', 'value'),
        Input('ts_color_variable', 'value'),
        Input('v_min', 'value'),
        Input('v_max', 'value'),
        Input('z_min', 'value'),
        Input('z_max', 'value'),
        Input('main_plot_selected_data', 'data'),
        Input("cruise_track_selected_data", "modified_timestamp"),
    ],
    State('cruise_track_selected_data', 'data'),
)
def update_profile_plot(csv_file, sub_sample,
                        color_var, vmin, vmax, zmin, zmax,
                        selected_data, cruise_track_flag, cruise_track_selection):
    """
    Update the vertical profile plot (Depth vs. Selected Variable).

    Behavior:
        - Cruise track selection is applied FIRST (primary filter)
        - Scatter selection refines the subset (secondary filter)
        - Displays median ± IQR (25–75%) for binned depth values
    """

    # --------------------------------------------------------
    # 1️⃣ Load Dataset
    # --------------------------------------------------------
    if not csv_file:
        return go.Figure().add_annotation(
            text="⚠️ No CSV found", x=0.5, y=0.5, showarrow=False
        ), []

    df = load_data(csv_file, sub_sample=sub_sample)
    #df['point_id'] = df.index

    # --------------------------------------------------------
    # 2️⃣ Apply Cruise Track Selection (Primary Filter)
    # --------------------------------------------------------
    if cruise_track_selection and "selected_ids" in cruise_track_selection:
        selected_ids = cruise_track_selection["selected_ids"]
        df = df[df['point_id'].isin(selected_ids)].reset_index(drop=True)

    # --------------------------------------------------------
    # 3️⃣ Apply Scatter Selection (Secondary Filter)
    # --------------------------------------------------------
    if selected_data and "selected_ids" in selected_data:
        df = df[df["point_id"].isin(selected_data["selected_ids"])].reset_index(drop=True)

    # --------------------------------------------------------
    # 4️⃣ Handle Empty Data
    # --------------------------------------------------------
    if df.empty:
        return go.Figure().add_annotation(
            text=f"⚠️ No {color_var} data available",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=16, color="red")
        )

    # --------------------------------------------------------
    # 5️⃣ Compute Depth-Binned Summary Statistics
    # --------------------------------------------------------
    step = 1  # 1 m depth bins
    df['depth'] = np.floor((df['depth'] + step / 2) / step) * step

    main_summary = df.groupby('depth')[color_var].agg(
        median=lambda x: x.median(),
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).reset_index()

    # Ensure proper order for plotting
    main_summary = main_summary.sort_values('depth', ascending=True)

    # --------------------------------------------------------
    # 6️⃣ Determine Color Scale Range (if not provided)
    # --------------------------------------------------------
    if vmin is None or vmax is None:
        q = df[color_var].quantile([0.05, 0.95])
        vmin = q[0.05] if vmin is None else vmin
        vmax = q[0.95] if vmax is None else vmax

    # --------------------------------------------------------
    # 7️⃣ Build Profile Plot 
    # --------------------------------------------------------
    fig = go.Figure()

    # Shaded interquartile range (IQR)
    fig.add_trace(
        go.Scatter(
            x=pd.concat([main_summary['q75'], main_summary['q25'][::-1]]),
            y=pd.concat([main_summary['depth'], main_summary['depth'][::-1]]),
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.25)',
            line=dict(width=0),
            hoverinfo='skip',
            showlegend=False,
            name='IQR'
        )
    )

    # Median line
    fig.add_trace(
        go.Scatter(
            x=main_summary['median'],
            y=main_summary['depth'],
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(symbol='circle', size=5, color='blue'),
            showlegend=False,
            hovertemplate=(
                "<b>Depth:</b> %{y:.1f} m<br>"
                f"<b>{color_var.capitalize()}:</b> %{{x:.2f}} {sled_units.get(color_var, '')}"
                "<extra></extra>"
            )
        )
    )

    # --------------------------------------------------------
    # 8️⃣ Layout and Axis Formatting
    # --------------------------------------------------------
    fig.update_layout(
        dragmode="zoom",
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
    )

    # Reverse y-axis (depth increases downward)
    y_range = [zmax, zmin - 10] if (zmin is not None and zmax is not None) else \
              [df['depth'].max(), df['depth'].min()]

    fig.update_yaxes(
        range=y_range,
        title='Depth (m)',
        showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)',
        showline=True, linecolor='black', linewidth=1,
        mirror=True, ticks='outside', tickwidth=1, tickcolor='black'
    )

    fig.update_xaxes(
        title=f'{color_var.replace("_", " ").capitalize()} ({sled_units.get(color_var, "")})',
        range=[vmin, vmax],
        showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)',
        showline=True, linecolor='black', linewidth=1,
        mirror=True, ticks='outside', tickwidth=1, tickcolor='black'
    )

    return fig

# ============================================================
# === Display Clicked Point Details (from Main Plot) ===
# ============================================================

@app.callback(
    Output('click-output', 'children'),
    Input('main_plot', 'clickData'),
    State('available-sensor-vars', 'data')
)
def display_click_data(clickData, sensor_vars):
    """
    Display detailed information about a clicked point in the main scatter plot.

    Triggered by:
        - User click on any point in the 'main_plot'

    Behavior:
        - Extracts custom_data values attached to the point
        - Displays all metadata (media links, position, depth, timestamp)
        - Dynamically lists all available sensor variables with formatted values
    """

    # --------------------------------------------------------
    # 1️⃣ Handle No Click Case
    # --------------------------------------------------------
    if not clickData or 'points' not in clickData or not clickData['points']:
        return 'Click on a point to see full details.'

    try:
        # --------------------------------------------------------
        # 2️⃣ Extract Custom Data
        # --------------------------------------------------------
        point = clickData['points'][0]
        custom_data = point.get('customdata', [])

        # Fixed indices: based on consistent custom_data structure in main_plot
        media        = custom_data[1] if len(custom_data) > 1 and pd.notna(custom_data[1]) else 'N/A'
        frame        = custom_data[2] if len(custom_data) > 2 and pd.notna(custom_data[2]) else 'N/A'
        media_2      = custom_data[3] if len(custom_data) > 3 and pd.notna(custom_data[3]) else 'N/A'
        frame_2      = custom_data[4] if len(custom_data) > 4 and pd.notna(custom_data[4]) else 'N/A'
        raw_time     = custom_data[5] if len(custom_data) > 5 and pd.notna(custom_data[5]) else None
        latitude     = custom_data[6] if len(custom_data) > 6 and pd.notna(custom_data[6]) else None
        longitude    = custom_data[7] if len(custom_data) > 7 and pd.notna(custom_data[7]) else None
        media_link   = custom_data[8] if len(custom_data) > 8 and pd.notna(custom_data[8]) else None
        media_link_2 = custom_data[9] if len(custom_data) > 9 and pd.notna(custom_data[9]) else None
        depth        = point.get('y', 'N/A')

        # --------------------------------------------------------
        # 3️⃣ Format Core Fields
        # --------------------------------------------------------
        formatted_time = (
            pd.to_datetime(raw_time).strftime('%Y-%m-%d %H:%M:%S')
            if raw_time else 'N/A'
        )

        # --------------------------------------------------------
        # 4️⃣ Build Dynamic Variable List
        # --------------------------------------------------------
        variable_details = []
        for i, var in enumerate(sensor_vars or [], start=10):  # start=10: first indices reserved for metadata
            value = custom_data[i] if i < len(custom_data) else None

            # Adaptive formatting
            if pd.isna(value):
                display_value = 'N/A'
            elif isinstance(value, (int, float)):
                if abs(value) >= 1e6:  # Very large → scientific notation
                    display_value = f'{value:.2e}'
                elif abs(value) < 1 and value != 0:  # Small floats → 4 decimals
                    display_value = f'{value:.4f}'
                else:  # Normal range → 2 decimals + comma separator
                    display_value = f'{value:,.2f}'
            else:
                display_value = str(value)

            variable_details.append(
                html.Div([
                    html.Span(
                        f'📈 {var.capitalize().replace("_", " ")} ({sled_units.get(var, "")}):',
                        style={'flex': '7', 'text-align': 'left'}
                    ),
                    html.Span(
                        display_value,
                        style={'flex': '3', 'text-align': 'right'}
                    )
                ],
                style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'})
            )

        # --------------------------------------------------------
        # 5️⃣ Return Full Info Panel
        # --------------------------------------------------------
        return html.Div([
            # ISIIS 1 info
            html.Div([
                html.Span('📽️ ISIIS 1:', style={'flex': '3', 'text-align': 'left'}),
                html.A(media, href=media_link, target='_blank', style={'flex': '7', 'text-align': 'right'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),

            # ISIIS 2 info
            html.Div([
                html.Span('📽️ ISIIS 2:', style={'flex': '3', 'text-align': 'left'}),
                html.A(media_2, href=media_link_2, target='_blank', style={'flex': '7', 'text-align': 'right'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),

            # Time
            html.Div([
                html.Span('⏳ Time:', style={'flex': '1', 'text-align': 'left'}),
                html.Span(formatted_time, style={'flex': '1', 'text-align': 'right'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),

            # Latitude
            html.Div([
                html.Span('🌍 Latitude:', style={'flex': '1', 'text-align': 'left'}),
                html.Span(f'{latitude:.2f}°' if latitude is not None else 'N/A',
                          style={'flex': '1', 'text-align': 'right'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),

            # Longitude
            html.Div([
                html.Span('🌍 Longitude:', style={'flex': '1', 'text-align': 'left'}),
                html.Span(f'{longitude:.2f}°' if longitude is not None else 'N/A',
                          style={'flex': '1', 'text-align': 'right'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),

            # Depth
            html.Div([
                html.Span('🌊 Depth:', style={'flex': '1', 'text-align': 'left'}),
                html.Span(f'{depth:.2f} m' if depth != 'N/A' else 'N/A',
                          style={'flex': '1', 'text-align': 'right'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),

            html.Hr(),

            # Dynamic sensor variables
            *variable_details
        ])

    # --------------------------------------------------------
    # 6️⃣ Handle Any Parsing Error
    # --------------------------------------------------------
    except Exception as e:
        return f'⚠️ Error processing click data: {str(e)}'

if __name__ == '__main__':
    # Command-line arguments
    args = cli()
    args.port = str(args.port)
    app.run(host=args.host, port=args.port, processes=MAX_WORKERS, threaded=False, debug=False)
else:
    # WSGI-compatible Flask server (eg gunicorn)
    application = app.server
