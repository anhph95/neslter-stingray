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
from plotly.colors import sample_colorscale

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
CSV_HEADER_CACHE = {}
TS_CONTOUR_CACHE = {}
SENSOR_VAR_CACHE = {}

# ============================================
# 🔹 Units dictionary
# ============================================
unit_patterns = {
    "°C": ["temperature","temp","t090","t190","t2","tv","_t"],
    "S m⁻¹": ["conductivity","cond","c0","c1","mS/cm"],
    "dbar": ["pressure","press","p_","pr","prd"],
    "m": ["depth","dep","z","altitude","alt"],
    "psu": ["salinity","sal","sal00","sal11","practical_salinity"],
    "kg m⁻³": ["density","dens","sigma"],
    "°": ["latitude","lat","longitude","lon","pitch","roll","heading"],
    "µM": ["nitrate","no3","suna","concentration"],
    "µg l⁻¹": ["chlorophyll","chl","fluor","fchl","fl"],
    "m⁻¹ sr⁻¹": ["backscattering","bb","bbp"],
    "µmol photons m⁻² s⁻¹": ["par","irradiance","ed"],
    "m s⁻¹": ["sound_velocity","sv","svcm"],
    "%": ["saturation","oxsat","o2sat"],
    "ind m⁻³": [
                "amphipod", "appendicularian", "chaetognath", "copepod", "ctenophore", "doliolid", "euphausids", "fish", "medusa",
                "polychaete", "pteropod", "radiolarian", "salp", "siphonophore", "trichodesmium","veliger"
               ],
}

def get_unit(varname):
    vn = varname.lower()
    # split variable name into tokens
    tokens = vn.replace("-", "_").split("_")
    for unit, pats in unit_patterns.items():
        for p in pats:
            if p in tokens:
                return unit
    return ""

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
def scan_datasets() -> list[str]:
    """
    Returns a list of available dataset folders under DATA_DIR.
    Example:
      /dash_data/data/NESLTER_2022/...
      returns ['NESLTER_2022']
    """
    if not DATA_DIR.exists():
        return []
    return sorted([f.name for f in DATA_DIR.iterdir() if f.is_dir()])

def get_csv_files(dataset: str) -> list[str]:
    """
    Returns CSV stem names inside the selected dataset folder.
    Example:
        dataset = 'NESLTER_2022'
        scans /dash_data/data/NESLTER_2022/*.csv
    """
    dataset_path = DATA_DIR / dataset
    if not dataset_path.exists():
        return []
    return sorted(f.stem for f in dataset_path.glob("*.csv") if f.is_file())

@lru_cache(maxsize=4)
def load_csv(path: Path) -> pd.DataFrame | None:
    """Read a small auxiliary CSV file (stations, bathymetry)."""
    return pd.read_csv(path, dtype=str, encoding="utf-8") if path.exists() else None

def dynamic_ticks(vmin, vmax, nticks=6):
    """Dynamic tick label with range"""
    span = abs(vmax - vmin)
    if span == 0:
        return np.array([vmin]), 2
    raw_step = span / (nticks - 1)
    magnitude = 10 ** np.floor(np.log10(raw_step))
    frac = raw_step / magnitude
    if frac < 1.5:
        step = 1 * magnitude
    elif frac < 3:
        step = 2 * magnitude
    elif frac < 7:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    # determine decimals
    if step >= 1:
        digits = 0
    else:
        digits = int(abs(np.floor(np.log10(step))))
    # compute nice bounds
    start = np.floor(vmin / step) * step
    end   = np.ceil(vmax / step) * step
    ticks = np.arange(start, end + step * 0.5, step)
    return ticks, digits

# ============================================
# 🔹 Main Data Loader with Smart Caching
# ============================================
def load_data(dataset: str, file_name: str, sub_sample: int = 1):
    """
    Load, clean, and optionally sub-sample a dataset.
    - Keeps essential cleaning (datetime, numeric, coordinate limits, media columns)
    - Adds a stable point_id if missing
    - No overcomplicated cache or timestamp tracking
    """
    dataset_path = DATA_DIR / dataset
    csv_path = dataset_path / f"{file_name}.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    base_key = f"{dataset}/{file_name}"
    if base_key not in DATA_CACHE:
        df = pd.read_csv(csv_path, low_memory=False)
        # Convert numeric columns to float32 to reduce memory + WebGL transfer
        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].astype(np.float32)
        # Convert integers to int32
        int_cols = df.select_dtypes(include=["int64"]).columns
        df[int_cols] = df[int_cols].astype(np.int32)
        if "point_id" not in df.columns:
            df["point_id"] = np.arange(len(df))
        df = df.set_index("point_id", drop=False)
        if "times" in df.columns:
            df["times"] = pd.to_datetime(df["times"], errors="coerce")
        DATA_CACHE[base_key] = df
    cache_key = f"{dataset}/{file_name}/{sub_sample}"
    if cache_key not in DATA_CACHE:
        DATA_CACHE[cache_key] = DATA_CACHE[base_key].iloc[::sub_sample]
    return DATA_CACHE[cache_key]

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

datasets = scan_datasets()
selected_dataset = datasets[-1] if datasets else None

csv_files = get_csv_files(selected_dataset) if selected_dataset else []
df = load_data(selected_dataset, csv_files[-1]) if csv_files else pd.DataFrame()

meta_vars = [
    "timestamp", "times", "matdate",
    "latitude", "longitude", "depth",
    "media", "media_path", "frame", "id", "link",
    "media_2", "media_path_2", "frame_2", "id_2", "link_2"
]

sensor_vars = [col for col in df.columns if "_std" not in col and col not in meta_vars] if not df.empty else []
colormaps = [name for name in px.colors.sequential.__dict__ if not name.startswith("_")]

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
    dcc.Interval(id="file-scan-interval", interval=600 * 1000, n_intervals=0),
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
            # Dataset selection
            html.Div([
                html.Label('Dataset:', style={'font-size': '20px'}),
                dcc.Dropdown(
                    id='dataset_selector',
                    options=[{'label': f, 'value': f} for f in datasets],
                    value=selected_dataset,
                    clearable=False
                ),
                html.Label('Data file:', style={'font-size': '20px'}),
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
                    html.Label('Y-Axis:'),
                    dcc.Dropdown(
                        id='y_axis_variable',
                        options=[{'label': var.capitalize(), 'value': var}
                                 for var in ['depth','latitude']],
                        value='depth'
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
                    html.Label('Marker size:'),
                    dcc.Input(id='size', type='number', value=5, debounce=True)
                ]),
                html.Div([
                    html.Label('Min Depth:'),
                    dcc.Input(id='z_min', type='number', value=0, debounce=True)
                ]),
                html.Div([
                    html.Label('Max Depth:'),
                    dcc.Input(id='z_max', type='number', value=200, debounce=True)
                ]),
                html.Div([
                    html.Label('Min Latitude:'),
                    dcc.Input(id='lat_min', type='number', value=None, debounce=True)
                ]),
                html.Div([
                    html.Label('Max Latitude:'),
                    dcc.Input(id='lat_max', type='number', value=None, debounce=True)
                ]),
                html.Div([
                    html.Label('Min Longitude:'),
                    dcc.Input(id='lon_min', type='number', value=None, debounce=True)
                ]),
                html.Div([
                    html.Label('Max Longitude:'),
                    dcc.Input(id='lon_max', type='number', value=None, debounce=True)
                ]),
                html.Div([
                    html.Label('Color Min:'),
                    dcc.Input(id='v_min', type='number', debounce=True)
                ]),
                html.Div([
                    html.Label('Color Max:'),
                    dcc.Input(id='v_max', type='number', debounce=True)
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
                    dcc.Input(id='ts_v_min', type='number', debounce=True)
                ]),
                html.Div([
                    html.Label('Color Max:'),
                    dcc.Input(id='ts_v_max', type='number', debounce=True)
                ])
            ], className='panel'),
            # Row 3.5: Profile controls
            html.Div([
                html.Label('PROFILE PLOT:', className='section-label'),
                html.Div([
                    html.Label('Variable:'),
                    dcc.Dropdown(
                        id='profile_variable',
                        options=[{'label': var.capitalize(), 'value': var}
                                for var in sensor_vars],
                        value='temperature'
                    )
                ]),
                html.Div([
                    html.Label('Colormap:'),
                    dcc.Dropdown(
                        id='profile_color_map',
                        options=[{'label': cmap, 'value': cmap} for cmap in colormaps],
                        value='Viridis'
                    )
                ]),
            ], className='panel'),
            # Row 4: Other options (moved to bottom)
            html.Div([
                html.Label('OTHER OPTIONS:', className='section-label'),
                html.Div([
                    html.Label('Sub-sampling:'),
                    dcc.Input(id='sub_sample', type='number', value=DEFAULT_SUBSAMPLE, debounce=True)
                ]),
                html.Div([
                    html.Label('Opacity:'),
                    dcc.Input(id='hidden_opacity', type='number', value=0.05, debounce=True)
                ]),
                html.Hr(),
                html.Label('PLOT LAYOUT:', className='section-label'),
                html.Div([
                    html.Label('Cruise Track Width (px):'),
                    dcc.Input(id='track_width', type='number', debounce=True)
                ]),
                html.Div([
                    html.Label('Cruise Track Height (px):'),
                    dcc.Input(id='track_height', type='number', debounce=True)
                ]),
                html.Div([
                    html.Label('Main Plot Width (px):'),
                    dcc.Input(id='main_width', type='number', debounce=True)
                ]),
                html.Div([
                    html.Label('Main Plot Height (px):'),
                    dcc.Input(id='main_height', type='number', debounce=True)
                ]),
                html.Div([
                    html.Label('TS Plot Width (px):'),
                    dcc.Input(id='ts_width', type='number', debounce=True)
                ]),
                html.Div([
                    html.Label('TS Plot Height (px):'),
                    dcc.Input(id='ts_height', type='number', debounce=True)
                ]),
                html.Div([
                    html.Label('Profile Plot Width (px):'),
                    dcc.Input(id='profile_width', type='number', debounce=True)
                ]),
                html.Div([
                    html.Label('Profile Plot Height (px):'),
                    dcc.Input(id='profile_height', type='number', debounce=True)
                ]),
                html.Div([
                    html.Label('Font Size:'),
                    dcc.Input(id='plot_font_size', type='number', value=10, debounce=True)
                ]),
            ], className='panel')
        ], className='left-panel'),
        # --- MIDDLE PLOTS ---
        html.Div([
            html.Div([dcc.Graph(id='cruise_track')], id='track_container', className='cruise-track-graph resizable'),
            dcc.Store(id='cruise_track_selected_data'),
            dcc.Store(id='cruise_track_selection_store', data={"selected_ids": []}),
            dcc.Store(id='main_plot_selected_data'),
            html.Div([dcc.Graph(id='main_plot')], id='main_container', className='main-graph resizable'),
            html.Div([dcc.Graph(id='ts_plot')], id='ts_container', className='ts-graph resizable'),
            html.Div([dcc.Graph(id='profile_plot')], id='profile_container', className='profile-graph resizable'),
            dcc.Store(id="plot_size_store", storage_type="local")
        ], className='middle-panel'),
        # --- RIGHT PANEL (Selected Data Info only) ---
        html.Div([
            html.Label('Details:', className='section-label', style={'font-size': '16px'}),
            html.Div(id='click-output', className='card', style={
                'font-size': '13px',
                'line-height': '1.4em'
            }),
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

# --- Callback: Refresh available dataset list ---
@app.callback(
    Output("dataset_selector", "options"),
    Input("file-scan-interval", "n_intervals"),
)
def refresh_dataset_list(_):
    ds = scan_datasets()
    return [{'label': f, 'value': f} for f in ds]

# --- Callback: Refresh available CSV file list ---
@app.callback(
    Output("csv_selector", "options"),
    Output("csv_selector", "value"),
    Input("dataset_selector", "value"),
    Input("refresh-button", "n_clicks"),
)
def update_csv_files(dataset, n_clicks):
    # Case 1: No dataset selected → nothing to load
    if not dataset:
        return [], None
    triggered = ctx.triggered_id  # dash context
    # Case 2: Refresh button pressed → clear cache
    if triggered == "refresh-button":
        global DATA_CACHE, CURRENT_FILE
        DATA_CACHE.clear()
        CURRENT_FILE = None
    # Case 3: Load CSV list for selected dataset
    csv_files = get_csv_files(dataset)
    options = [{'label': f, 'value': f} for f in csv_files]
    value = csv_files[-1] if csv_files else None
    return options, value

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
        Output('profile_variable', 'options'),
        Output('profile_variable', 'value')
    ],
    Input('dataset_selector', 'value'),
    Input('csv_selector', 'value'),
    State('color_variable', 'value'),
    State('ts_color_variable', 'value'),
    State('profile_variable', 'value'),
    prevent_initial_call=True
)
def update_color_variable_options(dataset, csv_file,
                                  current_color,
                                  current_ts_color,
                                  current_profile_var):
    if not csv_file:
        return [], None, [], None, [], None
    csv_path = DATA_DIR / dataset / f"{csv_file}.csv"
    if csv_path not in SENSOR_VAR_CACHE:
        if csv_path not in CSV_HEADER_CACHE:
            CSV_HEADER_CACHE[csv_path] = pd.read_csv(csv_path, nrows=0).columns
        cols = CSV_HEADER_CACHE[csv_path]
        SENSOR_VAR_CACHE[csv_path] = [
            c for c in cols
            if "_std" not in c and c not in meta_vars
        ]
    sensor_vars = SENSOR_VAR_CACHE[csv_path]
    options = [{'label': v.capitalize(), 'value': v} for v in sensor_vars]
    default = sensor_vars[0] if sensor_vars else None
    color_val = current_color if current_color in sensor_vars else default
    ts_val = current_ts_color if current_ts_color in sensor_vars else default
    profile_val = current_profile_var if current_profile_var in sensor_vars else default
    return options, color_val, options, ts_val, options, profile_val

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

app.clientside_callback(
"""
function(_, existing) {
    if (!window.plotResizeObserver) {
        window.plotResizeObserver = new ResizeObserver(() => {
            function getSize(id){
                const el = document.getElementById(id);
                if(!el) return null;
                const rect = el.getBoundingClientRect();
                return {
                    width: Math.round(rect.width),
                    height: Math.round(rect.height)
                };
            }
            const sizes = {
                track: getSize("track_container"),
                main: getSize("main_container"),
                ts: getSize("ts_container"),
                profile: getSize("profile_container")
            };
            window.dash_clientside.set_props("plot_size_store", {data: sizes});
        });
        ["track_container", "main_container", "ts_container", "profile_container"].forEach(id => {
            const el = document.getElementById(id);
            if (el) window.plotResizeObserver.observe(el);
        });
    }
    return existing;
}
""",
Output("plot_size_store", "data"),
Input("main_container", "id"),
State("plot_size_store", "data")
)

@app.callback(
    Output("track_width","value"),
    Output("track_height","value"),
    Output("main_width","value"),
    Output("main_height","value"),
    Output("ts_width","value"),
    Output("ts_height","value"),
    Output("profile_width","value"),
    Output("profile_height","value"),
    Input("plot_size_store","data"),
)
def update_size_inputs(data):
    if not data:
        raise dash.exceptions.PreventUpdate
    return (
        data["track"]["width"],
        data["track"]["height"],
        data["main"]["width"],
        data["main"]["height"],
        data["ts"]["width"],
        data["ts"]["height"],
        data["profile"]["width"],
        data["profile"]["height"],
    )

@app.callback(
    Output("track_container","style"),
    Output("main_container","style"),
    Output("ts_container","style"),
    Output("profile_container","style"),
    Input("track_width", "value"),
    Input("track_height", "value"),
    Input("main_width","value"),
    Input("main_height","value"),
    Input("ts_width","value"),
    Input("ts_height","value"),
    Input("profile_width","value"),
    Input("profile_height","value"),
    prevent_initial_call=True
)
def apply_manual_layout(trw, trh, mw, mh, tsw, tsh, pw, ph):
    def style(w, h):
        if w is None or h is None:
            return no_update
        return {
            "width": f"{int(w)}px",
            "height": f"{int(h)}px"
        }
    return (
        style(trw, trh),
        style(mw, mh),
        style(tsw, tsh),
        style(pw, ph)
    )
    
# ============================================================
# === Cruise Track Plot (Latitude vs. Time or Longitude) ===
# ============================================================
@app.callback(
    Output("cruise_track", "figure"),
    Input("dataset_selector", "value"),
    Input("csv_selector", "value"),
    Input("cruise_track_xaxis", "value"),
    Input("cruise_track_yaxis", "value"),
    Input("plot_font_size", "value"),
    State("sub_sample", "value"),
    prevent_initial_call=True,
)
def draw_cruise_track(dataset, csv_file, xaxis, yaxis, fontsize, sub_sample):
    trigger = ctx.triggered_id
    # 🧱 ignore redraws unless the trigger is one of these:
    if trigger not in ("csv_selector", "cruise_track_xaxis", "cruise_track_yaxis", "plot_font_size"):
        raise dash.exceptions.PreventUpdate
    if not csv_file:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ No CSV found",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False
        )
        return fig
    df = load_data(dataset, csv_file, sub_sample=sub_sample)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df[xaxis],
        y=df[yaxis],
        mode="markers",
        marker=dict(size=5, color="blue"),
        meta=df["point_id"].astype(int).tolist(),  # <-- meta always survives
        customdata=df["point_id"].astype(int).to_numpy().reshape(-1,1)
    ))
    fig.update_traces(
        mode="markers",
        selected=dict(marker=dict(color="red")),
        unselected=dict(marker=dict(color="blue"))
    )
    fig.update_layout(
        dragmode="select",
        selectdirection="any",
        # newselection_mode="immediate",
        clickmode="select",
        uirevision="cruise-track",
        font=dict(size=fontsize if fontsize else 10),
        xaxis=dict(
            title=xaxis.capitalize(),
            rangeslider=dict(visible=False),
            showgrid=True, gridcolor="rgba(0,0,0,0.1)",
            showline=True, linecolor="black", mirror=True
        ),
        yaxis=dict(
            title=yaxis.capitalize(),
            autorange=True,
            fixedrange=False,
            showgrid=True, gridcolor="rgba(0,0,0,0.1)",
            showline=True, linecolor="black", mirror=True
        ),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    if xaxis in ["longitude", "latitude"]:
        fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig

# ============================================================
# === Selections & Range Change Tracking ===
# ============================================================
@app.callback(
    Output("cruise_track_selection_store", "data"),
    Input("cruise_track", "selectedData"),
    Input("dataset_selector", "value"),
    Input("csv_selector", "value"),
    State("cruise_track_selection_store", "data"),
    prevent_initial_call=True,
)
def persist_cruise_track_selection(selectedData, dataset, csv_file, prev_data):
    if not csv_file:
        raise dash.exceptions.PreventUpdate
    df = load_data(dataset, csv_file)
    trigger = ctx.triggered_id
    # Reset when dataset changes
    if trigger in ("csv_selector", "dataset_selector"):
        return {"selected_ids": df["point_id"].astype(int).tolist()}
    # User selection
    if trigger == "cruise_track" and selectedData and selectedData.get("points"):
        selected_ids = [
            int(p["meta"])
            for p in selectedData["points"]
            if p.get("meta") is not None
        ]
        if not selected_ids:
            return {"selected_ids": df["point_id"].astype(int).tolist()}
        return {"selected_ids": selected_ids}
    # Double-click reset
    if trigger == "cruise_track" and selectedData is None:
        return {"selected_ids": df["point_id"].astype(int).tolist()}
    raise dash.exceptions.PreventUpdate

# --- Callback: Mirror selections between main scatter and TS plots ---
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
    trigger = ctx.triggered_id
    patched_scatter = Patch()
    patched_ts = Patch()
    # --- Detect clearing ---
    if trigger == "main_plot" and not scatter_sel:
        for i, _ in enumerate(fig_ts.get("data", [])):
            patched_ts["data"][i]["selectedpoints"] = None
        for i, _ in enumerate(fig_scatter.get("data", [])):
            patched_scatter["data"][i]["selectedpoints"] = None
        return patched_scatter, patched_ts
    if trigger == "ts_plot" and not ts_sel:
        for i, _ in enumerate(fig_scatter.get("data", [])):
            patched_scatter["data"][i]["selectedpoints"] = None
        for i, _ in enumerate(fig_ts.get("data", [])):
            patched_ts["data"][i]["selectedpoints"] = None
        return patched_scatter, patched_ts
    # --- Collect selected IDs ---
    selected_ids = set()
    if trigger == "main_plot" and scatter_sel and scatter_sel.get("points"):
        selected_ids = {
            int(p["customdata"])
            for p in scatter_sel["points"]
            if p.get("customdata") is not None
        }
    elif trigger == "ts_plot" and ts_sel and ts_sel.get("points"):
        selected_ids = {
            int(p["customdata"])
            for p in ts_sel["points"]
            if p.get("customdata") is not None
        }
    # --- Apply selection to both figures ---
    for fig, patch in [(fig_scatter, patched_scatter), (fig_ts, patched_ts)]:
        for i, trace in enumerate(fig.get("data", [])):
            if "customdata" not in trace:
                continue
            custom_ids = trace["customdata"]
            selected_idx = [
                j for j, pid in enumerate(custom_ids)
                if pid in selected_ids
            ]
            patch["data"][i]["selectedpoints"] = selected_idx or None
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
    selected_ids = [
        int(p["customdata"])
        for p in selectedData["points"]
        if p.get("customdata") is not None
    ]
    return {"selected_ids": selected_ids}

# --- Callback: Track user coordinate/range edits ---
@app.callback(
    Output('user_range_change', 'data'),
    Input('lat_min', 'value'),
    Input('lat_max', 'value'),
    Input('z_min', 'value'),
    Input('z_max', 'value'),
    State('lat_min', 'value'),
    State('lat_max', 'value'),
    State('z_min', 'value'),
    State('z_max', 'value'),
    State('user_range_change', 'data'),
    prevent_initial_call=True
)
def track_range_change(lat_min, lat_max, ymin, ymax,
                       prev_lat_min, prev_lat_max, prev_ymin, prev_ymax,
                       was_changed):
    """Track if the user manually changed coordinate or depth limits."""
    changed = any([
        lat_min != prev_lat_min,
        lat_max != prev_lat_max,
        ymin != prev_ymin,
        ymax != prev_ymax
    ])
    return True if changed else was_changed

# # --- Callback: Download CSV file ---
# @app.callback(
#     Output('download_dataframe_csv', 'data'),
#     Input('download-button', 'n_clicks'),
#     State('dataset_selector', 'value'),
#     State('csv_selector', 'value'),
#     prevent_initial_call=True
# )
# def download_csv(n_clicks, dataset, csv_file):
#     '''Download the selected CSV file.'''
#     if ctx.triggered_id == 'download-button' and csv_file:
#         df = load_data(dataset,csv_file)
#         return dcc.send_data_frame(df.to_csv, filename=f'{csv_file}.csv', index=False)
#     return no_update

# ============================================================
# === Main Plot Update: Depth vs. Variable / Coordinate ===
# ============================================================
@app.callback(
    Output('main_plot', 'figure'),
    [
        Input('dataset_selector', 'value'),
        Input('csv_selector', 'value'),
        Input('sub_sample', 'value'),
        Input('x_axis_variable', 'value'),
        Input('y_axis_variable', 'value'),
        Input('color_variable', 'value'),
        Input('color_map', 'value'),
        Input('size', 'value'),
        Input('v_min', 'value'),
        Input('v_max', 'value'),
        Input('z_min', 'value'),
        Input('z_max', 'value'),
        Input('lat_min', 'value'),
        Input('lat_max', 'value'),
        Input('lon_min', 'value'),
        Input('lon_max', 'value'),
        Input('hidden_opacity', 'value'),
        Input('plot_font_size', 'value'),
        Input('bathymetry', 'value'),
        Input('station', 'value'),
        Input('user_range_change', 'data'),
        Input('cruise_track_selection_store', 'data'),
        Input('main_plot', 'relayoutData'),
    ]
)
def update_main_plot(dataset, csv_file, sub_sample,
                     x_axis, y_axis, color_var, color_map, size, vmin, vmax,
                     zmin, zmax, lat_min, lat_max, lon_min, lon_max,
                     hidden_opacity, fontsize, bathymetry, station,
                     user_changed_range, cruise_track_selection, relayoutData):
    # --------------------------------------------------------
    # 1️⃣ Load Data
    # --------------------------------------------------------
    trigger = ctx.triggered_id
    if trigger == "main_plot" and relayoutData:
        valid_relayout = any(
            k.startswith("xaxis.range") or
            k.startswith("yaxis.range") or
            k.endswith(".autorange")
            for k in relayoutData.keys()
        )
        if not valid_relayout:
            raise dash.exceptions.PreventUpdate
    if not csv_file:
        return (
            go.Figure().add_annotation(
                text="⚠️ No CSV found", x=0.5, y=0.5, showarrow=False
            ),
            []
        )
    df = load_data(dataset, csv_file, sub_sample=sub_sample)
    # --------------------------------------------------------
    # 2️⃣ Apply Cruise Track Selection
    # --------------------------------------------------------
    if cruise_track_selection and "selected_ids" in cruise_track_selection:
        df = df[df["point_id"].isin(cruise_track_selection["selected_ids"])]
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
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df[x_axis],
        y=df[y_axis],
        mode="markers",
        marker=dict(
            size=size,
            color=df[color_var],
            colorscale=color_map,
            cmin=vmin,
            cmax=vmax,
            coloraxis="coloraxis"
        ),
        customdata=df["point_id"].astype(np.int32).to_numpy(),
        hovertemplate=(
            "Depth: %{y:.2f}<br>"
            "Lat: %{x:.2f}<br>"
            f"{color_var}: %{{marker.color:.2f}}<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.update_traces(
        marker=dict(size=size),
        selected=dict(marker=dict(opacity=1)),
        unselected=dict(marker=dict(opacity=hidden_opacity))
    )
    # --------------------------------------------------------
    # 5️⃣ Axis Ranges & Tick Formatting (Clean Version)
    # --------------------------------------------------------
    def get_visible_range(axis_name, relayoutData):
        if relayoutData:
            r0 = f"{axis_name}.range[0]"
            r1 = f"{axis_name}.range[1]"
            if r0 in relayoutData:
                return [relayoutData[r0], relayoutData[r1]]
        return None
    visible_xrange = get_visible_range("xaxis", relayoutData)
    visible_yrange = get_visible_range("yaxis", relayoutData)
    def resolve_range(axis_name, visible_range, user_min, user_max, data_series):
        if visible_range is not None:
            return min(visible_range), max(visible_range)
        if (user_min is not None) and (user_max is not None):
            return user_min, user_max
        return data_series.min(), data_series.max()
    # =========================
    # X AXIS
    # =========================
    xticks = xticktext = None
    if x_axis in df.columns:
        xmin, xmax = resolve_range(
            x_axis,
            visible_xrange,
            lat_min if x_axis == "latitude" else lon_min,
            lat_max if x_axis == "latitude" else lon_max,
            df[x_axis]
        )
        xticks, digits = dynamic_ticks(xmin, xmax, nticks=6)
        if x_axis == "latitude":
            xticktext = [
                f"{abs(v):.{digits}f}°{'N' if v >= 0 else 'S'}"
                for v in xticks
            ]
            x_range = [xmin, xmax]
        elif x_axis == "longitude":
            xticktext = [
                f"{abs(v):.{digits}f}°{'E' if v >= 0 else 'W'}"
                for v in xticks
            ]
            x_range = [xmin, xmax]
        else:
            x_range = [xmin, xmax]
    elif x_axis == "times":
        x_range = [df["times"].min(), df["times"].max()]
        xticks = xticktext = None
    else:
        x_range = xticks = xticktext = None
    # =========================
    # Y AXIS
    # =========================
    yticks = yticktext = None
    ylabel = y_axis.capitalize()
    if y_axis in df.columns:
        ymin, ymax = resolve_range(
            y_axis,
            visible_yrange,
            zmin if y_axis == "depth" else lon_min,
            zmax if y_axis == "depth" else lon_max,
            df[y_axis]
        )
        if y_axis == "depth":
            ylabel = "Depth (m)"
            y_range = [ymax, ymin]  # reverse depth
        else:
            y_range = [ymin, ymax]
        if y_axis in ["latitude", "longitude"]:
            yticks, digits = dynamic_ticks(ymin, ymax, nticks=6)
            if y_axis == "latitude":
                yticktext = [
                    f"{abs(v):.{digits}f}°{'N' if v >= 0 else 'S'}"
                    for v in yticks
                ]
            else:
                yticktext = [
                    f"{abs(v):.{digits}f}°{'E' if v >= 0 else 'W'}"
                    for v in yticks
                ]
    else:
        y_range = None
    # --------------------------------------------------------
    # 6️⃣ Global Dynamic Font Size (applies to everything)
    # --------------------------------------------------------
    if fontsize:
        base_font = fontsize
    else:
        if x_axis == "times" and x_range is not None:
            # Time axis: use number of days as proxy for span
            try:
                span_days = (pd.to_datetime(x_range[1]) - pd.to_datetime(x_range[0])).days
                span_days = max(span_days, 1)
                base_font = max(7, min(14, 10 - 0.5 * np.log10(span_days)))
            except Exception:
                base_font = 10
        else:
            # Numeric axis
            if x_range is not None and np.all(np.isfinite(x_range)):
                span = abs(x_range[1] - x_range[0])
            else:
                span = 1
            base_font = max(7, min(14, 10 - np.log10(span + 1e-6)))
    # --------------------------------------------------------
    # 7️⃣ Layout (with dynamic font everywhere)
    # --------------------------------------------------------
    fig.update_layout(
        dragmode="zoom",
        # uirevision=None if user_changed_range else "keep",
        uirevision=f"{dataset}-{csv_file}",   # stable per file
        # global font
        font=dict(size=base_font, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            title=x_axis.capitalize(),
            range=x_range,
            autorange="reversed" if x_axis == "latitude" else None,
            tickvals=xticks,
            ticktext=xticktext,
            tickmode='array',
            # titlefont=dict(size=base_font + 2),
            tickfont=dict(size=base_font),
            showgrid=True, gridcolor='rgba(0,0,0,0.1)',
            showline=True, linecolor="black", mirror=True,
            ticks="outside", tickwidth=1, tickcolor="black",
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            title=ylabel,
            range=y_range,
            tickvals=yticks,
            ticktext=yticktext,
            tickmode='array',
            # titlefont=dict(size=base_font + 2),
            tickfont=dict(size=base_font),
            showgrid=True, gridcolor='rgba(0,0,0,0.1)',
            showline=True, linecolor="black", mirror=True,
            ticks="outside", tickwidth=1, tickcolor="black",
        ),
        coloraxis=dict(
                colorbar=dict(
                    title=dict(
                        text=f"{color_var.replace('_', ' ').capitalize()} ({get_unit(color_var)})",
                        side="bottom",
                        font=dict(size=base_font + 1),
                    ),
                    tickfont=dict(size=base_font * 0.9),
                    orientation="h",
                    x=0.5, xanchor="center",
                    y=0, yanchor="top",
                    ypad=80,
                    lenmode="fraction",
                    len=0.75,
                    thickness=15,
                    ticks="outside",
                    ticklabelposition="outside bottom",
                    tickmode='auto',
                    nticks=5,
                ),
                colorscale=color_map,
                cmin=vmin,
                cmax=vmax,
            )
        )
    # --------------------------------------------------------
    # 8️⃣ Bathymetry Overlay
    # --------------------------------------------------------
    if 'True' in bathymetry and bathy is not None and x_axis == 'latitude' and y_axis == 'depth':
        bathy_mask = (
            (bathy['latitude'] <= df['latitude'].max() + 0.1) &
            (bathy['latitude'] >= df['latitude'].min() - 0.1)
        )
        fig.add_trace(
            go.Scatter(
                x=bathy['latitude'][bathy_mask],
                y=bathy['bottom_depth_meters'][bathy_mask],
                mode="lines",
                line=dict(color="black", width=1),
                name="Bathymetry",
                showlegend=False,
            )
        )
    # --------------------------------------------------------
    # 9️⃣ Station Overlay (filtered exactly like bathymetry)
    # --------------------------------------------------------
    if 'True' in station and stations is not None and x_axis == 'latitude' and y_axis == 'depth':
        station_mask = (
            (stations['latitude'] <= df['latitude'].max() + 0.1) &
            (stations['latitude'] >= df['latitude'].min() - 0.1)
        )
        visible_stations = stations[station_mask]
        station_labels = [
            dict(
                x=lat,
                y=1,
                xref="x",
                yref="paper",
                text=label,
                showarrow=False,
                font=dict(size=base_font),
                align="center",
                yshift=35
            )
            for lat, label in zip(
                visible_stations['latitude'],
                visible_stations['station']
            )
        ]
        station_lines = [
            dict(
                type="line",
                x0=lat, x1=lat,
                y0=1, y1=1.02,
                xref="x",
                yref="paper",
                line=dict(color="black", width=1),
            )
            for lat in visible_stations['latitude']
        ]
        fig.update_layout(
            annotations=station_labels,
            shapes=station_lines
        )
    return fig

# ============================================================
# === Time–Salinity (T–S) Diagram Update ===
# ============================================================
@app.callback(
    Output('ts_plot', 'figure'),
    [
        Input('dataset_selector', 'value'),
        Input('csv_selector', 'value'),
        Input('sub_sample', 'value'),
        Input('ts_color_variable', 'value'),
        Input('ts_color_map', 'value'),
        Input('ts_v_min', 'value'),
        Input('ts_v_max', 'value'),
        Input('hidden_opacity', 'value'),
        Input('plot_font_size', 'value'),
        Input('cruise_track_selection_store', 'data'),
    ]
)
def update_ts_plot(dataset, csv_file, sub_sample,
                   color_var, color_map, vmin, vmax,
                   hidden_opacity, fontsize, cruise_track_selection):
    # --------------------------------------------------------
    # 1️⃣ Load Data
    # --------------------------------------------------------
    if not csv_file:
        return go.Figure().add_annotation(
            text="⚠️ No CSV found", x=0.5, y=0.5, showarrow=False
        )
    df = load_data(dataset, csv_file, sub_sample=sub_sample)
    if 'temperature' not in df.columns or 'salinity' not in df.columns:
        return go.Figure().add_annotation(
            text="⚠️ Temperature or Salinity data missing",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=16, color="red")
        )
    if color_var not in df.columns:
        return go.Figure().add_annotation(
            text=f"No {color_var} data available",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=16, color="red")
        )
    # --------------------------------------------------------
    # 2️⃣ Apply Cruise Track Selection
    # --------------------------------------------------------
    if cruise_track_selection and "selected_ids" in cruise_track_selection:
        df = df[df["point_id"].isin(cruise_track_selection["selected_ids"])]
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
    # 4️⃣ Compute Density Contours (σθ)
    # --------------------------------------------------------
    tmin, tmax = df['temperature'].quantile([0.01, 0.99]).round().astype(int)
    smin, smax = df['salinity'].quantile([0.01, 0.99]).round().astype(int)
    tmin -= 2
    tmax += 2
    smin -= 2
    smax += 2
    T, S = np.meshgrid(
        np.arange(tmin, tmax, 0.5),
        np.arange(smin, smax, 0.5),
        indexing='ij'
    )
    D = ies80(S, T, 0) - 1000
    # --------------------------------------------------------
    # 5️⃣ Configure Color Range
    # --------------------------------------------------------
    if vmin is None or vmax is None:
        q = df[color_var].quantile([0.05, 0.95])
        vmin = q[0.05] if vmin is None else vmin
        vmax = q[0.95] if vmax is None else vmax
    # --------------------------------------------------------
    # 6️⃣ Create T-S Scatter Plot
    # --------------------------------------------------------
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=df['salinity'],
            y=df['temperature'],
            mode="markers",
            marker=dict(
                size=5,
                color=df[color_var],
                colorscale=color_map,
                cmin=vmin,
                cmax=vmax,
                coloraxis="coloraxis"
            ),
            # Only send point_id
            customdata=df["point_id"].astype(np.int32).to_numpy(),
            selected=dict(marker=dict(opacity=1)),
            unselected=dict(marker=dict(opacity=hidden_opacity)),
            hovertemplate=(
                "Salinity: %{x:.2f}<br>"
                "Temperature: %{y:.2f} °C<br>"
                f"{color_var}: %{{marker.color:.2f}}<extra></extra>"
            ),
        )
    )
    # --------------------------------------------------------
    # 7️⃣ Dynamic Font Size
    # --------------------------------------------------------
    if fontsize:
        base_font = fontsize
    else:
        span = max(abs(tmax - tmin), abs(smax - smin))
        base_font = max(7, min(14, 9 * (span / 1.0)))
    # --------------------------------------------------------
    # 8️⃣ Add Density Contours
    # --------------------------------------------------------
    fig.add_trace(
        go.Contour(
            z=D,
            x=np.arange(smin, smax, 0.5),
            y=np.arange(tmin, tmax, 0.5),
            colorscale=[[0, 'black'], [1, 'black']],
            contours=dict(
                coloring='lines',
                showlabels=True,
                labelfont=dict(size=base_font - 1, color='black')
            ),
            line=dict(color='black', width=1),
            hoverinfo='skip',
            showscale=False,
            name="σθ"
        )
    )
    # --------------------------------------------------------
    # 9️⃣ Layout
    # --------------------------------------------------------
    fig.update_layout(
        dragmode="zoom",
        uirevision='keep',
        font=dict(size=base_font, color='black'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(
            title='Salinity (psu)',
            range=[smin, smax],
            tickfont=dict(size=base_font),
            showgrid=True,
            gridcolor='rgba(0, 0, 0, 0.1)',
            showline=True,
            linecolor='black',
            mirror=True,
            ticks='outside'
        ),
        yaxis=dict(
            title='Temperature (°C)',
            range=[tmin, tmax],
            tickfont=dict(size=base_font),
            showgrid=True,
            gridcolor='rgba(0, 0, 0, 0.1)',
            showline=True,
            linecolor='black',
            mirror=True,
            ticks='outside'
        ),
        coloraxis=dict(
            colorscale=color_map,
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(
                title=dict(
                    text=f'{color_var.replace("_"," ").capitalize()} ({get_unit(color_var)})',
                    side='bottom',
                    font=dict(size=base_font + 1),
                ),
                tickfont=dict(size=base_font * 0.9),
                orientation='h',
                x=0.5,
                xanchor='center',
                y=0,
                yanchor='top',
                ypad=70,
                lenmode='fraction',
                len=0.75,
                thickness=15,
                ticks='outside',
                ticklabelposition="outside bottom",
                tickmode='auto',
                nticks=5,
            )
        )
    )
    return fig

# ============================================================
# === Vertical Profile Plot Update (Depth vs. Variable) ===
# ============================================================
@app.callback(
    Output('profile_plot', 'figure'),
    [
        Input('dataset_selector', 'value'),
        Input('csv_selector', 'value'),
        Input('sub_sample', 'value'),
        Input('profile_variable', 'value'),   
        Input('profile_color_map', 'value'),
        Input('plot_font_size', 'value'),   
        Input('main_plot_selected_data', 'data'),
        Input('cruise_track_selection_store', 'data'),
    ]
)
def update_profile_plot(dataset, csv_file, sub_sample,
                        color_var, color_map, fontsize, selected_data, cruise_track_selection):
    """
    Update the vertical profile plot (Depth vs. Selected Variable).
    Behavior:
        - Cruise track selection is applied FIRST (primary filter)
        - Scatter selection expands to FULL profiles (secondary filter)
        - If 'cast' exists: plot raw casts (each cast one color)
        - If 'cast' does NOT exist: fallback to median profile
    """
    fig = go.Figure()
    # --------------------------------------------------------
    # 1️⃣ Load Dataset
    # --------------------------------------------------------
    if not csv_file:
        fig.add_annotation(text="⚠️ No CSV found", x=0.5, y=0.5, showarrow=False)
        return fig
    df = load_data(dataset, csv_file, sub_sample=sub_sample)
    if not color_var or color_var not in df.columns:
        fig.add_annotation(
            text="⚠️ No variable selected",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=16, color="red")
        )
        return fig
    # --------------------------------------------------------
    # 2️⃣ Apply Cruise Track Selection (Primary Filter)
    # --------------------------------------------------------
    if cruise_track_selection and "selected_ids" in cruise_track_selection:
        df = df[df["point_id"].isin(cruise_track_selection["selected_ids"])]
    # --------------------------------------------------------
    # 3️⃣ Expand scatter selection → full profiles OR subset
    # --------------------------------------------------------
    if selected_data and "selected_ids" in selected_data:
        selected_ids = set(selected_data["selected_ids"])
        if 'cast' in df.columns:
            selected_profiles = (
                df.loc[df["point_id"].isin(selected_ids), "cast"]
                .dropna()
                .unique()
            )
            if len(selected_profiles):
                df = df[df["cast"].isin(selected_profiles)]
        else:
            # 🔥 No cast → subset directly by selected points
            df = df[df["point_id"].isin(selected_ids)]
    # --------------------------------------------------------
    # 4️⃣ Clean bad values
    # --------------------------------------------------------
    df = df.loc[
        np.isfinite(df.get('depth', np.nan)) &
        np.isfinite(df.get(color_var, np.nan)) &
        np.isfinite(df.get('latitude', np.nan)) &
        np.isfinite(df.get('longitude', np.nan))
    ]
    if df.empty:
        fig.add_annotation(
            text=f"⚠️ No {color_var} data available",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=16, color="red")
        )
        return fig
    # --------------------------------------------------------
    # 5️⃣ Build Profile Plot
    # --------------------------------------------------------
    def get_palette(name):
        if hasattr(px.colors.sequential, name):
            return getattr(px.colors.sequential, name), "sequential"
        if hasattr(px.colors.qualitative, name):
            return getattr(px.colors.qualitative, name), "discrete"
        return px.colors.sequential.Viridis, "sequential"
    if 'cast' in df.columns and df['cast'].notna().any():
        df = df.copy()
        df['cast'] = pd.to_numeric(df['cast'], errors='coerce').astype('Int64')
        unique_profiles = sorted(df['cast'].dropna().unique())
        palette, palette_type = get_palette(color_map)
        n = len(unique_profiles)
        if palette_type == "sequential":
            colors = sample_colorscale(palette, [i / max(n - 1, 1) for i in range(n)])
        else:
            colors = [palette[i % len(palette)] for i in range(n)]
        color_map = {p: colors[i] for i, p in enumerate(unique_profiles)}
        for prof, g in df.groupby('cast'):
            g = g.sort_values('depth')
            color = color_map.get(prof, 'rgba(0,0,0,0.6)')
            fig.add_trace(
                go.Scatter(
                    x=g[color_var],
                    y=g['depth'],
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=4, color=color),
                    name=f'Cast {int(prof)}',
                    customdata=np.c_[g['latitude'], g['longitude']],
                    hovertemplate=(
                        f"<b>Cast:</b> {int(prof)}<br>"
                        "<b>Depth:</b> %{y:.1f} m<br>"
                        f"<b>{color_var.replace('_',' ').capitalize()}:</b> %{{x:.2f}} {get_unit(color_var)}<br>"
                        "<b>Latitude:</b> %{customdata[0]:.4f}<br>"
                        "<b>Longitude:</b> %{customdata[1]:.4f}<br>"
                        "<extra></extra>"
                    )
                )
            )
    else:
        # --------------------------------------------------------
        # 🔹 Bin profiles to 2 m depth bins
        # --------------------------------------------------------
        step = 2
        depth_bin = np.floor((df["depth"] + step / 2) / step) * step
        summary = (
            df.assign(_depth_bin=depth_bin)
            .groupby("_depth_bin")
            .agg(
                depth=("depth", "median"),
                median=(color_var, "median"),
                q05=(color_var, lambda x: np.nanpercentile(x, 5)),
                q95=(color_var, lambda x: np.nanpercentile(x, 95)),
                latitude=("latitude", "median"),
                longitude=("longitude", "median"),
            )
            .reset_index(drop=True)
            .sort_values("depth")
        )
        # --------------------------------------------------------
        # 🔹 Percentile envelope (5–95%)
        # --------------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([summary["q05"], summary["q95"][::-1]]),
                y=np.concatenate([summary["depth"], summary["depth"][::-1]]),
                fill="toself",
                fillcolor="rgba(0,100,200,0.2)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # --------------------------------------------------------
        # 🔹 Median profile line
        # --------------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=summary["median"],
                y=summary["depth"],
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(size=5, color="blue"),
                showlegend=False,
                customdata=summary[["latitude", "longitude"]].to_numpy(),
                hovertemplate=(
                    "<b>Depth:</b> %{y:.1f} m<br>"
                    f"<b>{color_var.replace('_',' ').capitalize()}:</b> %{{x:.2f}} {get_unit(color_var)}<br>"
                    "<b>Latitude:</b> %{customdata[0]:.4f}<br>"
                    "<b>Longitude:</b> %{customdata[1]:.4f}<br>"
                    "<extra></extra>"
                ),
            )
        )
    # --------------------------------------------------------
    # 6️⃣ Layout & Axes (auto reverse depth)
    # --------------------------------------------------------
    fig.update_layout(
        dragmode="zoom",
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black',size=fontsize if fontsize else 10),
        legend=dict(title="Cast")
    )
    fig.update_yaxes(
        autorange="reversed",   # depth increases downward
        title='Depth (m)',
        showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)',
        showline=True, linecolor='black', linewidth=1,
        mirror=True, ticks='outside', tickwidth=1, tickcolor='black'
    )
    fig.update_xaxes(
        title=f'{color_var.replace("_", " ").capitalize()} ({get_unit(color_var)})',
        showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)',
        showline=True, linecolor='black', linewidth=1,
        mirror=True, ticks='outside', tickwidth=1, tickcolor='black'
    )
    return fig

# ============================================================s
# === Display Clicked Point Details (from Main Plot) ===
# ============================================================
@app.callback(
    Output('click-output', 'children'),
    Input('main_plot', 'clickData'),
    State('dataset_selector', 'value'),
    State('csv_selector', 'value'),
)
def display_click_data(clickData, dataset, csv_file):
    """
    Display detailed information about a clicked point in the main scatter plot.
    Uses point_id to fetch the full row from the server instead of sending
    all variables through Plotly customdata.
    """
    if not clickData or 'points' not in clickData or not clickData['points']:
        return 'Click on a point to see full details.'
    try:
        # Get clicked point_id
        point_id = clickData["points"][0]["customdata"]
        # Load dataframe
        df = load_data(dataset, csv_file)
        # Retrieve full row
        if point_id not in df.index:
            return "Point no longer in filtered dataset."
        row = df.loc[point_id]
        def clean(v, default=None):
            if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
                return default
            return v
        # Metadata
        media        = clean(row.get('media'), 'N/A')
        frame        = clean(row.get('frame'), 'N/A')
        media_2      = clean(row.get('media_2'), 'N/A')
        frame_2      = clean(row.get('frame_2'), 'N/A')
        raw_time     = clean(row.get('times'))
        latitude     = clean(row.get('latitude'))
        longitude    = clean(row.get('longitude'))
        media_link   = clean(row.get('link'))
        media_link_2 = clean(row.get('link_2'))
        depth        = clean(row.get('depth'))
        formatted_time = (
            pd.to_datetime(raw_time).strftime('%Y-%m-%d %H:%M:%S')
            if raw_time not in [None, "", pd.NaT] else 'N/A'
        )
        lat_display = (
            f'{latitude:.2f}°'
            if isinstance(latitude, (int, float)) and pd.notna(latitude)
            else 'N/A'
        )
        lon_display = (
            f'{longitude:.2f}°'
            if isinstance(longitude, (int, float)) and pd.notna(longitude)
            else 'N/A'
        )
        depth_display = (
            f'{depth:.2f} m'
            if isinstance(depth, (int, float)) and pd.notna(depth)
            else 'N/A'
        )
        # Metadata fields to exclude
        META_COLS = {
            'point_id',
            'media','frame','media_path','id','link',
            'media_2','frame_2','media_path_2','id_2','link_2',
            'times','latitude','longitude','depth','timestamp','matdate'
        }
        # Build sensor variable display
        variable_details = []
        for var in row.index:
            if var in META_COLS:
                continue
            if var.endswith("_std"):
                continue
            value = row.get(var)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                display_value = 'N/A'
            elif isinstance(value, (int, float)):
                if abs(value) >= 1e6:
                    display_value = f'{value:.2e}'
                elif abs(value) < 1 and value != 0:
                    display_value = f'{value:.4f}'
                else:
                    display_value = f'{value:,.2f}'
            else:
                display_value = str(value)
            variable_details.append(
                html.Div([
                    html.Span(
                        f'📈 {var.replace("_", " ").capitalize()} ({get_unit(var)}):',
                        style={'flex': '7', 'text-align': 'left'}
                    ),
                    html.Span(
                        display_value,
                        style={'flex': '3', 'text-align': 'right'}
                    )
                ],
                style={
                    'display': 'flex',
                    'justify-content': 'space-between',
                    'width': '100%'
                })
            )
        return html.Div([
            html.Div([
                html.Span('📽️ ISIIS 1:', style={'flex': '3'}),
                html.A(media, href=media_link, target='_blank',
                       style={'flex': '7', 'text-align': 'right'})
            ], style={'display': 'flex', 'justify-content': 'space-between'}),
            html.Div([
                html.Span('📽️ ISIIS 2:', style={'flex': '3'}),
                html.A(media_2, href=media_link_2, target='_blank',
                       style={'flex': '7', 'text-align': 'right'})
            ], style={'display': 'flex', 'justify-content': 'space-between'}),
            html.Div([
                html.Span('⏳ Time:'),
                html.Span(formatted_time)
            ], style={'display': 'flex', 'justify-content': 'space-between'}),
            html.Div([
                html.Span('🌍 Latitude:'),
                html.Span(lat_display)
            ], style={'display': 'flex', 'justify-content': 'space-between'}),
            html.Div([
                html.Span('🌍 Longitude:'),
                html.Span(lon_display)
            ], style={'display': 'flex', 'justify-content': 'space-between'}),
            html.Div([
                html.Span('🌊 Depth:'),
                html.Span(depth_display)
            ], style={'display': 'flex', 'justify-content': 'space-between'}),
            html.Hr(),
            *variable_details
        ])
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