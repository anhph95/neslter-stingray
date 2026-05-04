# StingrayTools

Tools for processing, organizing, and visualizing NES-LTER Stingray / ISIIS sensor and imaging data.

This package includes sensor-processing utilities, CTD/profile handling, image-link helpers, CSV I/O tools, statistical helpers, and a Dash dashboard for interactive exploration of Stingray cruise data.

[![DOI](https://zenodo.org/badge/946902610.svg)](https://doi.org/10.5281/zenodo.15025961)

---

## Repository layout

```text
src/
├── stingray/
│   ├── cli/              # Command-line entry points
│   ├── config/           # Column definitions and unit mappings
│   ├── dashboard/        # Dash dashboard application
│   ├── data_reference/   # Reference files used by processing routines
│   ├── images/           # Image/frame/media-link helper scripts
│   ├── io/               # CSV, SUNA, and indexing utilities
│   ├── logging/          # Logging setup
│   ├── profiles/         # Profile/cast identification utilities
│   ├── sensors/          # Sensor-specific processing modules
│   ├── stats/            # Statistical utilities
│   ├── utils/            # Spatial, temporal, and gridding utilities
│   ├── getctd.py
│   ├── process.py
│   └── utils.py
└── neslter_stingray.egg-info/
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/anhph95/StingrayTools.git
cd StingrayTools
```

### Install with a Python virtual environment

```bash
sudo apt update
sudo apt install python3-venv -y
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Or install with Conda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda env create -f environment.yml
conda activate stingray
pip install -e .
```

---

## Command-line usage

### Process Stingray sensor data

```bash
python -m stingray.process \
  --cruise EN706 \
  --start YYYY-MM-DD \
  --end YYYY-MM-DD \
  --root sensor_data \
  --out-dir dash_data/data/stingray_timebinned
```

Common options:

```text
--cruise CRUISE
    Cruise ID, e.g. EN706.

--start START
    Cruise start date in YYYY-MM-DD format.

--end END
    Cruise end date in YYYY-MM-DD format.

--root ROOT
    Path to the raw sensor-data directory. Default: sensor_data.

--cal-year CAL_YEAR
    Sensor calibration year. Default: 2021.

--time-bin-seconds TIME_BIN_SECONDS
    Time-bin size in seconds. Default: 5.

--out-dir OUT_DIR
    Output directory for dashboard-ready CSV files.

--media-list-dirs MEDIA_LIST_DIRS ...
    Media-list directories for ISIIS image links.

--overwrite-index
    Rebuild cached sensor-file indexes.

--log-level {DEBUG,INFO,WARNING,ERROR}
    Logging level. Default: INFO.
```

### Sensor and image utilities

The package also includes modules for sensor-specific processing and image/media metadata handling:

```text
stingray.sensors.ctd
stingray.sensors.fluorometer
stingray.sensors.par
stingray.sensors.suna
stingray.sensors.merge
stingray.images.frame_timestamp
stingray.images.get_tator_link
stingray.images.abundance
stingray.images.generate_training
```

Use the relevant module directly or import functions from Python scripts and notebooks as needed.

---

## Dashboard usage

The dashboard reads data from the configured `dash_data/` directory. Expected local structure:

```text
dash_data/
├── data/
│   └── <dataset_name>/
│       └── *.csv
└── misc/
    ├── NESLTER_station_list.csv
    └── NESLTER_transect_bathymetry.csv
```

### Run the dashboard locally

```bash
python -m stingray.dashboard.app --host 0.0.0.0 --port 8050
```

Then open:

```text
http://localhost:8050
```

### Use the WHOI-hosted dashboard

Dashboard-ready data from `dash_data/data` can be copied to:

```text
\\vast.whoi.edu\proj\nes-lter\stingray_dashboard\dash_data\data
```

The dashboard can then be accessed at:

```text
https://stingraydash.whoi.edu/
```

---

## Docker usage

The dashboard can also be run with Docker Compose.

### Start the dashboard

```bash
docker compose up --build
```

This builds the image, mounts `dash_data/`, and serves the app at:

```text
http://localhost:8050
```

### Stop the dashboard

```bash
docker compose down
```

### Run in detached mode

```bash
docker compose up -d
```

Stop it with:

```bash
docker compose down
```

### Updating data

If `dash_data/` is mounted as a bind volume, updating CSV files does not require rebuilding the Docker image.

### Rebuilding after code changes

Rebuild after modifying source code, dashboard code, assets, dependencies, or Docker configuration:

```bash
docker compose up --build
```

---

## Development notes

Install in editable mode during development:

```bash
pip install -e .
```

Useful checks:

```bash
python -m stingray.process --help
python -m stingray.dashboard.app --help
```

Generated folders such as `__pycache__/`, `.egg-info/`, and local data/output directories should generally not be committed unless intentionally tracked.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributors

- Anh Pham
- Sidney Batchelder
- Heidi Sosik

---

GitHub Repository: https://github.com/anhph95/StingrayTools
