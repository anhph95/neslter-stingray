# NES-LTER Stingray

## 📌 Overview
This repository contains code and data processing scripts for the In-situ Ichthyoplankton Imaging System (ISIIS), also known as **Stingray**.

[![DOI](https://zenodo.org/badge/946902610.svg)](https://doi.org/10.5281/zenodo.15025961)

---

## 📦 Installation

### Clone this repository
```bash
git clone https://github.com/anhph95/nes-lter-stingray.git
cd nes-lter-stingray
```

### Create Python virtual environment (venv):
```bash
sudo apt update
sudo apt install python3-venv -y
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Or create Conda environment:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda env create -f environment.yml
conda activate venv
```

---

## 🚀 Usage

### Merge sensor data
```
usage: data_merge.py [-h] [--path PATH] [--cal_year CAL_YEAR] --cruise CRUISE --start_date START_DATE --end_date END_DATE

Merge sensor data from multiple CSV files.

options:
  -h, --help            show this help message and exit
  --path PATH           Path to the sensor data, default is /mnt/vast/nes-lter/Stingray/data/sensor_data
  --cal_year CAL_YEAR   Sensor calibration year, default is 2021
  --cruise CRUISE       Cruise ID, required, e.g. EN706
  --start_date START_DATE
                        Cruise start date (YYYY-MM-DD), required
  --end_date END_DATE   Cruise end date (YYYY-MM-DD), required
```

### (Optional) Get media list and Tator link (not recommended for real-time, requires `tator`, `opencv`)
```
usage: media_list.py [-h] [--cruise CRUISE] [--host HOST] [--project-id PROJECT_ID] [--token TOKEN] [--media-dir MEDIA_DIR]

options:
  -h, --help            show this help message and exit
  --cruise CRUISE       Cruise ID, required, e.g. EN706
  --host HOST           Tator host IP address, default is https://tator.whoi.edu
  --project-id PROJECT_ID
                        Project ID
  --token TOKEN         Tator login token, string or token file
  --media-dir MEDIA_DIR
                        Path to the media data
```

### Bin data and compute mean, standard deviation
```
usage: data_bin.py [-h] --cruise CRUISE [--sensor_dir SENSOR_DIR] [--media_dir MEDIA_DIR] [--bin_cols BIN_COLS [BIN_COLS ...]] [--bin_steps BIN_STEPS [BIN_STEPS ...]]

Process sensor and media data for a given cruise.

options:
  -h, --help            show this help message and exit
  --cruise CRUISE       Cruise ID
  --sensor_dir SENSOR_DIR
                        Path to the sensor data
  --media_dir MEDIA_DIR
                        Path to the media data
  --bin_cols BIN_COLS [BIN_COLS ...]
                        Columns to bin (space-separated list, e.g., "matdate depth")
  --bin_steps BIN_STEPS [BIN_STEPS ...]
                        Steps to bin (space-separated list, e.g., "0.000347 1" [30 seconds/86400, 1 meter])
```

### Visualize data
Data from `dash_data/data` can be copied into:
```
\\vast.whoi.edu\proj\nes-lter\stingray_dashboard\dash_data\data
```

The dashboard can then be accessed via:  
👉 [https://stingraydash.whoi.edu/](https://stingraydash.whoi.edu/)

---

### OR run the visualization app locally (requires `plotly`, `dash`)
```
usage: dashapp.py [-h] [--host HOST] [--port PORT]

Stingray Dashboard

options:
  -h, --help   show this help message and exit
  --host HOST  Host IP address for the Dash app, default is 0.0.0.0
  --port PORT  Port number for the Dash app, default is 8050
```

---

## 🐳 Running via Docker

The recommended way to run the Stingray Dashboard is with **Docker Compose**.

### 🛠 Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed  
- [Docker Compose](https://docs.docker.com/compose/) (bundled with Docker Desktop on most systems)  

---

### 1. Start the app (build if first time or code changed)
```bash
docker compose up --build
```

This will:
- Build the Docker image (if not already built or if code changed)  
- Mount the `dash_data/` folder so new data is available without rebuilding  
- Run the dashboard on [http://localhost:8050](http://localhost:8050)  

---

### 2. Stop the app
```bash
docker compose down
```

This stops and removes the running container(s).

---

### 3. Start in background (detached mode)
```bash
docker compose up -d
```

The dashboard will run in the background.  
Stop it anytime with:
```bash
docker compose down
```

---

### 4. Update data
Because `dash_data/` is mounted as a bind volume, any changes to CSVs or input files in `dash_data/` on your machine are immediately reflected inside the container.  

👉 **No rebuild is required** when updating data.

---

### 5. Rebuild when code changes
You only need to rebuild the image if you:
- Modify `dashapp.py`  
- Change code in the `assets/` folder  
- Update `requirements.txt`  

Rebuild with:
```bash
docker compose up --build
```

---

### 6. Access the dashboard
After startup, visit in your browser:  
👉 [http://localhost:8050](http://localhost:8050)  

Or, from another computer on the same network (LAN):  
👉 `http://<your-computer-ip>:8050`  

---

## 📚 License
This project is licensed under the **MIT License**. See the LICENSE file for details.

---

## ✨ Contributors
- **Anh Pham**  
- **Sidney Batchelder**  
- **Heidi Sosik**  

---
🔗 **GitHub Repository:** [https://github.com/anhph95/nes-lter-stingray](https://github.com/anhph95/nes-lter-stingray)  
