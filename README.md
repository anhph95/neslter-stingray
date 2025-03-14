# NES-LTER Stingray

## 📌 Overview
This repository contains code and data processing scripts for the In-situ Ichthyoplankton Imaging System (ISIIS), also known as Stingray.

## 📦 Installation
### Clone this repository
```sh
git clone https://github.com/anhph95/nes-lter-stingray.git
cd nes-lter-stingray
```

###  Create Python virtual environment (venv):
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

### (Optional) Get media list and Tator link, not reccommended for real time, requires tator, opencv
```
usage: media_list.py [-h] [--cruise CRUISE] [--host HOST] [--project-id PROJECT_ID] [--token TOKEN] [--media-dir MEDIA_DIR]

options:
  -h, --help            show this help message and exit
  --cruise CRUISE       Cruise ID, required, e.g. EN706
  --host HOST           Tator host IP address, defaul is https://tator.whoi.edu
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

### Visualize data via calm.whoi.edu host
Data from dash_data\data folder can then be copied into 
```
\\vast.whoi.edu\proj\nes-lter\stingray_dashboard\dash_data\data
```
The dash board can then be accessed via http://calm.whoi.edu:8050


### OR run the visualization app on your own, required plotly, dash
```
usage: dashapp.py [-h] [--host HOST] [--port PORT]

Stingray Dashboard

options:
  -h, --help   show this help message and exit
  --host HOST  Host IP address for the Dash app, default is 0.0.0.0
  --port PORT  Port number for the Dash app, default is 8050
```

## 📚 License
This project is licensed under the **MIT License**. See the LICENSE file for details.

## ✨ Contributors
- **Anh Pham** - Developer

---
🔗 **GitHub Repository:** https://github.com/anhph95/nes-lter-stingray

