# NES-LTER Stingray

## 📌 Overview
This repository contains code and data processing scripts for the In-situ Ichthyoplankton Imaging System (ISIIS), also known as Stingray.

## 📦 Installation
### Clone this repository
```sh
git clone https://github.com/anhph95/nes-lter-stingray.git
cd nes-lter-stingray
```

###  Installation
####  Create a Python virtual environment and install dependencies:
```bash
sudo apt update
sudo apt install python3-venv -y
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Using Conda 
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda env create -f environment.yml
conda activate venv
```

## 🚀 Usage
```bash
python data_merge.py --cruise EN999 --start_date 2025-01-01 --end_date 2025-01-31
python data_bin.py --cruise EN999
python dashapp.py
```

## 📚 License
This project is licensed under the **MIT License**. See the LICENSE file for details.

## ✨ Contributors
- **Anh Pham** - Developer

---
🔗 **GitHub Repository:** https://github.com/anhph95/nes-lter-stingray

🚀 **Stingray Dashboard:** http://calm.whoi.edu:8050
