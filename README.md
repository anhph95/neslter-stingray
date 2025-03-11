# If venv not installed
sudo apt update
sudo apt install python3-venv -y

# Create venv
python3 -m venv stingray
source stingray/bin/activate

# Install requirements
pip install -r requirements.txt