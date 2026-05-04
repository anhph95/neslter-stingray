# src/stingray/config/columns.py

from typing import Dict

SLED_COLUMNS: Dict[str, str] = {
    "Timestamp": "timestamp",
    "Times": "times",
    "matdate": "matdate",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "anglePitch": "pitch",
    "angleRoll": "roll",
    "angleHeading": "heading",
    "distanceAltitude": "altitude",
    "Temperature": "temperature",
    "Conductivity": "conductivity",
    "Pressure": "pressure",
    "Depth": "depth",
    "Salinity": "salinity",
    "Sound Velocity": "sound_velocity",
    "Density": "density",
    "Density_IES80": "density_ies80",
    "Chlorophyll": "chlorophyll",
    "Backscattering": "backscattering",
    "Raw PAR [V]": "par",
    "O2Concentration": "oxygen_concentration",
    "AirSaturation": "oxygen_saturation",
    "NitrateConcentration[uM]": "nitrate",
}