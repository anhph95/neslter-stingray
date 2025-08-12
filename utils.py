### Import libraries
import os
import re
import glob
import concurrent.futures
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.io import loadmat


# column_mappings.py
sled_columns = {
    'Timestamp': 'timestamp',
    'Times': 'times',
    'matdate': 'matdate',
    'Latitude': 'latitude',
    'Longitude': 'longitude',
    'anglePitch': 'pitch',
    'angleRoll': 'roll',
    'angleHeading': 'heading',
    'distanceAltitude': 'altitude',
    'Temperature': 'temperature',
    'Conductivity': 'conductivity',
    'Pressure': 'pressure',
    'Depth': 'depth',
    'Salinity': 'salinity',
    'Sound Velocity': 'sound_velocity',
    'Density': 'density',
    'Density_IES80': 'density_ies80',
    'Chlorophyll': 'chlorophyll',
    'Backscattering': 'backscattering',
    'Raw PAR [V]': 'par',
    'O2Concentration': 'oxygen_concentration',
    'AirSaturation': 'oxygen_saturation',
    'NitrateConcentration[uM]': 'nitrate',
}

sled_units = {
    'latitude': '°',
    'longitude': '°',
    'pitch': '°',
    'roll': '°',
    'heading': '°',
    'altitude': 'm',
    'temperature': '°C',
    'conductivity': 'S m⁻¹',
    'pressure': 'dbar',
    'depth': 'm',
    'salinity': 'psu',
    'sound_velocity': 'm s⁻¹',
    'density': 'kg m⁻³',
    'density_ies80': 'kg m⁻³',
    'chlorophyll': 'µg l⁻¹',
    'backscattering': 'm⁻¹ sr⁻¹',
    'par': 'µmol photons m⁻² s⁻¹',
    'oxygen_concentration': 'µM',
    'oxygen_saturation': '%',
    'nitrate': 'µM',
}

calibration_data = {
        "chlorophyll": {
            "2019": {'scale_factor': 0.0073, 'dark_count': 48},
            "2021": {'scale_factor': 0.0073, 'dark_count': 51},
            "function": lambda raw, scale_factor, dark_count: scale_factor * (raw - dark_count)
        },
        "backscatter": {
            "2019": {'scale_factor': 1.684e-06, 'dark_count': 48},
            "2021": {'scale_factor': 1.861e-06, 'dark_count': 46},
            "function": lambda raw, scale_factor, dark_count: scale_factor * (raw - dark_count)
        },
        "par": {
            "2019": {'a0': 0.983748497, 'a1': 8.129573432e-01, 'Im': 1.3589},
            "2021": {'a0': 0.942280491, 'a1': 8.215386e-01, 'Im': 1.3589},
            "function": lambda raw, a0, a1, Im: Im * 10 ** ((raw - a0) / a1)
        },
        
        "gps": {
            "function": lambda raw: np.floor(raw / 100) + (raw % 100) / 60
        }
    }

# Stingray sensor processing functions
def calibrate(sensor_type, raw_value, year=None):
    """
    Apply the calibration function to raw sensor data.
    
    :param sensor_type: str, type of sensor ("chlorophyll", "backscatter", "par").
    :param raw_value: float, raw sensor reading.
    :param year: str, year of calibration ("2019" or "2021" where applicable).
    :return: float, calibrated value.
    """  
    
    if sensor_type not in calibration_data:
        raise ValueError(f"Unknown sensor type: {sensor_type}")

    sensor_info = calibration_data[sensor_type]
    
    # Get calibration function and parameters
    calibration_func = sensor_info["function"]
    
    if year:
        params = sensor_info[year]
        # Apply function with extracted parameters
        return calibration_func(raw_value, **params)
    
    else:
        return calibration_func(raw_value)

## SUNA data processing functions
# Function to parse MATLAB .mat file to python dictionary
def parse_mat(mat_path,name=None):
    if not name:
        name = mat_path.split('/')[-1].split('.')[0]
    matfile = loadmat(mat_path)[name]
    data = {}
    for name in matfile.dtype.names:
        data[name] = matfile[name][0][0]
    return data

# Function to check if required lines are in CAL path, needed for correction function
def check_lines(file_path, initials='', out_dir='modified_CAL'):
    out_path = out_dir + '/' + os.path.basename(file_path).split('.')[0] + f'_{initials}_modified.CAL'
    if os.path.exists(out_path):
        print("Modified CAL file already exits.") 
        return out_path
    else:
        # Create output directory for modified CAL file
        os.makedirs(out_dir,exist_ok=True)
        
        # Block of lines to check and insert if missing
        required_lines = """H,Pixel base,1,,,
H,Sensor Depth offset,0,,,
H,Br wavelength offset,210,,,
H,Min fit wavelength,,,,
H,Max fit wavelength,,,,
H,Use seawater dark current,No,,,
H,Pressure coef,0.0265,,,"""
        
        # Load the file
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Check if the required block is present
        if required_lines not in content:
            # Find the last "H,File creation time" line
            insert_index = content.rfind("H,File creation time")
        
            if insert_index != -1:
                # Insert the missing block before the last "H,File creation time" line
                content = content[:insert_index] + required_lines + "\n" + content[insert_index:]
        
                # Write the updated content back to the file
                with open(out_path, 'w') as file:
                    file.write(content)
        
            print("Required lines not present in orginal file.")
            print(f"Generating new CAL file as {out_path}")
        else:
            print("Required lines are already present. Keeping original CAL file.")
            
    return out_path

# Function to parse Nitrate calibration file
def parse_SOSIK_NO3cal(cal_file):
    # Initialize the output dictionary with NaNs or empty lists/strings
    cal = {
        'type': 'SUNA',
        'SN': '',
        'CalTemp': None,
        'CalSDN': None,
        'CalDateStr': '',
        'DC_flag': 1,
        'pixel_base': None,
        'depth_lag': None,
        'WL_offset': None,
        'min_fit_WL': None,
        'max_fit_WL': None,
        'pres_coef': None,
    }

    replacements = {
        'New ': '',
        'DI DC Corr': 'Ref',
        'Reference': 'Ref',
        'Wavelength': 'WL',
        'WaveLen': 'WL',
        ',NO3': ',ENO3',
        ',SWA': ',ESW',
        ',ASW,': ',ESW,',
        ',T*ASW': ',TSWA'
    }

    try:
        with open(cal_file, 'r') as f:
            lines = f.readlines()

        data_lines = []


        for tline in lines:
            tline = tline.strip().split(',')

            # Header info
            if tline[0]=='H':
                if tline[1].startswith('SUNA'):
                    cal['type'] = 'SUNA'
                    cal['SN'] = re.search(r'SUNA\s+(\d+)', tline[1]).group(1)
                if tline[1].startswith('Lamp#'):
                    cal['SN'] = ' '.join(tline[1:])

                if re.search(r'\d+/\d+/\d{4}', tline[1]):
                    d_str = re.search(r'\d+/\d+/\d{4}', tline).group(0)
                    cal['CalDateStr'] = d_str
                    cal['CalSDN'] = datetime.strptime(d_str, '%m/%d/%Y').date()

                if 'creation time' in tline[1]:
                    d_str = re.search(r'\d{2}-\w{3}-\d{4}', tline[1]).group(0)
                    cal['CalDateStr'] = d_str
                    cal['CalSDN'] = datetime.strptime(d_str, '%d-%b-%Y').date()

                if 'CalTemp' in tline[1]:
                    cal['CalTemp'] = float(tline[2]) if tline[2] else None

                if re.search(r'T_CAL', tline[1]) and cal['CalTemp'] is None:
                    cal['CalTemp'] = float(tline[1].split(' ')[1])                    

                if 'Pixel base' in tline[1]:
                    cal['pixel_base'] = int(tline[2]) if tline[2] else None

                if 'Sensor Depth' in tline[1]:
                    cal['depth_lag'] = [float(x) if x else None for x in tline[2:4]]

                if 'Br wavelength' in tline[1]:
                    cal['WL_offset'] = float(tline[2]) if tline[2] else None

                if 'Min fit' in tline[1]:
                    cal['min_fit_WL'] = float(tline[3]) if tline[3] else None

                if 'Max fit' in tline[1]:
                    cal['max_fit_WL'] = float(tline[3]) if tline[3] else None

                if 'Use seawater dark' in tline[1]:
                    cal['DC_flag'] = 0 if 'yes' in tline[1].lower() else 1

                if 'Pressure coef' in tline[1]:
                    cal['pres_coef'] = float(tline[2])

                if 'Wavelength' in tline and 'NO3' in tline:
                    tline = ','.join(tline)
                    for old, new in replacements.items():
                        tline = tline.replace(old, new)
                    hdr = tline.split(',')[1:]

            if tline[0]=='E':
                data_lines.append(tline[1:])

        if len(data_lines) == 0:
            print(f"File exists but no data lines found for: {cal_file}")
            return cal

        # Convert data lines to a numpy array
        data = np.array(data_lines, dtype=float)

        # # To data frame
        # data = pd.DataFrame(data, columns = hdr)

        #Assign data to the respective fields in cal
        for i, name in enumerate(hdr):
            cal[name] = data[:, i]

    except FileNotFoundError:
        print(f"Cannot find calibration file at: {cal_file}")
        return cal

    return cal


# Bofu's correction algorithm
def calc_bofu_no3(spec, ncal, pcor_flag):
    fig_flag = 0  # Set to 1 to show nitrate fit
    WL_offset = ncal['WL_offset']
    
    d = spec.copy()
    d['P'] = d['STP'][2]
    d['T'] = d['STP'][1]
    d['S'] = d['STP'][0]
    del d['STP']
    
    if ncal['pixel_base'] == 0:
        d['spectra_pix_range'] += 1
        d['pix_fit_win'] += 1
        print('Pixel registration offset by +1')
    
    DC = d['DC'] if ncal['DC_flag'] != 0 else d['SWDC']
    
    if ncal['min_fit_WL']:
        d['pix_fit_win'][0] = np.where(ncal['WL'] >= ncal['min_fit_WL'])[0][0]
        d['WL_fit_win'][0] = ncal['WL'][d['pix_fit_win'][0]]
    
    if ncal['max_fit_WL']:
        d['pix_fit_win'][1] = np.where(ncal['WL'] <= ncal['max_fit_WL'])[0][-1]
        d['WL_fit_win'][1] = ncal['WL'][d['pix_fit_win'][1]]
    
    # Check data range
    if d['spectra_WL_range'][0][0] > d['WL_fit_win'][0][0]:
        print(f"Min WL of returned spectra ({d['spectra_WL_range'][0][0]}) is greater than Min WL of fit window ({d['WL_fit_win'][0][0]}).")
        d['WL_fit_win'][0][0] = d['spectra_WL_range'][0][0]
        print(f"Fit window adjusted [{d['WL_fit_win'][0][0]} {d['WL_fit_win'][0][1]}] & NO3 estimate will be compromised!")
    
    if d['spectra_WL_range'][0][1] < d['WL_fit_win'][0][1]:
        print(f"Max WL of returned spectra ({d['spectra_WL_range'][0][1]}) is less than Max WL of fit window ({d['WL_fit_win'][0][1]}).")
        d['WL_fit_win'][0][1] = d['spectra_WL_range'][0][1]
        print(f"Fit window adjusted [{d['WL_fit_win'][0][0]} {d['WL_fit_win'][0][1]}] & NO3 estimate will be compromised!")
    
    # Get pixel fit window for SUNA floats based on WL fit range
    if np.isnan(np.sum(d['pix_fit_win'])) and 'WL_fit_win' in d:
        pfit_low = np.where(ncal['WL'] >= d['WL_fit_win'][0][0])[0][0]
        pfit_hi = np.where(ncal['WL'] <= d['WL_fit_win'][0][1])[0][-1]
        d['pix_fit_win'] = [pfit_low, pfit_hi]
    
    # Get sample pixel range indexes
    ind1 = np.where(ncal['WL'] >= d['spectra_WL_range'][0][0])[0][0]
    ind2 = np.where(ncal['WL'] <= d['spectra_WL_range'][0][1])[0][-1]
    d['spectra_pix_range'] = [ind1, ind2]
    saved_pixels = np.arange(d['spectra_pix_range'][0], d['spectra_pix_range'][1] + 1)
    
    # Size of sample intensity spectra matrix
    rows, cols = d['UV_INTEN'].shape
    
    # Subset calibration data over pixel range identified from sample line
    REF = np.tile(ncal['Ref'][saved_pixels], (rows, 1))
    WL = np.tile(ncal['WL'][saved_pixels], (rows, 1))
    ESW = np.tile(ncal['ESW'][saved_pixels], (rows, 1))
    ENO3 = ncal['ENO3'][saved_pixels]
    
    if 'EHS' in ncal:
        EHS = ncal['EHS'][saved_pixels]
    
    # Subtract dark current intensities
    UV_INTEN_SW = d['UV_INTEN'] - DC
    
    # Check for UV_INTEN_SW <= 0 and set to NaN
    if np.count_nonzero(UV_INTEN_SW <= 0):
        UV_INTEN_SW[UV_INTEN_SW <= 0] = np.nan
        print('UV Intensities <= DC found. Setting these low intensities to NaN')
    
    # Absorbance spectra for all samples in profile
    ABS_SW = -np.log10(UV_INTEN_SW / REF)
    
    # # Calculate temperature corrected absorbances
    # A = 1.1500276
    # B = 0.02840
    # C = -0.3101349
    # D = 0.001222
    
    ctd_temp = np.tile(d['T'], (cols, 1)).T
    ctd_sal = np.tile(d['S'], (cols, 1)).T
    cal_temp = np.tile(ncal['CalTemp'], (rows, cols))
    
    Tcorr_coef = [1.27353e-07, -7.56395e-06, 2.91898e-05, 1.67660e-03, 1.46380e-02]
    Tcorr = np.polyval(Tcorr_coef, (WL - WL_offset)) * (ctd_temp - cal_temp)
    ESW_in_situ = ESW * np.exp(Tcorr)
    
    if pcor_flag == 1:
        pres_term = (1 - d['P'] / 1000 * ncal['pres_coef']).reshape(-1, 1)
        ESW_in_situ *= pres_term
    
    ABS_Br_tcor = ESW_in_situ * ctd_sal
    ABS_cor = ABS_SW - ABS_Br_tcor
    
    # Calculate the nitrate concentration and the slope and intercept of the baseline absorbance
    t_fit = (saved_pixels >= d['pix_fit_win'][0]) & (saved_pixels <= d['pix_fit_win'][1])
    
    Fit_WL = WL[0, t_fit]
    Fit_ENO3 = ENO3[t_fit]
    Ones = np.ones_like(Fit_ENO3)
    
    M = np.column_stack((Fit_ENO3, Ones / 100, Fit_WL / 1000))
    M_INV = np.linalg.pinv(M)
    colsM = M.shape[1]
    
    NO3 = np.full((rows, colsM + 3), np.nan)
    
    for i in range(rows):
        samp_ABS_cor = ABS_cor[i, t_fit]
        NO3[i, :3] = M_INV @ samp_ABS_cor
        NO3[i, 1] /= 100  # baseline intercept
        NO3[i, 2] /= 1000  # baseline slope
        
        ABS_BL = WL[i, :] * NO3[i, 2] + NO3[i, 1]
        ABS_NO3 = ABS_cor[i, :] - ABS_BL
        ABS_NO3_EXP = ENO3 * NO3[i, 0]
        FIT_DIF = ABS_cor[i, :] - ABS_BL - ABS_NO3_EXP
        RMS_ERROR = np.sqrt(np.sum(FIT_DIF[t_fit] ** 2) / np.sum(t_fit))
        ind_240 = np.where(Fit_WL <= 240)[0][-1]
        ABS_240 = [Fit_WL[ind_240], samp_ABS_cor[ind_240]]
        
        NO3[i, colsM:colsM + 3] = [RMS_ERROR, *ABS_240]
        
        # if fig_flag == 1 and not np.isnan(NO3[i, 0]):
        #     import matplotlib.pyplot as plt
            
        #     plt.figure(101, figsize=(12, 8))
        #     plt.plot(WL[i, :], ABS_SW[i, :], 'k-', linewidth=2)
        #     plt.plot(WL[i, :], ABS_BL, 'g-', linewidth=2)
        #     plt.plot(WL[i, :], ABS_Br_tcor[i, :] + ABS_BL, 'b-', linewidth=2)
        #     plt.plot(WL[i, :], ABS_cor[i, :], 'ro', linewidth=1)
        #     plt.plot(WL[i, :], ABS_NO3_EXP + ABS_BL, 'r-', linewidth=2)
        #     plt.xlim([Fit_WL[0] - 5, Fit_WL[-1] + 5])
        #     plt.xlabel('Wavelength, nm')
        #     plt.ylabel('Absorbance')
        #     plt.legend(['Sample', 'Baseline', 'Br+BL', 'NO3+BL obs', 'NO3+BL fit'], loc='upper right')
        #     plt.show()
    
    # Create an output dictionary
    out = {
        'hdr': ['SDN', 'AVG DC', 'Pres', 'Temp', 'Sal', 'SBE_NO3, uM/L', 'NO3, uM/L', 'BL_B', 'BL_M', 'RMS ERROR', 'WL~240', 'ABS~240'],
        'info': {
            'WL_fit_window': d['WL_fit_win'],
            'spectra_WL_range': d['spectra_WL_range'],
            'WL_offset': WL_offset
        }
    }
    
    avg_DC = np.nanmean(DC, axis=1)
    NO3 = np.column_stack((d['SDN'], avg_DC, d['P'], d['T'], d['S'], d['NO3'], NO3))
    tnan = np.isnan(NO3[:, 5])
    out['data'] = NO3[~tnan, :]
    
    return out

def calibrate_nitrate(df, CRUISE, nitrate_col='NitrateConcentration[uM]', dark_col='DarkValueUsedForFit'):
    """
    Calibrates nitrate concentration from SUNA sensor data.

    Parameters:
        df (pd.DataFrame): Dataframe containing nitrate data.
        CRUISE (str): Cruise identifier to find the calibration file.
        nitrate_col (str, optional): Column name for raw nitrate concentration. Default is 'NitrateConcentration[uM]'.
        dark_col (str, optional): Column name for dark value used in fit. Default is 'DarkValueUsedForFit'.

    Returns:
        np.array: Corrected nitrate concentrations.
    """
    df[nitrate_col] = pd.to_numeric(df[nitrate_col], errors='coerce')

    # Define calibration directory
    cal_dir = 'suna_calibration'
    spec_path = os.path.join(cal_dir, 'spec.mat')

    # Find the corresponding CAL file for the cruise
    try:
        cal_file = next(file for file in os.listdir(cal_dir) if CRUISE in file)
        cal_path = os.path.join(cal_dir, cal_file)
    except StopIteration:
        print(f'No calibration file found for cruise {CRUISE}, using raw nitrate values.')
        df.loc[df[dark_col] == 0, nitrate_col] = np.nan
        return df[nitrate_col].to_numpy()

    # Ensure required lines in CAL file
    cal_path = check_lines(cal_path, 'AP')

    # Load SPEC and CAL data
    spec = parse_mat(spec_path)
    ncal = parse_SOSIK_NO3cal(cal_path)

    # Get UV spectrum headers
    hdr_file = os.path.join(cal_dir, 'suna_hdr.txt')
    with open(hdr_file, 'r') as file:
        nitrate_hdr = file.readline().strip().split(',')

    # Extract UV spectrum column names
    uv_cols = [col for col in df.columns if re.search(r'SpectrumCh', col)]
    uv_list = [col for col in nitrate_hdr if re.search(r'UV\(\d+(\.\d+)?\)', col)]

    # Rename spectrum columns for consistency
    df.rename(columns=dict(zip(uv_cols, uv_list)), inplace=True)

    # Drop rows with missing nitrate values
    df[uv_list] = df[uv_list].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=[nitrate_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Prepare data for correction
    spec['STP'] = [df['Salinity'].to_numpy(), df['Temperature'].to_numpy(), df['Pressure'].to_numpy()]
    spec['UV_INTEN'] = df[uv_list].to_numpy(dtype=float)
    spec['DC'] = np.outer(df[dark_col], np.ones(len(uv_list)))  # Create a dark current matrix
    spec['NO3'] = df[nitrate_col].to_numpy()  # Raw SUNA-measured nitrate concentration
    spec['SDN'] = df['matdate'].to_numpy()

    # Define spectral wavelength range from calibration
    spec['spectra_WL_range'][0][0] = ncal['WL'][0]
    spec['spectra_WL_range'][0][1] = ncal['WL'][-1]

    # Perform nitrate correction calculation
    out = calc_bofu_no3(spec, ncal, pcor_flag=1)

    # Apply filtering: Set values to NaN where dark value mask is zero
    dv_mask = out['data'][:, 1] == 0
    out['data'][dv_mask, 6] = np.nan

    print(f'Calibration file found for cruise {CRUISE}, TSP correction applied.')
    return out['data'][:, 6]

### Date processing functions

# Function to convert datetime object(s) to MATLAB serial date numbers
def datenum(dts):
    # Define the base date (MATLAB uses Jan 0, 0000, but Python uses a different base)
    base_date = datetime(1, 1, 1)

    # Function to calculate the serial date number for a single datetime object
    def date2num(dt):
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()
        elif not isinstance(dt, datetime):
            raise TypeError("Input must be a datetime object or pandas.Timestamp")
        delta = dt - base_date
        return delta.days + delta.seconds / 86400 + 366 + 1

    # Check if the input is a list of datetime objects
    if isinstance(dts, (list, np.ndarray, pd.Series)):
        # Process each datetime object in the list
        return [date2num(dt) for dt in dts]
    elif isinstance(dts, (datetime, pd.Timestamp)):
        # Process a single datetime object
        return date2num(dts)
    else:
        raise TypeError("Input must be a datetime object or a list of datetime objects")

# Function to convert serial date number(s)/timestamp(s) back to datetime object
def datestr(date_numbers, fmt=0, as_str=False):
    # MATLAB base date is 0000-01-01, using closest Python date 0001-01-01
    # Adjust by one day because Python's datetime starts at 0001-01-01
    matlab_base_date = datetime(1, 1, 1)
    
    # Define format options similar to MATLAB's `datestr`
    formats = {
        0: '%d-%b-%Y %H:%M:%S',  # Default format
        1: '%d-%b-%Y',            # Day-Month-Year
        2: '%m/%d/%y',            # Month/Day/Year
        3: '%Y-%m-%d',            # Year-Month-Day
        4: '%d/%m/%Y',            # Day/Month/Year
        5: '%H:%M:%S',            # Time
    }
    
    def num2date(date_number):
        # Convert serial date number to datetime
        date_time_obj = matlab_base_date + timedelta(days=date_number - 366.0 -1.0)
        
        if as_str:
            # Use custom format if provided
            if isinstance(fmt, str):
                return date_time_obj.strftime(fmt)
            else:
                return date_time_obj.strftime(formats.get(fmt, '%d-%b-%Y %H:%M:%S'))
        else:
            return date_time_obj
    
    # Check if the input is a single serial date number or a list
    if isinstance(date_numbers, (int, float)):  # Single int or float
        return num2date(date_numbers)
    elif isinstance(date_numbers, (list, pd.Series, np.ndarray)):  # List, Series, or ndarray
        return [num2date(num) for num in date_numbers]
    elif date_numbers is None:  # Handle None case
        return None
    else:
        raise ValueError("Input should be an int, float, list, pandas Series, or numpy ndarray.")

# Function to convert timestamp to datetime and MATLAB serial date number
def convert_timestamp(timestamp, origin_date = datetime(1904, 1, 1)):
    times = [origin_date + timedelta(seconds=x) for x in timestamp]
    matdate = datenum(times)
    return times, matdate

def rawdate2date(series):
    # Extract year and day from the series and convert it to datetime in one step
    year = series.astype(str).str[:4]
    day_of_year = series.astype(str).str[4:].astype(int) - 1
    return pd.to_datetime(year, format='%Y') + pd.to_timedelta(day_of_year, unit='D')

def rawtime2time(series):
    # Convert string to float, divide by 24 to get a fraction of a day, then convert to timedelta
    day_fractions = series.astype(float) / 24
    return pd.to_timedelta(day_fractions, unit='D').dt.components.apply(
        lambda x: f"{x['hours']:02}:{x['minutes']:02}:{x['seconds']:02}", axis=1
    )

### Data processing functions
## New sensor data processing functions
# Function to read multiple CSV files concurrently with exception handling directly in the parallel process
def read_csv_parallel(file_list, max_workers=None):
    """
    Reads multiple CSV files in parallel using ThreadPoolExecutor and concatenates them into a single DataFrame.
    Exceptions are handled directly in the parallel process.
    
    Parameters:
    - file_list (list): A list of file paths to CSV files.
    - max_workers (int): The maximum number of worker threads to use. If not provided, it will use all available CPUs minus one.
    
    Returns:
    - concatenated_df (pandas.DataFrame): A single DataFrame containing the data from all the CSV files.
    """
    def safe_read_csv(file):
        try:
            return pd.read_csv(file)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            return pd.DataFrame()  # Return empty DataFrame if reading fails

    # Determine the number of workers based on available CPUs and number of files
    if max_workers is None:
        max_workers = min(os.cpu_count() - 1, len(file_list))
    
    # Use ThreadPoolExecutor to read files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        dfs = list(executor.map(safe_read_csv, file_list))

    # Concatenate the results into a single DataFrame
    concatenated_df = pd.concat(dfs, ignore_index=True)
    
    return concatenated_df

# Example usage
# filtered_files = [list of your csv file paths]
# concatenated_df = read_and_concat_csv_parallel(filtered_files)
# print(concatenated_df)

# Funciton to get the list of files in the directory
def filter_csv(dir_root, start_date, end_date = None):
    """
    Filters CSV files in the specified directory based on the date embedded in the filename.

    Parameters:
    - dir_root (str): The directory containing the CSV files.
    - start_date (datetime): The start of the date range for filtering.
    - end_date (datetime): The end of the date range for filtering.

    Returns:
    - file_filter (list): A list of filtered file paths based on the date range.
    """
    # Get list of all .csv files in the directory
    list_all = glob.glob(f'{dir_root}/*.csv')
    
    # Extract date from the filename and associate it with the file path
    # Get base name, remove anything before the first '-' and remove the extension (after the last '.')
    base = np.array([[datetime.strptime(os.path.basename(file).rsplit('.', 1)[0].split('-', 1)[1], '%Y-%m-%d %H-%M-%S.%f'), file] for file in list_all])

    # If end_date is not provided, set it to a far future date to include all files after start_date
    if end_date is None:
        end_date = datetime.max
        
    # Filter files based on the date range
    list_filter = base[:, 1][(base[:, 0] >= start_date) & (base[:, 0] <= end_date)]
    
    return list_filter.tolist()



# Function to merge two dataframes using merge_asof and optionally mask duplicates
def merge_df(df1, df2, on, cols=None, direction='backward', duplicates=False):
    """
    Merges two dataframes using merge_asof and optionally masks duplicates in specified columns.
    
    Parameters:
    - df1 (pd.DataFrame): The first dataframe.
    - df2 (pd.DataFrame): The second dataframe.
    - on (str): The column to merge on.
    - cols (list): List of columns to merge from df2. If None, uses columns from df2 that don't overlap with df1.
    - direction (str): Direction for merge_asof. Default is 'backward'.
    - duplicates (bool): If False, masks duplicate matches in df2.

    Returns:
    - pd.DataFrame: The merged dataframe, optionally with duplicates masked.
    """    
    
    # Make a copy of the dataframes to avoid modifying the original data
    df1 = df1.copy().sort_values(on).reset_index(drop=True)
    df2 = df2.copy().sort_values(on).reset_index(drop=True)

    # If cols is None, get non-overlapping columns from df2
    if not cols:
        cols = df2.columns.difference(df1.columns).tolist()
        
    # Add an 'id' column to track original rows in df2
    df2['id'] = df2.index
    
    # Perform the merge using merge_asof
    merged_df = pd.merge_asof(df1, df2[[on] + cols + ['id']], on=on, direction=direction)

    # Mask duplicates if the duplicates flag is False
    if not duplicates:
        merged_df[cols] = merged_df[cols].mask(merged_df.duplicated(subset='id'))

    # Drop the 'id' column as it's no longer needed
    merged_df.drop(columns='id', inplace=True)
    
    return merged_df


## Data binning
# Function to bin data based on specified columns and steps
def bin_data(df, cols, steps): # 15 seconds and 1 meter
    # Sort the dataframe by the specified columns
    df = df.copy()#.sort_values(by=cols)
    
    # Create bins for each column
    for col, step in zip(cols, steps):
        df[f'{col}_bin'] = np.floor((df[col] + step / 2) / step) * step # Round to the nearest step
        
    # Return the mean of each group
    return df

# Function got get ranges from midpoints
def mid2range(midpoints):
    # Initialize the ranges list with the first range boundary
    ranges = [midpoints[0] - (midpoints[1] - midpoints[0]) / 2]
    
    # Calculate midpoints between each adjacent pair
    for i in range(len(midpoints) - 1):
        midpoint = (midpoints[i] + midpoints[i + 1]) / 2
        ranges.append(midpoint)
        
    # Append the last range boundary
    ranges.append(midpoints[-1] + (midpoints[-1] - midpoints[-2]) / 2)
    
    return np.array(ranges)

# Function to read a single file
def read_file(filename, hdr = None):
    try:
        if hdr == True:
            hdr = open('suna_hdr.txt').readlines()[0].strip().split(',')
        # Read file efficiently and process the necessary columns
        temp = pd.read_csv(filename, skiprows=14, header=None)

        # Process only the necessary columns and optimize the DataFrame creation
        lhs = pd.DataFrame({
            0: pd.Series([float('nan')] * len(temp), dtype='float64'),
            1: rawdate2date(temp.iloc[:, 1]),  # Convert dates
            2: rawtime2time(temp.iloc[:, 2])   # Convert times
        })

        # Concatenate lhs and rhs
        rhs = temp.iloc[:, 3:-6].copy()  # Copy relevant part of temp
        df = pd.concat([lhs, rhs], axis=1)

        # Set column names more efficiently
        if hdr:
            df.columns = hdr[:-8]
        return df
    
    except Exception as e:
        print(f"Failed to read {filename}: {e}")
        return pd.DataFrame() 
    
## Joe Futrelle get nearest stations
def get_cruise_stations(cruise):
    url = 'https://nes-lter-data.whoi.edu/api/stations/{}.csv'.format(cruise)
    return pd.read_csv(url)

class StationLocator(object):
    def __init__(self, cruise):
        self.station_metadata = get_cruise_stations(cruise.lower())

    def station_distances(self, lat, lon):
        distances = []
        index = []
        for station in self.station_metadata.itertuples():
            index.append(station.Index)
            distance = geo_distance([lat,lon], [station.latitude, station.longitude]).km
            distances.append(distance)
        distances = pd.Series(distances, index=index)
        return distances

    def nearest_station(self, lat, lon):
        distances = self.station_distances(lat, lon)
        i = distances.idxmin()
        distance = distances.loc[i]
        station_name = self.station_metadata['name'][i]
        return station_name, distance
    
    def nearest_stations(self, df, lat_col='latitude', lon_col='longitude'):
        names, distances, index = [], [], []
        for row in df.itertuples():
            lat = getattr(row, lat_col)
            lon = getattr(row, lon_col)
            name, distance = self.nearest_station(lat, lon)
            names.append(name)
            distances.append(distance)
        
        return names, distances
    
    # Usage
    # locator = StationLocator(cruise)
    # df['nearest_station'], df['station_distance'] = locator.nearest_stations(df)
    
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
