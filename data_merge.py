import pathlib
import argparse
from datetime import datetime
from utils import *

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to the sensor data', default='/mnt/vast/nes-lter/Stingray/data/sensor_data')
    parser.add_argument('--cal_year', type=str, help='Calibration year', default='2021')
    parser.add_argument('--cruise', type=str, help='Cruise ID')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)')
    args = parser.parse_args()
    return args

def main():
    args = cli()
    root = args.path
    cal_year = args.cal_year
    CRUISE = args.cruise
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    pathlib.Path('rawdata').mkdir(parents=True, exist_ok=True)

    # Load each sensor data
    ctd = read_csv_parallel(filter_csv(f'{root}/CTD', start_date, end_date))
    dvl = read_csv_parallel(filter_csv(f'{root}/DVL', start_date, end_date))
    fluorometer = read_csv_parallel(filter_csv(f'{root}/Fluorometer', start_date, end_date))
    gps = read_csv_parallel(filter_csv(f'{root}/GPS', start_date, end_date))
    oxygen = read_csv_parallel(filter_csv(f'{root}/Oxygen', start_date, end_date))
    par = read_csv_parallel(filter_csv(f'{root}/PAR', start_date, end_date))
    nitrate = read_csv_parallel(filter_csv(f'{root}/SUNA', start_date, end_date))

    # Merge all data
    # Initialize the merged dataframe with the CTD data since it have the highest frequency
    # ctd['Density'] = gsw.rho(ctd['Salinity'], ctd['Temperature'], ctd['Pressure']-10.1325)
    ctd['Density'] = ies80(ctd['Salinity'], ctd['Temperature'], ctd['Pressure']/10)
    ctd['Times'], ctd['matdate'] = convert_timestamp(ctd)
    sled = ctd[['Timestamp', 'Times','matdate', 'Temperature', 'Conductivity', 'Pressure', 'Depth','Salinity', 'Density']]

    # Merge GPS data
    # Transform GPS coordinates to decimal degrees
    gps_corr = gps.copy()
    gps_corr['Latitude'] = calibrate('gps', gps_corr['Latitude'])
    gps_corr['Longitude'] = calibrate('gps', gps_corr['Longitude'])
    sled = merge_df(sled, gps_corr, on='Timestamp', cols=['Latitude', 'Longitude'], direction='nearest', duplicates=True)

    # Merge Nitrate data
    nitrate_corr = nitrate.copy()
    nitrate_corr = merge_df(nitrate_corr, sled, on='Timestamp', duplicates=True)
    nitrate_corr['NitrateConcentration[uM]'] = calibrate_nitrate(nitrate_corr, CRUISE)
    sled = merge_df(sled, nitrate_corr, on='Timestamp', cols=['NitrateConcentration[uM]'])

        
    # Merge Fluorometer data
    fluorometer_corr = fluorometer.copy()
    fluorometer_corr['Chlorophyll'] = calibrate('chlorophyll', fluorometer_corr['Chlorophyll'], cal_year)
    fluorometer_corr['Backscattering'] = calibrate('backscatter', fluorometer_corr['Backscattering'], cal_year)
    sled = merge_df(sled, fluorometer_corr, on='Timestamp', cols=['Chlorophyll', 'Backscattering'])

    # Merge PAR data
    par_corr = par.copy()
    par_corr['Raw PAR [V]'] = calibrate('par', par_corr['Raw PAR [V]'], cal_year)
    sled= merge_df(sled, par_corr, on='Timestamp', cols=['Raw PAR [V]'])

    # Merge Oxygen data
    sled = merge_df(sled, oxygen, on='Timestamp', cols=['O2Concentration','AirSaturation'])

    # Merge DVL data
    sled = merge_df(sled, dvl, on='Timestamp', cols=['anglePitch', 'angleRoll', 'angleHeading', 'distanceAltitude'])

    # Change column names
    sled.rename(columns=sled_columns, inplace=True)

    # Save output CSV
    output_filename = f'rawdata/{start_date.strftime("%Y%m%d")}_{CRUISE}.csv'
    sled.to_csv(output_filename, encoding='utf-8', index=False)
    print(f'Data merge completed! File saved as {output_filename}')

if __name__ == '__main__':
    main()