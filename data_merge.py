import pathlib
import argparse
import logging
from datetime import datetime
from utils import *

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def cli():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Merge sensor data from multiple CSV files.")
    parser.add_argument('--path', type=str, default='/mnt/vast/nes-lter/Stingray/data/sensor_data', 
                        help='Path to the sensor data, default is /mnt/vast/nes-lter/Stingray/data/sensor_data')
    parser.add_argument('--cal_year', type=int, default=2021, 
                        help='Sensor calibration year, default is 2021')
    parser.add_argument('--cruise', type=str, required=True, 
                        help='Cruise ID, required, e.g. EN706')
    parser.add_argument('--start_date', type=str, required=True, 
                        help='Cruise start date (YYYY-MM-DD), required')
    parser.add_argument('--end_date', type=str, required=True, 
                        help='Cruise end date (YYYY-MM-DD), required')
    return parser.parse_args()

def main():
    try:
        args = cli()
        args.cal_year = str(args.cal_year)
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        pathlib.Path('rawdata').mkdir(parents=True, exist_ok=True)

        # Load sensor data
        logging.info("Reading sensor data...")
        ctd = read_csv_parallel(filter_csv(f'{args.path}/CTD', start_date, end_date))
        dvl = read_csv_parallel(filter_csv(f'{args.path}/DVL', start_date, end_date))
        fluorometer = read_csv_parallel(filter_csv(f'{args.path}/Fluorometer', start_date, end_date))
        gps = read_csv_parallel(filter_csv(f'{args.path}/GPS', start_date, end_date))
        oxygen = read_csv_parallel(filter_csv(f'{args.path}/Oxygen', start_date, end_date))
        par = read_csv_parallel(filter_csv(f'{args.path}/PAR', start_date, end_date))
        nitrate = read_csv_parallel(filter_csv(f'{args.path}/SUNA', start_date, end_date))

        # Merge all data
        logging.info("Processing CTD data...")
        ctd['Density_IES80'] = ies80(ctd['Salinity'], ctd['Temperature'], ctd['Pressure'] / 10)
        ctd['Times'], ctd['matdate'] = convert_timestamp(ctd['Timestamp'])
        sled = ctd[['Timestamp', 'Times', 'matdate', 'Temperature', 'Conductivity', 
                    'Pressure', 'Depth', 'Salinity', 'Sound Velocity', 'Density', 'Density_IES80']].copy()

        # Merge GPS data
        if not gps.empty:
            gps['Latitude'] = calibrate('gps', gps['Latitude'])
            gps['Longitude'] = calibrate('gps', gps['Longitude'])
            sled = merge_df(sled, gps, on='Timestamp', cols=['Latitude', 'Longitude'], direction='nearest', duplicates=True)
        else:
            logging.warning("GPS data is missing. Skipping GPS merge.")

        # Merge DVL data
        if not dvl.empty:
            sled = merge_df(sled, dvl, on='Timestamp', cols=['anglePitch', 'angleRoll', 'angleHeading', 'distanceAltitude'])
        else:
            logging.warning("DVL data is missing. Skipping DVL merge.")
        
        # Merge Fluorometer data
        if not fluorometer.empty:
            fluorometer['Chlorophyll'] = calibrate('chlorophyll', fluorometer['Chlorophyll'], args.cal_year)
            fluorometer['Backscattering'] = calibrate('backscatter', fluorometer['Backscattering'], args.cal_year)
            sled = merge_df(sled, fluorometer, on='Timestamp', cols=['Chlorophyll', 'Backscattering'])
        else:
            logging.warning("Fluorometer data is missing. Skipping Fluorometer merge.")
        
        # Merge PAR data
        if not par.empty:
            par['Raw PAR [V]'] = calibrate('par', par['Raw PAR [V]'], args.cal_year)
            sled = merge_df(sled, par, on='Timestamp', cols=['Raw PAR [V]'])
        else:
            logging.warning("PAR data is missing. Skipping PAR merge.")
        
        # Merge Oxygen data
        if not oxygen.empty:
            sled = merge_df(sled, oxygen, on='Timestamp', cols=['O2Concentration', 'AirSaturation'])
        else:
            logging.warning("Oxygen data is missing. Skipping Oxygen merge.")
        
        # Merge Nitrate data
        if not nitrate.empty:
            nitrate = merge_df(nitrate, sled, on='Timestamp', duplicates=True)
            nitrate['NitrateConcentration[uM]'] = calibrate_nitrate(nitrate, args.cruise)
            sled = merge_df(sled, nitrate, on='Timestamp', cols=['NitrateConcentration[uM]'])
        else:
            logging.warning("Nitrate data is missing. Skipping Nitrate merge.")

        # Change column names
        sled.rename(columns=sled_columns, inplace=True)

        # Save output CSV
        output_filename = f'rawdata/{start_date.strftime("%Y%m%d")}_{args.cruise}.csv'
        sled.to_csv(output_filename, encoding='utf-8', index=False)
        logging.info(f"Data merge completed! File saved as {output_filename}")

    except Exception as e:
        logging.error(f"Error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    main()
