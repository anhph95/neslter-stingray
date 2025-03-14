import os
import argparse
import pandas as pd
import pathlib
import logging
from utils import *

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def cli():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process sensor and media data for a given cruise.")
    parser.add_argument('--cruise', type=str, required=True, help='Cruise ID, required, e.g. EN706')
    parser.add_argument('--sensor_dir', type=str, default='rawdata', help='Path to the sensor data, default is rawdata')
    parser.add_argument('--media_dir', type=str, default='medialist', help='Path to the media data, default is medialist')
    parser.add_argument('--bin_cols', type=str, nargs='+', default=['matdate', 'depth'],
                        help='Columns to bin (space-separated list, e.g., \"matdate depth\")')
    parser.add_argument('--bin_steps', type=float, nargs='+', default=[30/86400, 1],
                        help='Steps to bin (space-separated list, e.g., \"0.000347 1\" [30 seconds converted to day by dividing for 86400, 1 meter])')
    return parser.parse_args()

def main():
    args = cli()
    cruise, sensor_dir, media_dir = args.cruise, args.sensor_dir, args.media_dir
    cols, steps = args.bin_cols, args.bin_steps

    pathlib.Path('dash_data').mkdir(parents=True, exist_ok=True)

    # Find matching sensor file
    try:
        sensor_file = next((file for file in os.listdir(sensor_dir) if file.endswith('.csv') and cruise in file), None)
    except FileNotFoundError:
        logging.error(f"Sensor directory '{sensor_dir}' not found.")
        return

    if sensor_file:
        try:
            sensor_df = pd.read_csv(os.path.join(sensor_dir, sensor_file))
            sensor_df['times'] = pd.to_datetime(sensor_df['times'], errors='coerce')
        except Exception as e:
            logging.error(f"Error reading sensor file {sensor_file}: {e}")
            return
    else:
        logging.error(f"No sensor file found for {cruise}. Process aborted.")
        return

    # Find matching media file
    try:
        media_file = next((file for file in os.listdir(media_dir) if file.endswith('.csv') and cruise in file), None)
    except FileNotFoundError:
        logging.warning(f"Media directory '{media_dir}' not found.")
        media_file = None

    if media_file:
        try:
            media_df = pd.read_csv(os.path.join(media_dir, media_file))
            media_df['times'] = pd.to_datetime(media_df['times'], errors='coerce')
            media_df.drop(columns=['id'], inplace=True, errors='ignore')
            df = merge_df(sensor_df, media_df, on='times', cols=media_df.columns[1:].to_list())
        except Exception as e:
            logging.error(f"Error reading media file {media_file}: {e}")
            return
    else:
        logging.warning(f"No media file found for {cruise}. Proceeding with sensor data only.")
        df = sensor_df

    df_bin = bin_data(df, cols, steps)
    df_bin['group'] = pd.factorize(pd.MultiIndex.from_arrays([df_bin[f'{col}_bin'] for col in cols]))[0]

    sensor_cols = sensor_df.columns[3:].to_list()
    meta_cols = [col for col in df.columns if col not in sensor_cols]

    df_mean = df_bin.groupby('group').agg(
        {col: 'first' for col in meta_cols} |
        {col: ['mean', 'std'] for col in sensor_cols}
    ).reset_index(drop=True)

    df_mean.columns = [
        col[0] if col[1] in ['first', 'mean'] else f"{col[0]}_{col[1]}" for col in df_mean.columns
    ]

    date_string = df_mean['times'][0].strftime('%Y%m%d')
    output_file = f'dash_data/data/{date_string}_{cruise}.csv'
    df_mean.to_csv(output_file, encoding='utf-8', index=False)
    logging.info(f"Processed {date_string}_{cruise}. Data saved as {output_file}")

if __name__ == '__main__':
    main()
