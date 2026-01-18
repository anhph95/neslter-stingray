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
    parser.add_argument('--sensor_dir', type=str, default='raw_data', help='Path to the sensor data, default is raw_data')
    parser.add_argument('--media_dir', type=str, default='media_list/ISIIS1', help='Path to the media data for ISIIS1, default is media_list/ISIIS1')
    parser.add_argument('--media_dir2', type=str, default='media_list/ISIIS2', help='Path to the media data for ISIIS2, default is media_list/ISIIS2')
    parser.add_argument('--out_dir', type=str, default='dash_data/data/stingray', help='Path to the output directory, default is dash_data/data')
    parser.add_argument('--bin_cols', type=str, nargs='+', default=['depth', 'times'],
                        help='Columns to bin (space-separated list, e.g., \"depth times\")')
    parser.add_argument('--bin_steps', type=float, nargs='+', default=[1, 20e9],
                        help='Steps to bin (space-separated list, e.g., \"1 20e9\" [1 meter, 20 seconds in nanoseconds])')
    parser.add_argument('--store_merge', action='store_true', help='Whether to store merged data')
    parser.add_argument('--merged_dir', type=str, default='merged_data', help='Path to the merged data directory, default is merged_data')
    return parser.parse_args()

def main():
    args = cli()
    cruise, sensor_dir, media_dir, media_dir2, out_dir, merged_dir = args.cruise, args.sensor_dir, args.media_dir, args.media_dir2, args.out_dir, args.merged_dir
    cols, steps = args.bin_cols, args.bin_steps
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Find matching sensor file
    try:
        logging.info(f"Looking for sensor files in {sensor_dir}")
        sensor_file = next((file for file in os.listdir(sensor_dir) if file.endswith('.csv') and cruise in file), None)
    except FileNotFoundError:
        logging.error(f"Sensor directory '{sensor_dir}' not found.")
        return

    if sensor_file:
        try:
            sensor_df = pd.read_csv(os.path.join(sensor_dir, sensor_file))
            sensor_df['times'] = pd.to_datetime(sensor_df['times'], errors='coerce')
            sensor_df.sort_values('times', inplace=True)
        except Exception as e:
            logging.error(f"Error reading sensor file {sensor_file}: {e}")
            return
    else:
        logging.error(f"No sensor file found for {cruise}. Process aborted.")
        return

    # Find matching media file
    try:
        logging.info(f"Looking for media files in {media_dir}")
        media_file = next((file for file in os.listdir(media_dir) if file.endswith('.csv') and cruise in file), None)
    except FileNotFoundError:
        logging.warning(f"Media directory '{media_dir}' not found.")
        media_file = None

    if media_file:
        try:
            logging.info("Merging media files...")
            media_df = pd.read_csv(os.path.join(media_dir, media_file))
            media_df['times'] = pd.to_datetime(media_df['times'], errors='coerce')
            media_df.sort_values('times', inplace=True)
            media_df.drop(columns=['id'], inplace=True, errors='ignore')
            df = pd.merge(sensor_df, media_df, on='times', how='outer').sort_values('times').reset_index(drop=True)
        except Exception as e:
            logging.error(f"Error reading media file {media_file}: {e}")
            return
    else:
        logging.warning(f"No media file found for {cruise}. Proceeding with sensor data only.")
        df = sensor_df

    # Find matching media file
    try:
        logging.info(f"Looking for media files in {media_dir2}")
        media_file2 = next((file for file in os.listdir(media_dir2) if file.endswith('.csv') and cruise in file), None)
    except FileNotFoundError:
        logging.warning(f"Media directory '{media_dir2}' not found.")
        media_file2 = None
        
    if media_file2:
        try:
            logging.info("Merging media files...")
            media_df2 = pd.read_csv(os.path.join(media_dir2, media_file2))
            media_df2['times'] = pd.to_datetime(media_df2['times'], errors='coerce')
            media_df2.sort_values('times', inplace=True)
            media_df2.drop(columns=['id'], inplace=True, errors='ignore')
            df = pd.merge(df, media_df2, on='times', how='outer', suffixes=('', '_2')).sort_values('times').reset_index(drop=True)
        except Exception as e:
            logging.error(f"Error reading media file {media_file2}: {e}")
            return
    else:
        logging.warning(f"No ISIIS 2 file found for {cruise}. Proceeding...")
        
    # Interpolate missing binning data
    df.dropna(subset=['times'], inplace=True)
    df.sort_values('times', inplace=True)
    if args.store_merge:
        date_string = df['times'][0].strftime('%Y%m%d')
        df = df.to_csv(f'{merged_dir}/{date_string}_{cruise}.csv')
        logging.info(f"Merged data saved as {merged_dir}/{date_string}_{cruise}.csv")
    logging.info("Interpolating binning data...")
    df.set_index('times', inplace=True)
    for col in cols:
        if col != "times":
            df[col] = df[col].interpolate(method="time")
    df = df.reset_index()

    logging.info(f"Binning data...")
    df_bin = bin_data(df, cols, steps)

    # Replace original cols with their binned versions
    for col in cols:
        bin_col = f"{col}_bin"
        if bin_col in df_bin.columns:
            df_bin[col] = df_bin[bin_col]   # overwrite original
            df_bin.drop(columns=[bin_col], inplace=True)  # drop _bin col

    # Create a group identifier
    df_bin['group'] = pd.factorize(
        pd.MultiIndex.from_arrays([df_bin[col] for col in cols])
    )[0]

    # Everything from sensor_df (except housekeeping cols) is a sensor col
    sensor_cols = [
        col for col in sensor_df.columns
        if col not in ['timestamp', 'matdate'] and col not in cols
    ]

    # Media cols are whatever is left in df_bin, excluding group + sensor cols
    media_cols = [
        col for col in df_bin.columns
        if col not in sensor_cols and col not in ['group', 'timestamp', 'matdate'] and col not in cols
    ]
    
    # Aggregate by group
    df_mean = df_bin.groupby('group').agg(
        {col: 'first' for col in cols} |
        {col: lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan for col in media_cols if col not in cols} |
        {col: ['mean', 'std'] for col in sensor_cols if col not in cols}
    ).reset_index(drop=True)

    df_mean.columns = [
        # if it's a sensor col, apply custom renaming
        (col[0] if col[1] == 'mean' else f"{col[0]}_{col[1]}")
        if col[0] in sensor_cols
        # otherwise, just keep the base name
        else col[0]
        for col in df_mean.columns
    ]

    # Reorder columns to match original dataframe
    df_mean = df_mean.sort_values('times')
    date_string = df_mean['times'][0].strftime('%Y%m%d')
    output_file = f'{out_dir}/{date_string}_{cruise}.csv'
    df_mean.to_csv(output_file, encoding='utf-8', index=False)
    logging.info(f"Processed {date_string}_{cruise}. Data saved as {output_file}")

if __name__ == '__main__':
    main()
