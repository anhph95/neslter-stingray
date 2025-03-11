import os
import argparse
import pandas as pd
import pathlib
from utils import *

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cruise', type=str, help='Cruise ID')
    parser.add_argument('--sensor_dir', type=str, help='Path to the sensor data', default='rawdata')
    parser.add_argument('--media_dir', type=str, help='Path to the media data', default='medialist')
    # Handle lists properly using nargs
    parser.add_argument('--bin_cols', type=str, nargs='+', default=['matdate', 'depth'],
                        help='Columns to bin (space-separated list, e.g., "matdate depth")')
    parser.add_argument('--bin_steps', type=float, nargs='+', default=[30/86400, 1],
                        help='Steps to bin (space-separated list, e.g., "0.000347 1" [30 seconds/86400, 1 meter])')
    args = parser.parse_args()
    return args

def main():
    args = cli()
    cruise = args.cruise
    sensor_dir = args.sensor_dir
    media_dir = args.media_dir
    cols = args.bin_cols
    steps = args.bin_steps
    pathlib.Path('dash_data').mkdir(parents=True, exist_ok=True)
    
    # Find matching sensor file
    sensor_file = next((file for file in os.listdir(sensor_dir) if file.endswith('.csv') and cruise in file), None)
    if sensor_file:
        sensor_df = pd.read_csv(os.path.join(sensor_dir, sensor_file))
        sensor_df['times'] = pd.to_datetime(sensor_df['times'])

    # Find matching media file
    media_file = next((file for file in os.listdir(media_dir) if file.endswith('.csv') and cruise in file), None)
    if media_file:
        media_df = pd.read_csv(os.path.join(media_dir, media_file))
        media_df['times'] = pd.to_datetime(media_df['times'])
        media_df.drop(columns=['id'], inplace=True)
        df = merge_df(sensor_df, media_df, on='times',cols=media_df.columns[1:].to_list())
    else:
        print(f'No media file found for {cruise}')
        df = sensor_df    
    
    df_bin = bin_data(df,cols,steps)
    df_bin['group'] = pd.factorize(pd.MultiIndex.from_arrays([df_bin[f'{col}_bin'] for col in cols]))[0]

    sensor_cols = sensor_df.columns[3:].to_list()
    meta_cols = [col for col in df.columns if col not in sensor_cols]     
        
    df_mean = df_bin.groupby('group').agg(
        {col: 'first' for col in meta_cols} |
        {col: ['mean','std'] for col in sensor_cols}).reset_index(drop=True)
    df_mean.columns = [
        col[0] if col[1] in ['first', 'mean'] else f"{col[0]}_{col[1]}" for col in df_mean.columns
    ]
    
    date_string = df_mean['times'][0].strftime('%Y%m%d')
    df_mean.to_csv(f'dash_data/{date_string}_{cruise}.csv',encoding='utf-8', index=False)
    print(f'Processed {date_string}_{cruise}')
    
if __name__ == '__main__':
    main()
