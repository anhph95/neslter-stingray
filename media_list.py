
import os
import argparse
import pandas as pd
import numpy as np
import cv2
import concurrent.futures
import pathlib
from datetime import datetime, timedelta
import tator

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cruise', type=str, help='Cruise ID, required, e.g. EN706')
    parser.add_argument('--host', type=str, help='Tator host IP address, defaul is https://tator.whoi.edu', default='https://tator.whoi.edu')
    parser.add_argument('--project-id', type=int, help='Project ID', default=1)
    parser.add_argument('--token', type=str, help='Tator login token, string or token file')
    parser.add_argument('--media-dir', type=str, help='Path to the media data')
    args = parser.parse_args()
    return args

def list_files(directory):
    """Recursively lists all files in a directory."""
    file_list = []
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_file():
                file_list.append(entry.path)
            elif entry.is_dir():
                file_list.extend(list_files(entry.path))  # Recursively scan subdirectories
    return file_list

def get_file_size(file_path):
    """Returns file path and size in bytes."""
    try:
        return (file_path, os.stat(file_path).st_size)  # os.stat is fast for metadata access
    except FileNotFoundError:
        return (file_path, None)  # Handle race conditions


# Define function to get frame count
def get_frame_count(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def main():
    args = cli()
    CRUISE = args.cruise
    HOST = args.host
    PROJ_ID = args.project_id
    TOKEN = args.token
    media_dir = args.media_dir
    if pathlib.Path(TOKEN).is_file():
        with open(TOKEN, 'r') as f:
            TOKEN = f.read().strip()
    api = tator.get_api(host=HOST, token=TOKEN)
    
    if media_dir is None:
        media_dir = f'/mnt/vast/nes-lter/Stingray/data/NESLTER_{CRUISE}/Basler_avA2300-25gm'

    # Step 1: Get all file paths
    file_paths = list_files(media_dir)
    
    if file_paths == []:
        print(f'No files found in {media_dir}')
        return

    # Step 2: Use Multi-threading to get file sizes faster
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        file_list_with_sizes = list(executor.map(get_file_size, file_paths))

    # Convert to DataFrame
    df = pd.DataFrame(file_list_with_sizes, columns=['media_path', 'media_size'])

    # Get media name
    df['media'] = df['media_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    # Get media time
    df['media_time'] = df['media'].apply(
        lambda x: datetime.strptime(x.split('-')[-1].rstrip('Z'), "%Y%m%dT%H%M%S.%f")
    )


    # Find default media size
    standard_size = df['media_size'].mode()[0]

    # Set default frame count
    df['frame_count'] = get_frame_count(df.loc[df['media_size'] == standard_size, 'media_path'].iloc[0])  # Default all frame counts to 900

    # Identify files with odd sizes
    mask = df['media_size'] != standard_size
    odd_size_files = df.loc[mask, 'media_path']

    # Apply function only to odd-sized files
    df.loc[mask, 'frame_count'] = odd_size_files.apply(get_frame_count)

    # Assume df contains 'File Path' and 'framecount'
    df = df.loc[df.index.repeat(df['frame_count'])]  # Repeat rows based on frame count
    df['frame'] = df.groupby('media_path').cumcount()  # Generate frame numbers

    # Reset index for a clean DataFrame
    df.reset_index(drop=True, inplace=True)

    # Using a safer approach by calculating adjusted timestamps within each group
    adjusted_timestamps = []

    for _, group in df.groupby("media",sort=False):
        base_timestamp = group["media_time"].iloc[0]
        adjusted = group["frame"].apply(lambda frame: base_timestamp + timedelta(seconds=frame/ 15))
        adjusted_timestamps.extend(adjusted)

    # Add the calculated timestamps back to the DataFrame
    df["times"] = adjusted_timestamps
    
    # Get list of video from df
    query = df['media'].dropna().astype(str)
    query = query.unique()
    # Get list of video from Tator
    project_objs = api.get_media_list(PROJ_ID, dtype='video')
    project_objs = sorted(project_objs, key = lambda p: p.id)
    # Get media ID
    media_id = [[d.name.strip('.avi'), d.id] for d in project_objs if d.name.strip('.avi') in query]
    media_id = pd.DataFrame(media_id, columns=['media','id'])
    # Merged df and ID
    df_merged = pd.merge(df[['times','media','frame','media_path']],media_id,on='media',how='left')
    # Generate link
    df_merged['link'] = np.where(
        df_merged['id'].notna() & df_merged['frame'].notna(),
        df_merged['id'].astype('Int64').astype(str).radd(f'{HOST}/{PROJ_ID}/annotation/') +
        '?frame=' + df_merged['frame'].astype('Int64').astype(str),
        np.nan
    )
    
    datestr = df_merged['times'][0].strftime('%Y%m%d')
    df_merged.to_csv(f'medialist/{datestr}_{CRUISE}.csv',index=False)
    
    print(f'{datestr}_{CRUISE}.csv is generated')

if __name__ == '__main__':
    main()