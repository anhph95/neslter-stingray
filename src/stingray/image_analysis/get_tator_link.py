#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import tator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-in", required=True, help="Input CSV from extract_media_timestamps.py")
    parser.add_argument("--csv-out", required=True, help="Output CSV with Tator links")
    parser.add_argument("--host", required=True)
    parser.add_argument("--project-id", required=True, type=int)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_in, parse_dates=["media_time", "times"])

    api = tator.get_api(host=args.host, token=args.token)
    project_objs = api.get_media_list(args.project_id, dtype="video")
    project_objs = sorted(project_objs, key=lambda p: p.id)

    query = df["media"].dropna().astype(str).unique()

    media_id = [
        [d.name.strip(".avi"), d.id]
        for d in project_objs
        if d.name.strip(".avi") in query
    ]
    media_id = pd.DataFrame(media_id, columns=["media", "id"])

    df = pd.merge(df, media_id, on="media", how="left")

    def make_link(row):
        try:
            if pd.notna(row["id"]) and pd.notna(row["frame"]):
                return f"{args.host}/{args.project_id}/annotation/{int(row['id'])}?frame={int(row['frame'])}"
        except Exception:
            pass
        return np.nan

    df["link"] = df.apply(make_link, axis=1)
    df.to_csv(args.csv_out, index=False)

    print(f"🔗 Added Tator links → {args.csv_out}")

if __name__ == "__main__":
    main()
