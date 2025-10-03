import argparse
import os
import pandas as pd

def download_data(url: str, sep: str, save_path: str = "data/raw/data.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.read_csv(url, sep=sep)
    df.to_csv(save_path, index=False)
    print(f"[ingest] Saved raw data to {save_path} (rows={len(df)})")
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="CSV URL or local path")
    parser.add_argument("--sep", type=str, default=";", help="CSV separator")
    parser.add_argument("--save_path", type=str, default="data/raw/winequality-red.csv")
    args = parser.parse_args()

    if args.url.startswith("http"):
        download_data(args.url, args.sep, args.save_path)
    else:
        # Local copy
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        df = pd.read_csv(args.url, sep=args.sep)
        df.to_csv(args.save_path, index=False)
        print(f"[ingest] Copied local file to {args.save_path} (rows={len(df)})")
