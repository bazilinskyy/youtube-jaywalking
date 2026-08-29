"""
FTP CSV Downloader for Crowd Jaywalking Project

Downloads object detection / pedestrian CSV files from the mobility FTP server
matching video sequences defined in mapping.csv.

Usage:
    python scripts/download_ftp_csvs.py [--secret secret] [--mapping mapping.csv] [--output data/csv]
"""

import argparse
import fnmatch
import json
import logging
import os
import sys
import time
from pathlib import Path
import ftplib
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_secret(secret_path: str) -> dict:
    """Loads FTP credentials from secret file."""
    p = Path(secret_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Secret file '{secret_path}' not found. Please create it with ftp_username and ftp_password."
        )
    with open(p, "r") as f:
        data = json.load(f)
    if not data.get("ftp_username") or not data.get("ftp_password"):
        raise ValueError("Secret file must contain non-empty 'ftp_username' and 'ftp_password'.")
    return data


def connect_ftp(ftp_host: str, username: str, password: str, alias: str, max_retries: int = 5):
    """Helper to connect to FTP with exponential backoff retry."""
    for attempt in range(1, max_retries + 1):
        try:
            ftp = ftplib.FTP(ftp_host, timeout=30)
            ftp.login(username, password)
            ftp.cwd(alias)
            return ftp
        except Exception as e:
            logger.warning(
                f"FTP connection attempt {attempt}/{max_retries} failed ({e}). Retrying in {attempt * 2}s...")
            time.sleep(attempt * 2)
    raise ConnectionError(
        f"Could not connect to FTP host '{ftp_host}' under alias '{alias}' after {max_retries} retries.")


def download_csv_files_from_ftp(
    secret_path: str = "secret",
    mapping_path: str = "experiments/legacy/mapping.csv",
    output_dir: str = "data/csv",
    ftp_host: str = "files.mobility-squad.com",
    aliases: list = None,
):
    """Downloads CSV files from FTP server based on video_id and start_time mapping."""
    if aliases is None:
        aliases = ["data"]

    credentials = load_secret(secret_path)
    username = credentials["ftp_username"]
    password = credentials["ftp_password"]

    logger.info(f"Connecting to FTP server '{ftp_host}' as '{username}'...")
    ftps = {}
    for alias in aliases:
        ftps[alias] = connect_ftp(ftp_host, username, password, alias)
        logger.info(f"Connected and changed directory to '{alias}'.")

    # Load mapping CSV
    if not os.path.exists(mapping_path):
        logger.error(f"Mapping CSV file not found at: {mapping_path}")
        return

    mapping_df = pd.read_csv(mapping_path)
    logger.info(f"Loaded mapping dataset with {len(mapping_df)} rows from '{mapping_path}'.")

    # Fetch remote file listing per alias
    remote_file_cache = {}
    for alias in aliases:
        try:
            remote_file_cache[alias] = ftps[alias].nlst()
            logger.info(f"Fetched {len(remote_file_cache[alias])} remote filenames under '{alias}'.")
        except Exception as e:
            logger.error(f"Failed to list files under alias '{alias}': {e}")
            remote_file_cache[alias] = []

    downloaded_count = 0
    skipped_count = 0
    missing_count = 0

    for idx, row in mapping_df.iterrows():
        videos = row.get("videos")
        start_times = row.get("start_time")

        if pd.isna(videos) or pd.isna(start_times):
            continue

        try:
            v_list = eval(str(videos)) if isinstance(videos, str) and videos.startswith("[") else [videos]
            st_list = eval(str(start_times)) if isinstance(
                start_times, str) and start_times.startswith("[") else [start_times]
        except Exception:
            v_list = [videos]
            st_list = [start_times]

        def flatten(lst):
            res = []
            for item in lst:
                if isinstance(item, list):
                    res.extend(flatten(item))
                else:
                    res.append(item)
            return res

        v_flat = flatten(v_list)
        st_flat = flatten(st_list)

        for v_idx, raw_vid in enumerate(v_flat):
            if str(raw_vid).startswith("#"):
                continue

            if isinstance(raw_vid, (int, float)):
                video_id = f"{int(raw_vid):04d}"
            else:
                video_id = str(raw_vid).zfill(4)

            start_time = st_flat[v_idx] if v_idx < len(st_flat) else st_flat[0]

            # Use wildcard '*' for fps as requested
            file_pattern = f"{video_id}_{start_time}_*.csv"

            for alias in aliases:
                match_found = False
                remote_files = remote_file_cache.get(alias, [])
                alias_output_dir = os.path.join(output_dir, alias)

                for remote_file in remote_files:
                    if fnmatch.fnmatch(remote_file, file_pattern):
                        match_found = True
                        local_file_path = os.path.join(alias_output_dir, remote_file)

                        if os.path.exists(local_file_path) and os.path.getsize(local_file_path) > 0:
                            skipped_count += 1
                        else:
                            logger.info(f"Downloading '{remote_file}' from FTP ('{alias}')...")
                            os.makedirs(alias_output_dir, exist_ok=True)

                            # Attempt download with reconnection safety
                            success = False
                            for attempt in range(1, 4):
                                try:
                                    with open(local_file_path, "wb") as f:
                                        ftps[alias].retrbinary(f"RETR {remote_file}", f.write)
                                    logger.info(f"Successfully downloaded '{remote_file}'.")
                                    downloaded_count += 1
                                    success = True
                                    break
                                except Exception as download_err:
                                    logger.warning(
                                        f"Download failed for '{remote_file}' (attempt {attempt}/3): {download_err}."
                                        " Reconnecting..."
                                    )
                                    try:
                                        ftps[alias] = connect_ftp(ftp_host, username, password, alias)
                                    except Exception as reconnect_err:
                                        logger.error(f"Reconnection failed: {reconnect_err}")

                            if not success and os.path.exists(local_file_path):
                                os.remove(local_file_path)
                        break

                if not match_found:
                    logger.warning(f"File pattern '{file_pattern}' not found on FTP under '{alias}'.")
                    missing_count += 1

    # Close FTP connections
    for alias, ftp in ftps.items():
        try:
            ftp.quit()
        except Exception:
            pass

    logger.info("=" * 60)
    logger.info("FTP DOWNLOAD COMPLETE")
    logger.info(f"Downloaded: {downloaded_count}")
    logger.info(f"Skipped (Already local): {skipped_count}")
    logger.info(f"Missing patterns: {missing_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download object detection CSVs from FTP server")
    parser.add_argument("--secret", type=str, default="secret", help="Path to secret credentials file")
    parser.add_argument("--mapping", type=str, default="experiments/legacy/mapping.csv", help="Path to mapping.csv")
    parser.add_argument("--output", type=str, default="data/csv", help="Output directory for downloaded CSVs")
    args = parser.parse_args()

    download_csv_files_from_ftp(
        secret_path=args.secret,
        mapping_path=args.mapping,
        output_dir=args.output,
    )
