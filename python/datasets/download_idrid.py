"""
download_idrid.py
=================
Automated downloader for the IDRiD (Indian Diabetic Retinopathy Image Dataset)
Part A - Segmentation Challenge dataset from the official/research-accessible Hugging Face mirror:
https://huggingface.co/datasets/MahsaTorki/IDRiD_Dataset

Downloads A.Segmentation.zip (557 MB) with chunked streaming and extracts to data/idrid/.
"""

import os
import sys
import time
import zipfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "idrid"
ZIP_URL = "https://huggingface.co/datasets/MahsaTorki/IDRiD_Dataset/resolve/main/A.Segmentation.zip"
ZIP_PATH = DATA_DIR / "A.Segmentation.zip"


def download_file(url: str, dest_path: Path, chunk_size: int = 2 * 1024 * 1024):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    existing_bytes = 0
    if dest_path.exists():
        existing_bytes = dest_path.stat().st_size

    req = urllib.request.Request(url, headers={"User-Agent": "RetinaGuard-Downloader/1.0"})
    
    with urllib.request.urlopen(req) as resp:
        total_bytes = int(resp.headers.get("Content-Length", 0))

    if existing_bytes == total_bytes and total_bytes > 0:
        print(f"[OK] File already fully downloaded: {dest_path} ({total_bytes / (1024*1024):.2f} MB)")
        return

    print(f"[*] Starting download from: {url}")
    print(f"[*] Target destination   : {dest_path}")
    print(f"[*] Total file size      : {total_bytes / (1024*1024):.2f} MB")

    resume_header = {}
    mode = "wb"
    downloaded = 0
    if 0 < existing_bytes < total_bytes:
        resume_header = {"Range": f"bytes={existing_bytes}-"}
        mode = "ab"
        downloaded = existing_bytes
        print(f"[*] Resuming from byte {existing_bytes} ({existing_bytes / (1024*1024):.2f} MB)")

    req = urllib.request.Request(url, headers={"User-Agent": "RetinaGuard-Downloader/1.0", **resume_header})
    
    start_time = time.time()
    last_print = start_time

    with urllib.request.urlopen(req) as resp, open(dest_path, mode) as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_print >= 2.0 or downloaded == total_bytes:
                pct = (downloaded / total_bytes) * 100 if total_bytes > 0 else 0
                elapsed = now - start_time
                speed = (downloaded - existing_bytes) / (1024 * 1024 * max(elapsed, 0.001))
                print(f"    Progress: {downloaded / (1024*1024):.1f} / {total_bytes / (1024*1024):.1f} MB "
                      f"({pct:.1f}%) — {speed:.2f} MB/s")
                last_print = now

    print(f"[OK] Download completed successfully in {time.time() - start_time:.1f}s.")


def extract_zip(zip_path: Path, target_dir: Path):
    print(f"[*] Extracting {zip_path.name} to {target_dir} ...")
    start_time = time.time()
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        members = z.namelist()
        total_members = len(members)
        print(f"[*] Total files in archive: {total_members}")
        z.extractall(target_dir)
        
    print(f"[OK] Extraction completed in {time.time() - start_time:.1f}s.")


def main():
    print("=" * 60)
    print("  RETINAGUARD — IDRiD DATASET ACQUISITION")
    print("=" * 60)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download zip archive
    download_file(ZIP_URL, ZIP_PATH)
    
    # Step 2: Extract archive
    extract_zip(ZIP_PATH, DATA_DIR)
    
    print("\n[OK] IDRiD dataset acquisition complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
