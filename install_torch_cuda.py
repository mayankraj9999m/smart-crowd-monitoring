"""
install_torch_cuda.py
Resumable downloader & installer for PyTorch with CUDA 12.4 for RTX 2050 GPU.
"""

import os
import sys
import time
import urllib.request
import subprocess

WHEELS_DIR = "wheels_cache"

PACKAGES = [
    {
        "name": "torch-2.6.0+cu124-cp312-cp312-win_amd64.whl",
        "url": "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl"
    },
    {
        "name": "torchvision-0.21.0+cu124-cp312-cp312-win_amd64.whl",
        "url": "https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp312-cp312-win_amd64.whl"
    }
]


def download_resumable(url, dest_path):
    # Check total size via HEAD request
    head_req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="HEAD")
    with urllib.request.urlopen(head_req) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))

    downloaded = 0
    mode = "wb"
    headers = {"User-Agent": "Mozilla/5.0"}

    if os.path.exists(dest_path):
        existing_size = os.path.getsize(dest_path)
        if existing_size >= total_size and total_size > 0:
            print(f"[Cache] Complete file already exists: {dest_path} ({existing_size / (1024*1024):.1f} MB)", flush=True)
            return
        elif existing_size > 0:
            downloaded = existing_size
            mode = "ab"
            headers["Range"] = f"bytes={existing_size}-"
            print(f"[Resume] Resuming {os.path.basename(dest_path)} from {existing_size / (1024*1024):.1f} MB (Total: {total_size / (1024*1024):.1f} MB)...", flush=True)

    if downloaded == 0:
        print(f"[Download] Starting download of {os.path.basename(dest_path)} ({total_size / (1024*1024):.1f} MB)...", flush=True)

    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    last_print = t0

    with urllib.request.urlopen(req) as resp, open(dest_path, mode) as out_f:
        chunk_size = 8 * 1024 * 1024  # 8 MB
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_print >= 4.0 or downloaded >= total_size:
                elapsed = max(0.1, now - t0)
                speed_mb = ((downloaded - (existing_size if mode == "ab" else 0)) / elapsed) / (1024 * 1024)
                pct = (downloaded / total_size) * 100 if total_size else 0
                rem_bytes = max(0, total_size - downloaded)
                eta_s = rem_bytes / (speed_mb * 1024 * 1024) if speed_mb > 0 else 0
                eta_m, eta_sec = divmod(int(eta_s), 60)
                print(f"[Progress] {downloaded/(1024*1024):.1f} / {total_size/(1024*1024):.1f} MB ({pct:.1f}%) | "
                      f"Speed: {speed_mb:.2f} MB/s | ETA: {eta_m:02d}m:{eta_sec:02d}s", flush=True)
                last_print = now

    print(f"--> Download complete: {dest_path}", flush=True)


def main():
    os.makedirs(WHEELS_DIR, exist_ok=True)
    wheel_paths = []
    
    for pkg in PACKAGES:
        dest = os.path.join(WHEELS_DIR, pkg["name"])
        download_resumable(pkg["url"], dest)
        wheel_paths.append(dest)
        
    print("\nInstalling PyTorch with CUDA support into .venv...", flush=True)
    cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps"] + wheel_paths
    res = subprocess.run(cmd)
    
    if res.returncode == 0:
        print("\n" + "=" * 60, flush=True)
        print("PYTORCH GPU VERIFICATION:", flush=True)
        import torch
        print("PyTorch Version :", torch.__version__, flush=True)
        print("CUDA Available  :", torch.cuda.is_available(), flush=True)
        if torch.cuda.is_available():
            print("GPU Device Name :", torch.cuda.get_device_name(0), flush=True)
            print("VRAM (Memory)   :", f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)
            print("🚀 RTX 2050 is now fully active for training!", flush=True)
        else:
            print("CUDA is still reporting False. Check NVIDIA drivers.", flush=True)
        print("=" * 60 + "\n", flush=True)
    else:
        print("Pip install failed with return code:", res.returncode, flush=True)


if __name__ == "__main__":
    main()
