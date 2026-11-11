#!/usr/bin/env python3
"""
Fast Zenodo FlowMol Checkpoint Downloader via 4-Thread HTTP Range Requests.
"""

import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def fetch_range(url: str, start: int, end: int, retries: int = 8) -> bytes:
    expected = end - start + 1
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
                },
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = resp.read()
            if len(data) == expected:
                return data
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Failed range {start}-{end} from {url}")


def download_tar_entry(url: str, tar_offset: int, out_path: str, num_threads: int = 4):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    hdr = fetch_range(url, tar_offset - 512, tar_offset - 1)
    exact_size = int(hdr[124:136].split(b"\x00", 1)[0].strip(), 8)

    if os.path.exists(out_path) and os.path.getsize(out_path) == exact_size:
        # Verify ZIP magic header PK\x03\x04
        with open(out_path, "rb") as f:
            magic = f.read(4)
        if magic == b"PK\x03\x04":
            print(f"Verified complete checkpoint: {out_path} ({exact_size/1e6:.2f} MB)", flush=True)
            return

    t0 = time.time()
    print(f"Downloading {out_path} ({exact_size/1e6:.2f} MB) with {num_threads} threads...", flush=True)
    chunk_sz = 4 * 1024 * 1024
    ranges = []
    for off in range(0, exact_size, chunk_sz):
        c_end = min(off + chunk_sz - 1, exact_size - 1)
        ranges.append((off, tar_offset + off, tar_offset + c_end))

    tmp_path = out_path + ".part"
    with open(tmp_path, "wb") as f:
        f.truncate(exact_size)

    def dl_part(item):
        local_off, b_start, b_end = item
        data = fetch_range(url, b_start, b_end)
        with open(tmp_path, "r+b") as f:
            f.seek(local_off)
            f.write(data)
        return len(data)

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        list(ex.map(dl_part, ranges))

    os.replace(tmp_path, out_path)
    dt = time.time() - t0
    print(
        f"Completed {out_path} ({exact_size/1e6:.2f} MB) in {dt:.1f}s ({exact_size/1e6/dt:.2f} MB/s)",
        flush=True,
    )


if __name__ == "__main__":
    base_dir = "/sjtu-combustion/wjw/chensq/PropMolFlow"
    zenodo_url = "https://zenodo.org/records/15700961/files/checkpoints.tar"
    targets = [
        ("checkpoints/out-of-distribution/gap/epoch=1637-step=640223.ckpt", 653435904),
        ("checkpoints/out-of-distribution/homo/epoch=1468-step=574066.ckpt", 736102912),
        ("checkpoints/out-of-distribution/lumo/epoch=1865-step=729605.ckpt", 818769920),
    ]
    for rel, offset in targets:
        download_tar_entry(zenodo_url, offset, f"{base_dir}/{rel}", num_threads=4)
