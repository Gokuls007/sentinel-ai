"""Download CAUCAFall (videos + per-frame fall labels, no PNG frames) from Mendeley Data.

    python training/fetch_caucafall.py              # about 136 MB into data/datasets/caucafall/

Dataset: Eraso, Muñoz, Muñoz & Fernández, "Dataset CAUCAFall", Mendeley Data, V4,
doi:10.17632/7w7fccy7ky.4. Licence: CC BY 4.0. 10 subjects x 10 activities: five falls
(forward, backward, left, right, from sitting) and five daily activities (walk, hop, pick up
an object, sit down, kneel). Recorded at 23 fps.

Each activity folder has one ``.avi`` and one YOLO label file per frame (``class cx cy w h``,
class 0 = no fall, 1 = fall), plus ``classes.txt``. The ~20,000 PNG frames (8.2 GB) are
skipped. After downloading, every video gets a compact ``labels.csv`` (frame, label).

Downloads resume: files already present with the right size are skipped.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "datasets", "caucafall")
API = "https://data.mendeley.com/public-api/datasets/7w7fccy7ky"
VERSION = 4
HEADERS = {"User-Agent": "sentinel-ai-eval/1.0", "Accept": "application/json"}


def get_json(url: str):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                return json.load(r)
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def folder_paths(folders: list[dict]) -> dict[str, str]:
    by_id = {f["id"]: f for f in folders}

    def path(fid):
        parts = []
        while fid in by_id:
            parts.append(by_id[fid]["name"])
            fid = by_id[fid].get("parent_id")
        return "/".join(reversed(parts))

    return {fid: path(fid) for fid in by_id}


def wanted(name: str) -> bool:
    return name.lower().endswith((".avi", ".txt", ".xlsx"))


def download(url: str, dest: str, size: int) -> bool:
    if os.path.isfile(dest) and os.path.getsize(dest) == size:
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"]}),
                                        timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            os.replace(tmp, dest)
            return True
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    return False


def write_label_csvs(base: str) -> int:
    """data/.../<Subject>/<Activity>/labels.csv from the per-frame YOLO files (frames in name order)."""
    n = 0
    for dirpath, _dirs, files in os.walk(base):
        txts = sorted(f for f in files if f.endswith(".txt") and f != "classes.txt")
        if not txts or not any(f.endswith(".avi") for f in files):
            continue
        rows = []
        for i, name in enumerate(txts, start=1):
            with open(os.path.join(dirpath, name), encoding="utf-8", errors="replace") as fh:
                first = fh.read().split()
            rows.append(f"{i},{int(first[0]) if first else -1}")
        with open(os.path.join(dirpath, "labels.csv"), "w", encoding="utf-8") as fh:
            fh.write("frame,label\n" + "\n".join(rows) + "\n")
        n += 1
    return n


def main() -> int:
    print("listing CAUCAFall files ...", flush=True)
    folders = get_json(f"{API}/folders/{VERSION}")
    paths = folder_paths(folders)
    jobs = []
    for fid in ["root", *paths]:
        for f in get_json(f"{API}/files?folder_id={fid}&version={VERSION}"):
            if wanted(f["filename"]):
                rel = os.path.join(paths.get(fid, ""), f["filename"])
                jobs.append((f["content_details"]["download_url"], os.path.join(DATA_DIR, rel), f["size"]))
    total = sum(s for _u, _d, s in jobs)
    print(f"{len(jobs)} files, {total / 1e6:.0f} MB", flush=True)
    done = fetched = 0
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, *job) for job in jobs]
        for fut in cf.as_completed(futures):
            fetched += fut.result()
            done += 1
            if done % 1000 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} ({fetched} downloaded)", flush=True)
    videos = write_label_csvs(DATA_DIR)
    print(f"CAUCAFall ready in {DATA_DIR}: {videos} videos with labels.csv")
    return 0 if videos else 1


if __name__ == "__main__":
    sys.exit(main())
