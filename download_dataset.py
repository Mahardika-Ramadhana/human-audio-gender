"""
Download and prepare the RAVDESS dataset for gender recognition.

RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song
  DOI  : https://doi.org/10.5281/zenodo.1188976
  Size : ~215 MB (Speech Audio only)
  URL  : https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip

Gender convention (standard across literature):
  Odd  actor IDs (01,03,05,...,23) → male
  Even actor IDs (02,04,06,...,24) → female

After running this script the directory layout will be:
  data/raw/
    male/
      Actor_01_03-01-01-01-01-01-01.wav
      ...
    female/
      Actor_02_03-01-01-01-01-01-02.wav
      ...
"""

import io
import os
import sys
import zipfile
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

# ── project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
RAVDESS_URL = (
    "https://zenodo.org/record/1188976/files/"
    "Audio_Speech_Actors_01-24.zip"
)
RAVDESS_ZIP  = config.PROJECT_ROOT / "data" / "Audio_Speech_Actors_01-24.zip"
RAVDESS_EXTRACT = config.PROJECT_ROOT / "data" / "ravdess_raw"


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, chunk_size: int = 1024 * 64) -> None:
    """Stream-download *url* to *dest* with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    desc  = dest.name

    with open(dest, "wb") as fh, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"  Downloading {desc}",
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                bar.update(len(chunk))

    print(f"  Saved → {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# RAVDESS filename parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_ravdess_actor(filename: str) -> int:
    """
    Extract the actor ID from a RAVDESS filename.

    RAVDESS filename format:
      Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
      e.g. 03-01-01-01-01-01-07.wav  →  Actor 07

    Parameters
    ----------
    filename : str  (basename, e.g. '03-01-03-01-01-01-07.wav')

    Returns
    -------
    actor_id : int (1–24)
    """
    stem = Path(filename).stem            # e.g. '03-01-03-01-01-01-07'
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected RAVDESS filename format: {filename}")
    return int(parts[6])


def actor_to_gender(actor_id: int) -> str:
    """
    Map actor ID to gender label.

    Convention (widely used in literature):
      Odd  actor IDs → male
      Even actor IDs → female

    Reference: Livingstone & Russo (2018), PLOS ONE,
               DOI: 10.1371/journal.pone.0196391
    """
    return "male" if actor_id % 2 == 1 else "female"


# ─────────────────────────────────────────────────────────────────────────────
# Organise into class directories
# ─────────────────────────────────────────────────────────────────────────────

def organise_dataset(
    extracted_dir: Path,
    output_dir: Path,
) -> dict:
    """
    Walk *extracted_dir* (the unzipped RAVDESS tree), infer gender from
    each filename, and copy the file to
    ``output_dir/male/`` or ``output_dir/female/``.

    Returns
    -------
    stats : dict  {'male': int, 'female': int}
    """
    male_dir   = output_dir / "male"
    female_dir = output_dir / "female"
    male_dir.mkdir(parents=True, exist_ok=True)
    female_dir.mkdir(parents=True, exist_ok=True)

    stats = {"male": 0, "female": 0}
    wav_files = sorted(extracted_dir.rglob("*.wav"))

    print(f"\n  Organising {len(wav_files)} WAV files …")

    for wav_path in tqdm(wav_files, desc="  Organising", unit="file"):
        try:
            actor_id = parse_ravdess_actor(wav_path.name)
            gender   = actor_to_gender(actor_id)
        except ValueError as exc:
            print(f"  [warn] {exc} — skipping {wav_path.name}")
            continue

        dest_dir  = male_dir if gender == "male" else female_dir
        # Prefix with actor folder to avoid name collisions
        dest_name = f"Actor_{actor_id:02d}_{wav_path.name}"
        dest_path = dest_dir / dest_name

        shutil.copy2(str(wav_path), str(dest_path))
        stats[gender] += 1

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def download_ravdess(
    force_download: bool = False,
    output_dir: Path = config.DATA_RAW_DIR,
) -> None:
    """
    Full download-and-prepare workflow for RAVDESS.

    Steps
    -----
    1. Download the Speech ZIP from Zenodo (skip if already present).
    2. Extract the ZIP.
    3. Reorganise files into ``output_dir/male/`` and ``output_dir/female/``.
    4. Print a summary.
    """
    print("\n" + "─" * 60)
    print("  RAVDESS Dataset Download & Preparation")
    print("  DOI: 10.5281/zenodo.1188976")
    print("─" * 60)

    # ── Check if already organised ────────────────────────────────────────
    male_dir   = output_dir / "male"
    female_dir = output_dir / "female"
    existing_male   = len(list(male_dir.glob("*.wav")))   if male_dir.exists()   else 0
    existing_female = len(list(female_dir.glob("*.wav"))) if female_dir.exists() else 0

    if not force_download and existing_male > 0 and existing_female > 0:
        print(f"\n  Dataset already present:")
        print(f"    male   : {existing_male} files")
        print(f"    female : {existing_female} files")
        print("  Use --force to re-download.\n")
        return

    # ── Step 1: Download ──────────────────────────────────────────────────
    if not RAVDESS_ZIP.exists() or force_download:
        print(f"\n  [1/3] Downloading from Zenodo …")
        print(f"        {RAVDESS_URL}")
        download_file(RAVDESS_URL, RAVDESS_ZIP)
    else:
        print(f"\n  [1/3] ZIP already downloaded: {RAVDESS_ZIP}")

    # ── Step 2: Extract ───────────────────────────────────────────────────
    print(f"\n  [2/3] Extracting ZIP …")
    RAVDESS_EXTRACT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(RAVDESS_ZIP), "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="  Extracting", unit="file"):
            zf.extract(member, str(RAVDESS_EXTRACT))
    print(f"  Extracted to: {RAVDESS_EXTRACT}")

    # ── Step 3: Organise ──────────────────────────────────────────────────
    print(f"\n  [3/3] Organising by gender …")
    stats = organise_dataset(RAVDESS_EXTRACT, output_dir)

    # ── Summary ───────────────────────────────────────────────────────────
    total = stats["male"] + stats["female"]
    print(f"\n{'─'*60}")
    print(f"  Download complete!")
    print(f"  Male samples   : {stats['male']}")
    print(f"  Female samples : {stats['female']}")
    print(f"  Total          : {total}")
    print(f"  Output dir     : {output_dir}")
    print(f"{'─'*60}\n")

    # Clean up extracted folder to save space
    answer = input("  Delete temporary extraction folder? [y/N]: ").strip().lower()
    if answer == "y":
        shutil.rmtree(str(RAVDESS_EXTRACT))
        print(f"  Removed: {RAVDESS_EXTRACT}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download RAVDESS dataset")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if files exist.")
    args = parser.parse_args()
    download_ravdess(force_download=args.force)
