"""
Generate synthetic audio samples for testing the Human Audio Gender Recognition pipeline.

Male voices:   fundamental frequency  85–180 Hz  (lower pitch)
Female voices: fundamental frequency 165–255 Hz  (higher pitch)

Each sample: 3 seconds, 22 050 Hz, mono WAV.
"""

import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config


def voiced_signal(
    duration: float,
    sr: int,
    f0: float,
    n_harmonics: int = 8,
    jitter: float = 0.005,
    shimmer: float = 0.05,
    noise_level: float = 0.02,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Synthesise a voiced speech-like signal using additive harmonic synthesis.

    Parameters
    ----------
    duration     : clip length in seconds
    sr           : sample rate
    f0           : fundamental frequency (Hz)
    n_harmonics  : number of harmonics to sum
    jitter       : relative pitch perturbation (cycle-to-cycle variability)
    shimmer      : amplitude perturbation (cycle-to-cycle)
    noise_level  : additive white-noise amplitude
    rng          : numpy random Generator (for reproducibility)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Pitch with micro-jitter
    f0_jittered = f0 * (1.0 + jitter * rng.standard_normal(n_samples))
    phase = 2.0 * np.pi * np.cumsum(f0_jittered) / sr

    # Sum harmonics with amplitude roll-off
    signal = np.zeros(n_samples)
    for k in range(1, n_harmonics + 1):
        amplitude = (1.0 / k) * (1.0 + shimmer * rng.standard_normal())
        signal += amplitude * np.sin(k * phase)

    # Normalise + add noise
    signal /= np.max(np.abs(signal) + 1e-9)
    signal += noise_level * rng.standard_normal(n_samples)

    # Simple amplitude envelope (fade-in / fade-out)
    fade = int(0.05 * sr)          # 50 ms
    signal[:fade]  *= np.linspace(0, 1, fade)
    signal[-fade:] *= np.linspace(1, 0, fade)

    # Normalise to ±0.9 and clip
    peak = np.max(np.abs(signal)) + 1e-9
    signal = np.clip(signal / peak * 0.9, -1.0, 1.0)
    return signal


def generate_dataset(
    n_male: int = 50,
    n_female: int = 50,
    output_dir: Path = config.DATA_RAW_DIR,
    sr: int = config.SAMPLE_RATE,
    duration: float = config.DURATION,
    seed: int = config.RANDOM_STATE,
) -> None:
    """
    Generate synthetic WAV files and save them under
    ``output_dir/male/``  and  ``output_dir/female/``.
    """
    rng = np.random.default_rng(seed)

    # Frequency ranges (Hz)
    male_f0_range   = (85,  180)
    female_f0_range = (165, 255)

    class_specs = [
        ("male",   n_male,   male_f0_range),
        ("female", n_female, female_f0_range),
    ]

    print(f"\n{'─'*60}")
    print("  Generating Synthetic Audio Dataset")
    print(f"{'─'*60}")
    print(f"  Output directory : {output_dir}")
    print(f"  Sample rate      : {sr} Hz")
    print(f"  Duration         : {duration} s")
    print(f"  Male samples     : {n_male}  (f0 {male_f0_range[0]}–{male_f0_range[1]} Hz)")
    print(f"  Female samples   : {n_female}  (f0 {female_f0_range[0]}–{female_f0_range[1]} Hz)")
    print(f"{'─'*60}\n")

    for gender, n, (f0_lo, f0_hi) in class_specs:
        class_dir = output_dir / gender
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            f0          = rng.uniform(f0_lo, f0_hi)
            n_harmonics = int(rng.integers(6, 12))
            jitter      = rng.uniform(0.002, 0.010)
            shimmer     = rng.uniform(0.02,  0.08)
            noise       = rng.uniform(0.01,  0.04)

            audio = voiced_signal(
                duration=duration,
                sr=sr,
                f0=f0,
                n_harmonics=n_harmonics,
                jitter=jitter,
                shimmer=shimmer,
                noise_level=noise,
                rng=rng,
            )

            # Convert to int16 for WAV
            audio_int16 = (audio * 32767).astype(np.int16)
            filename = class_dir / f"{gender}_{i:03d}.wav"
            wavfile.write(str(filename), sr, audio_int16)

            print(f"  [{gender:6s}] {i+1:3d}/{n}  f0={f0:6.1f} Hz  → {filename.name}")

    print(f"\n  Done.  Total files: {n_male + n_female}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    generate_dataset()
