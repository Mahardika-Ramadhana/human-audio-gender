"""
Data loading utilities for Human Audio Gender Recognition.

Handles:
- Loading and normalising raw audio files
- Scanning directory structures
- Stratified train/val/test splitting
- Persisting and restoring feature arrays
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import librosa
from sklearn.model_selection import train_test_split

from src import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level audio I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(
    file_path: str,
    sr: int = config.SAMPLE_RATE,
    duration: float = config.DURATION,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, resample to *sr*, and pad / trim to exactly
    ``sr * duration`` samples.

    Parameters
    ----------
    file_path : str or Path
        Path to the audio file (any format supported by librosa).
    sr : int
        Target sample rate.
    duration : float
        Target duration in seconds.

    Returns
    -------
    audio : np.ndarray, shape (sr * duration,)
        Mono, float32 audio waveform.
    sr : int
        Sample rate used.
    """
    file_path = str(file_path)
    target_length = int(sr * duration)

    try:
        audio, loaded_sr = librosa.load(file_path, sr=sr, mono=True, duration=duration)
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio file '{file_path}': {exc}") from exc

    # Pad if shorter than target
    if len(audio) < target_length:
        pad_width = target_length - len(audio)
        audio = np.pad(audio, (0, pad_width), mode='constant', constant_values=0.0)
    else:
        audio = audio[:target_length]

    return audio.astype(np.float32), sr


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader class
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    End-to-end data management: directory scanning, splitting, and feature
    array persistence.

    Expected directory layout
    -------------------------
    data_dir/
        male/
            file1.wav
            file2.wav
            ...
        female/
            file1.wav
            ...
    """

    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        duration: float = config.DURATION,
        random_state: int = config.RANDOM_STATE,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Directory scanning
    # ------------------------------------------------------------------

    def load_dataset(
        self, data_dir: str
    ) -> List[Tuple[str, int]]:
        """
        Scan *data_dir* for class subdirectories and return a list of
        ``(file_path, label_index)`` tuples.

        Parameters
        ----------
        data_dir : str or Path
            Root directory containing one subdirectory per class.

        Returns
        -------
        data : list of (str, int)
            Each element is ``(absolute_file_path, class_index)``.
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        data: List[Tuple[str, int]] = []

        for class_name, class_idx in config.CLASS_TO_IDX.items():
            class_dir = data_dir / class_name
            if not class_dir.is_dir():
                logger.warning("Class directory not found: %s — skipping.", class_dir)
                continue

            files_found = 0
            for file_path in sorted(class_dir.iterdir()):
                if file_path.suffix.lower() in self.AUDIO_EXTENSIONS:
                    data.append((str(file_path), class_idx))
                    files_found += 1

            logger.info("Class '%s': %d files found.", class_name, files_found)

        if not data:
            raise ValueError(
                f"No audio files found under '{data_dir}'. "
                "Ensure subdirectories named 'male' and 'female' exist."
            )

        logger.info("Total files loaded: %d", len(data))
        return data

    # ------------------------------------------------------------------
    # Stratified splitting
    # ------------------------------------------------------------------

    def split_dataset(
        self,
        data: List[Tuple[str, int]],
        test_size: float = config.TEST_SIZE,
        val_size: float = config.VAL_SIZE,
    ) -> Tuple[
        List[Tuple[str, int]],
        List[Tuple[str, int]],
        List[Tuple[str, int]],
    ]:
        """
        Stratified train / val / test split.

        Parameters
        ----------
        data : list of (path, label)
        test_size : float
            Fraction of the full dataset to use for testing.
        val_size : float
            Fraction of the *remaining* data (after removing test) to
            use for validation.

        Returns
        -------
        train_data, val_data, test_data : lists of (path, label)
        """
        paths  = [d[0] for d in data]
        labels = [d[1] for d in data]

        # First split: hold out test set
        paths_trainval, paths_test, labels_trainval, labels_test = train_test_split(
            paths,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=self.random_state,
        )

        # Second split: carve out validation from train
        # val_size is relative to original; adjust to be relative to trainval
        val_fraction = val_size / (1.0 - test_size)
        paths_train, paths_val, labels_train, labels_val = train_test_split(
            paths_trainval,
            labels_trainval,
            test_size=val_fraction,
            stratify=labels_trainval,
            random_state=self.random_state,
        )

        train_data = list(zip(paths_train, labels_train))
        val_data   = list(zip(paths_val,   labels_val))
        test_data  = list(zip(paths_test,  labels_test))

        logger.info(
            "Split — train: %d  val: %d  test: %d",
            len(train_data), len(val_data), len(test_data),
        )
        return train_data, val_data, test_data

    # ------------------------------------------------------------------
    # Feature persistence
    # ------------------------------------------------------------------

    def save_features(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        output_path: str,
    ) -> None:
        """
        Save feature matrix and label vector as a compressed .npz file.

        Parameters
        ----------
        features : np.ndarray, shape (n_samples, n_features) or (n_samples, H, W)
        labels   : np.ndarray, shape (n_samples,)
        output_path : str or Path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(output_path), features=features, labels=labels)
        logger.info("Features saved to %s  [shape=%s]", output_path, features.shape)

    def load_features(
        self, input_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load features and labels from a .npz file previously saved with
        :meth:`save_features`.

        Returns
        -------
        features : np.ndarray
        labels   : np.ndarray
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Feature file not found: {input_path}")

        data = np.load(str(input_path), allow_pickle=False)
        features = data['features']
        labels   = data['labels']
        logger.info("Features loaded from %s  [shape=%s]", input_path, features.shape)
        return features, labels

    # ------------------------------------------------------------------
    # Convenience: load a single audio file
    # ------------------------------------------------------------------

    def load_single(self, file_path: str) -> np.ndarray:
        """Return a normalised audio waveform for *file_path*."""
        audio, _ = load_audio(file_path, sr=self.sample_rate, duration=self.duration)
        return audio
