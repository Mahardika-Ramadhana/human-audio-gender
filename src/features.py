"""
Feature extraction for Human Audio Gender Recognition.

Provides:
- Individual feature extractors (MFCC, Chroma, Mel-spec, etc.)
- extract_all_features()   — 1-D feature vector for traditional ML
- extract_cnn_features()   — 2-D mel spectrogram for CNN input
- FeatureExtractor class   — batch processing with progress bar
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import librosa
from tqdm import tqdm

from src import config
from src.data_loader import DataLoader, load_audio

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Individual feature extractors
# ─────────────────────────────────────────────────────────────────────────────

def extract_mfcc(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    n_mfcc: int = config.N_MFCC,
) -> np.ndarray:
    """
    Mel-Frequency Cepstral Coefficients.

    Returns mean and standard deviation across time → 2 * n_mfcc features.
    Default: 80 features.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)], axis=0)


def extract_chroma(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
) -> np.ndarray:
    """
    Chroma feature (pitch class profile).

    Returns mean + std across time → 24 features.
    """
    stft = np.abs(librosa.stft(audio, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr, n_chroma=config.N_CHROMA)
    return np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)], axis=0)


def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    n_mels: int = config.N_MELS,
) -> np.ndarray:
    """
    Mel-scale power spectrogram (log-scaled).

    Returns mean + std per mel band → 2 * n_mels features.
    Default: 256 features.
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.concatenate([mel_db.mean(axis=1), mel_db.std(axis=1)], axis=0)


def extract_spectral_contrast(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
) -> np.ndarray:
    """
    Spectral contrast (valley vs. peak energy per sub-band).

    librosa returns (n_bands + 1, frames); default n_bands = 6
    → 7 bands × 2 (mean + std) = 14 features.
    """
    contrast = librosa.feature.spectral_contrast(
        y=audio, sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_bands=config.N_SPECTRAL_CONTRAST_BANDS,
    )
    return np.concatenate([contrast.mean(axis=1), contrast.std(axis=1)], axis=0)


def extract_zcr(audio: np.ndarray) -> np.ndarray:
    """
    Zero-Crossing Rate.

    Returns mean + std → 2 features.
    """
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=config.HOP_LENGTH)
    return np.array([zcr.mean(), zcr.std()], dtype=np.float32)


def extract_rms(audio: np.ndarray) -> np.ndarray:
    """
    Root Mean Square energy.

    Returns mean + std → 2 features.
    """
    rms = librosa.feature.rms(y=audio, hop_length=config.HOP_LENGTH)
    return np.array([rms.mean(), rms.std()], dtype=np.float32)


def extract_spectral_rolloff(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
) -> np.ndarray:
    """
    Spectral roll-off frequency (85th-percentile).

    Returns mean + std → 2 features.
    """
    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        roll_percent=0.85,
    )
    return np.array([rolloff.mean(), rolloff.std()], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Composite feature vector for traditional ML
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    cfg=config,
) -> np.ndarray:
    """
    Concatenate all hand-crafted features into a single 1-D vector.

    Feature breakdown (defaults):
      MFCC              80
      Chroma            24
      Mel spectrogram  256
      Spectral contrast 14
      ZCR                2
      RMS                2
      Spectral rolloff   2
      ─────────────────────
      Total            380

    Parameters
    ----------
    audio : np.ndarray
        Mono waveform, float32.
    sr : int
        Sample rate.
    cfg : module
        Config module (allows easy mocking in tests).

    Returns
    -------
    features : np.ndarray, shape (380,)
    """
    parts = [
        extract_mfcc(audio, sr, cfg.N_MFCC),
        extract_chroma(audio, sr),
        extract_mel_spectrogram(audio, sr, cfg.N_MELS),
        extract_spectral_contrast(audio, sr),
        extract_zcr(audio),
        extract_rms(audio),
        extract_spectral_rolloff(audio, sr),
    ]
    return np.concatenate(parts, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2-D mel spectrogram for CNN input
# ─────────────────────────────────────────────────────────────────────────────

def extract_cnn_features(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    cfg=config,
) -> np.ndarray:
    """
    Compute a fixed-size log-mel spectrogram suitable for CNN input.

    The time axis is padded or trimmed to ``cfg.CNN_TIME_FRAMES`` so every
    sample has identical shape ``(N_MELS, CNN_TIME_FRAMES)``.

    Parameters
    ----------
    audio : np.ndarray
        Mono waveform, float32.
    sr : int
        Sample rate.
    cfg : module
        Config module.

    Returns
    -------
    mel_db : np.ndarray, shape (N_MELS, CNN_TIME_FRAMES)
        Log-power mel spectrogram.
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=cfg.N_MELS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    target_frames = cfg.CNN_TIME_FRAMES
    current_frames = mel_db.shape[1]

    if current_frames < target_frames:
        pad = target_frames - current_frames
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='constant', constant_values=mel_db.min())
    else:
        mel_db = mel_db[:, :target_frames]

    return mel_db  # shape: (N_MELS, CNN_TIME_FRAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extract features from an entire dataset split and optionally persist
    the results to disk.

    Usage
    -----
    extractor = FeatureExtractor()
    extractor.extract_and_save(train_data, config.FEATURES_TRAIN_PATH)
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        duration: float = config.DURATION,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self._loader = DataLoader(sample_rate=sample_rate, duration=duration)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_one(
        self,
        file_path: str,
        mode: str,
    ) -> Optional[np.ndarray]:
        """
        Load audio and extract features for a single file.

        Parameters
        ----------
        file_path : str
        mode : str
            'flat' for traditional ML, 'cnn' for 2-D spectrogram.
        """
        try:
            audio = self._loader.load_single(file_path)
            if mode == 'flat':
                return extract_all_features(audio, self.sample_rate)
            elif mode == 'cnn':
                return extract_cnn_features(audio, self.sample_rate)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        except Exception as exc:
            logger.warning("Skipping '%s': %s", file_path, exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_and_save(
        self,
        data: List[Tuple[str, int]],
        output_path: str,
        mode: str = 'flat',
        desc: str = 'Extracting features',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features for every (file_path, label) in *data*, save to
        *output_path*, and return (features, labels).

        Parameters
        ----------
        data : list of (str, int)
        output_path : str or Path
        mode : str
            'flat' → 1-D vector for traditional ML
            'cnn'  → 2-D mel spectrogram for CNN
        desc : str
            tqdm progress-bar description.
        """
        feature_list: List[np.ndarray] = []
        label_list:   List[int]        = []

        for file_path, label in tqdm(data, desc=desc, unit='file'):
            feats = self._process_one(file_path, mode)
            if feats is not None:
                feature_list.append(feats)
                label_list.append(label)

        features = np.stack(feature_list, axis=0)
        labels   = np.array(label_list, dtype=np.int32)

        # Persist
        loader = DataLoader()
        loader.save_features(features, labels, output_path)

        logger.info(
            "Saved %d samples → %s  [feature shape per sample: %s]",
            len(labels), output_path, features[0].shape,
        )
        return features, labels

    def extract_split(
        self,
        data: List[Tuple[str, int]],
        mode: str = 'flat',
        desc: str = 'Extracting features',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Like :meth:`extract_and_save` but without saving to disk.
        """
        feature_list: List[np.ndarray] = []
        label_list:   List[int]        = []

        for file_path, label in tqdm(data, desc=desc, unit='file'):
            feats = self._process_one(file_path, mode)
            if feats is not None:
                feature_list.append(feats)
                label_list.append(label)

        features = np.stack(feature_list, axis=0)
        labels   = np.array(label_list, dtype=np.int32)
        return features, labels
