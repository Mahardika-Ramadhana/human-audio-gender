"""
Inference utilities for Human Audio Gender Recognition.

Provides:
- AudioPredictor class: load model, predict single file, batch predict
- visualize_prediction: waveform + mel spectrogram with prediction overlay
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display

from src import config
from src.data_loader import DataLoader, load_audio
from src.features import extract_all_features, extract_cnn_features
from src.models import ModelManager

logger = logging.getLogger(__name__)


class AudioPredictor:
    """
    Single-file and batch-file gender prediction from audio.

    Parameters
    ----------
    model_name : str
        One of 'svm', 'random_forest', 'gradient_boosting', 'cnn', 'lstm'.
    model_type : str
        'sklearn' or 'keras'. Inferred from model_name if not provided.
    """

    KERAS_MODELS   = {'cnn', 'lstm', 'gender_cnn', 'gender_lstm'}
    SKLEARN_MODELS = {'svm', 'random_forest', 'gradient_boosting'}

    def __init__(
        self,
        model_name: str = 'svm',
        model_type: Optional[str] = None,
        models_dir: Path = config.MODELS_DIR,
    ):
        self.model_name = model_name.lower()
        self.models_dir = Path(models_dir)

        # Determine type
        if model_type is not None:
            self.model_type = model_type.lower()
        elif self.model_name in self.KERAS_MODELS:
            self.model_type = 'keras'
        else:
            self.model_type = 'sklearn'

        self._manager = ModelManager(models_dir=self.models_dir)
        self._loader  = DataLoader()

        logger.info("Loading model '%s' (type=%s) …", self.model_name, self.model_type)
        self.model = self._manager.load_model(self.model_name)
        logger.info("Model loaded successfully.")

        # For CNN features, keep track of training min/max if available
        self._cnn_min: Optional[float] = None
        self._cnn_max: Optional[float] = None
        self._load_cnn_stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_cnn_stats(self) -> None:
        """Load CNN normalisation statistics from training features if available."""
        stats_path = self.models_dir / "cnn_norm_stats.npz"
        if stats_path.exists():
            data = np.load(str(stats_path))
            self._cnn_min = float(data['min'])
            self._cnn_max = float(data['max'])
            logger.debug("CNN norm stats loaded: min=%.4f  max=%.4f",
                         self._cnn_min, self._cnn_max)

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract features appropriate for the loaded model."""
        if self.model_type == 'keras':
            mel = extract_cnn_features(audio, config.SAMPLE_RATE)

            if self.model_name in ('lstm', 'gender_lstm'):
                # (N_MELS, T) → (T, N_MELS)
                features = mel.T[np.newaxis, ...]          # (1, T, N_MELS)
            else:
                features = mel[np.newaxis, ..., np.newaxis] # (1, H, W, 1)

            # Normalise
            _min = self._cnn_min if self._cnn_min is not None else features.min()
            _max = self._cnn_max if self._cnn_max is not None else features.max()
            features = (features - _min) / (_max - _min + 1e-8)
            return features
        else:
            return extract_all_features(audio, config.SAMPLE_RATE)[np.newaxis, :]

    def _decode(
        self, class_idx: int, probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """Convert raw model output to a human-readable dict."""
        class_name  = config.IDX_TO_CLASS[int(class_idx)]
        confidence  = float(probabilities[class_idx])
        prob_dict   = {
            config.IDX_TO_CLASS[i]: round(float(p), 4)
            for i, p in enumerate(probabilities)
        }
        return {
            'class':         class_name,
            'confidence':    round(confidence, 4),
            'probabilities': prob_dict,
        }

    # ------------------------------------------------------------------
    # Single-file prediction
    # ------------------------------------------------------------------

    def predict(self, audio_path: str) -> Dict[str, Any]:
        """
        Predict gender for a single audio file.

        Parameters
        ----------
        audio_path : str or Path
            Path to the audio file.

        Returns
        -------
        result : dict
            {
              'class':         'male' | 'female',
              'confidence':    float (0–1),
              'probabilities': {'male': float, 'female': float},
            }
        """
        audio_path = str(audio_path)
        audio, _ = load_audio(audio_path, sr=config.SAMPLE_RATE, duration=config.DURATION)
        features  = self._extract_features(audio)

        if self.model_type == 'keras':
            proba     = self.model.predict(features, verbose=0)[0]
            class_idx = int(np.argmax(proba))
        else:
            proba     = self.model.predict_proba(features)[0]
            class_idx = int(np.argmax(proba))

        result = self._decode(class_idx, proba)
        logger.info(
            "Prediction for '%s': class=%s  confidence=%.4f",
            audio_path, result['class'], result['confidence'],
        )
        return result

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(self, audio_dir: str) -> List[Dict[str, Any]]:
        """
        Predict gender for all audio files in *audio_dir*.

        Parameters
        ----------
        audio_dir : str or Path
            Directory containing audio files (non-recursive).

        Returns
        -------
        results : list of dicts
            Each dict has 'file', 'class', 'confidence', 'probabilities'.
        """
        audio_dir = Path(audio_dir)
        if not audio_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {audio_dir}")

        extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        files = sorted(
            f for f in audio_dir.iterdir()
            if f.suffix.lower() in extensions
        )

        if not files:
            logger.warning("No audio files found in '%s'.", audio_dir)
            return []

        results = []
        for file_path in files:
            try:
                result = self.predict(str(file_path))
                result['file'] = str(file_path)
                results.append(result)
                print(
                    f"  {file_path.name:<40}  {result['class']:6s}  "
                    f"(conf={result['confidence']:.4f})"
                )
            except Exception as exc:
                logger.warning("Failed to process '%s': %s", file_path, exc)
                results.append({
                    'file':          str(file_path),
                    'class':         'error',
                    'confidence':    0.0,
                    'probabilities': {},
                    'error':         str(exc),
                })

        return results

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize_prediction(
        self,
        audio_path: str,
        save_path: Optional[str] = None,
    ) -> Path:
        """
        Plot waveform and mel spectrogram with prediction overlay.

        Parameters
        ----------
        audio_path : str or Path
        save_path  : str or Path, optional
            If None, saved to ``results/plots/prediction_<stem>.png``.

        Returns
        -------
        save_path : Path
        """
        audio_path = Path(audio_path)
        audio, sr = load_audio(str(audio_path), sr=config.SAMPLE_RATE,
                               duration=config.DURATION)

        # Run prediction
        result = self.predict(str(audio_path))
        gender     = result['class']
        confidence = result['confidence']
        probs      = result['probabilities']

        # Build plot
        fig = plt.figure(figsize=(14, 8))
        gs  = gridspec.GridSpec(3, 2, figure=fig,
                                height_ratios=[1, 2, 0.6],
                                hspace=0.45, wspace=0.35)

        time_axis = np.linspace(0, config.DURATION, len(audio))

        # ── Waveform ──────────────────────────────────────────────────
        ax_wave = fig.add_subplot(gs[0, :])
        ax_wave.plot(time_axis, audio, lw=0.6,
                     color='steelblue', alpha=0.85)
        ax_wave.set_xlabel('Time (s)', fontsize=10)
        ax_wave.set_ylabel('Amplitude', fontsize=10)
        ax_wave.set_title(f'Waveform — {audio_path.name}', fontsize=11)
        ax_wave.set_xlim([0, config.DURATION])
        ax_wave.grid(True, alpha=0.3)

        # ── Mel Spectrogram ───────────────────────────────────────────
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_mels=config.N_MELS,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        ax_mel = fig.add_subplot(gs[1, :])
        img = librosa.display.specshow(
            mel_db,
            sr=sr,
            hop_length=config.HOP_LENGTH,
            x_axis='time',
            y_axis='mel',
            ax=ax_mel,
            cmap='magma',
        )
        fig.colorbar(img, ax=ax_mel, format='%+2.0f dB', pad=0.01)
        ax_mel.set_title('Mel Spectrogram', fontsize=11)

        # ── Probability bar ───────────────────────────────────────────
        ax_prob = fig.add_subplot(gs[2, :])
        classes = list(probs.keys())
        values  = list(probs.values())
        bar_colors = ['#2196F3' if c == 'male' else '#E91E63' for c in classes]
        bars = ax_prob.barh(classes, values, color=bar_colors,
                            height=0.45, edgecolor='white')
        for bar, val in zip(bars, values):
            ax_prob.text(
                min(val + 0.01, 0.98), bar.get_y() + bar.get_height() / 2,
                f'{val:.1%}', va='center', ha='left', fontsize=11,
                fontweight='bold',
            )
        ax_prob.set_xlim(0, 1.0)
        ax_prob.set_xlabel('Probability', fontsize=10)
        ax_prob.set_title('Class Probabilities', fontsize=11)
        ax_prob.grid(axis='x', alpha=0.3)

        # Main title
        gender_color = '#2196F3' if gender == 'male' else '#E91E63'
        fig.suptitle(
            f'Prediction: {gender.upper()}  (confidence {confidence:.1%})',
            fontsize=15,
            fontweight='bold',
            color=gender_color,
            y=1.01,
        )

        # ── Save ──────────────────────────────────────────────────────
        if save_path is None:
            save_path = config.PLOTS_DIR / f"prediction_{audio_path.stem}.png"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Prediction visualisation saved → %s", save_path)
        return save_path
