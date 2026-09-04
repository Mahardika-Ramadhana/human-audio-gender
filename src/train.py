"""
Training pipeline for Human Audio Gender Recognition.

Handles:
- Training traditional ML models (SVM, RF, GB) with cross-validation metrics
- Training CNN / LSTM models with callbacks
- Full pipeline orchestration: feature loading → train → save → log
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from src import config
from src.data_loader import DataLoader
from src.models import (
    ModelManager,
    get_svm_model,
    get_random_forest_model,
    get_gradient_boosting_model,
    build_cnn_model,
    build_lstm_model,
)

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Orchestrates the end-to-end training of all models.
    """

    def __init__(self, models_dir: Path = config.MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.model_manager = ModelManager(models_dir=self.models_dir)
        self.loader = DataLoader()
        self.training_log: Dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Sklearn / traditional ML
    # ──────────────────────────────────────────────────────────────────────

    def train_sklearn_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Fit a sklearn pipeline and evaluate on the validation set.

        Parameters
        ----------
        model      : sklearn Pipeline
        X_train    : np.ndarray, shape (n_train, n_features)
        y_train    : np.ndarray, shape (n_train,)
        X_val      : np.ndarray, shape (n_val, n_features)
        y_val      : np.ndarray, shape (n_val,)
        model_name : str

        Returns
        -------
        metrics : dict
            {'train_acc', 'val_acc', 'training_time_s'}
        """
        logger.info("Training %s …", model_name)
        t0 = time.time()

        model.fit(X_train, y_train)

        elapsed = time.time() - t0
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc   = accuracy_score(y_val,   model.predict(X_val))

        metrics = {
            'model_name':      model_name,
            'train_acc':       round(float(train_acc), 4),
            'val_acc':         round(float(val_acc), 4),
            'training_time_s': round(elapsed, 2),
        }

        logger.info(
            "%s — train_acc=%.4f  val_acc=%.4f  time=%.1fs",
            model_name, train_acc, val_acc, elapsed,
        )

        # Save model
        self.model_manager.save_model(model, model_name)
        return metrics

    # ──────────────────────────────────────────────────────────────────────
    # CNN / LSTM (Keras)
    # ──────────────────────────────────────────────────────────────────────

    def train_cnn_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        cfg=config,
    ) -> Dict[str, Any]:
        """
        Train a Keras CNN / LSTM model with callbacks.

        Callbacks used
        --------------
        - EarlyStopping(patience=10, restore_best_weights=True)
        - ModelCheckpoint (saves best epoch)
        - ReduceLROnPlateau(patience=5, factor=0.5)

        Parameters
        ----------
        model  : tf.keras.Model
        X_train, y_train, X_val, y_val : arrays
        cfg    : config module

        Returns
        -------
        result : dict
            {'model_name', 'history', 'train_acc', 'val_acc', 'training_time_s'}
        """
        import tensorflow as tf

        model_name = model.name  # 'gender_cnn' or 'gender_lstm'
        checkpoint_path = str(self.models_dir / f"{model_name}_best.h5")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=cfg.EARLY_STOP_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=cfg.LR_FACTOR,
                patience=cfg.LR_PATIENCE,
                min_lr=cfg.MIN_LR,
                verbose=1,
            ),
        ]

        logger.info("Training Keras model '%s' …", model_name)
        t0 = time.time()

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg.EPOCHS,
            batch_size=cfg.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )

        elapsed = time.time() - t0

        # Metrics from best epoch
        val_acc   = float(max(history.history['val_accuracy']))
        train_acc = float(history.history['accuracy'][
            np.argmax(history.history['val_accuracy'])
        ])

        result = {
            'model_name':      model_name,
            'train_acc':       round(train_acc, 4),
            'val_acc':         round(val_acc, 4),
            'training_time_s': round(elapsed, 2),
            'epochs_run':      len(history.history['loss']),
            'history': {
                k: [round(float(v), 6) for v in vals]
                for k, vals in history.history.items()
            },
        }

        logger.info(
            "%s — best val_acc=%.4f  epochs=%d  time=%.1fs",
            model_name, val_acc, result['epochs_run'], elapsed,
        )

        # Save final model
        self.model_manager.save_model(model, model_name.replace('gender_', ''))
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Full orchestration
    # ──────────────────────────────────────────────────────────────────────

    def train_all_models(
        self,
        features_path: Optional[Path] = None,
        cnn_features_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Load pre-extracted features and train all models.

        Parameters
        ----------
        features_path : Path or None
            Path to flat feature .npz files (train / val / test).
            Defaults to config paths.
        cnn_features_path : Path or None
            Path to CNN feature .npz files. Defaults to config paths.

        Returns
        -------
        all_metrics : dict
            Training metrics for every model.
        """
        # ── Load flat features ─────────────────────────────────────────
        train_path = features_path or config.FEATURES_TRAIN_PATH
        val_path   = config.FEATURES_VAL_PATH
        test_path  = config.FEATURES_TEST_PATH

        logger.info("Loading flat features …")
        X_train, y_train = self.loader.load_features(train_path)
        X_val,   y_val   = self.loader.load_features(val_path)

        logger.info(
            "Flat features — train: %s  val: %s",
            X_train.shape, X_val.shape,
        )

        all_metrics: Dict[str, Any] = {}

        # ── SVM ───────────────────────────────────────────────────────
        svm = get_svm_model()
        all_metrics['svm'] = self.train_sklearn_model(
            svm, X_train, y_train, X_val, y_val, 'svm'
        )

        # ── Random Forest ─────────────────────────────────────────────
        rf = get_random_forest_model()
        all_metrics['random_forest'] = self.train_sklearn_model(
            rf, X_train, y_train, X_val, y_val, 'random_forest'
        )

        # ── Gradient Boosting ─────────────────────────────────────────
        gb = get_gradient_boosting_model()
        all_metrics['gradient_boosting'] = self.train_sklearn_model(
            gb, X_train, y_train, X_val, y_val, 'gradient_boosting'
        )

        # ── CNN ───────────────────────────────────────────────────────
        cnn_train = cnn_features_path or config.CNN_FEATURES_TRAIN_PATH
        cnn_val   = config.CNN_FEATURES_VAL_PATH

        if Path(cnn_train).exists() and Path(cnn_val).exists():
            logger.info("Loading CNN features …")
            X_cnn_train, y_cnn_train = self.loader.load_features(cnn_train)
            X_cnn_val,   y_cnn_val   = self.loader.load_features(cnn_val)

            # CNN expects (N, H, W, 1)
            if X_cnn_train.ndim == 3:
                X_cnn_train = X_cnn_train[..., np.newaxis]
                X_cnn_val   = X_cnn_val[..., np.newaxis]

            # Normalise to [0, 1]
            _min = X_cnn_train.min()
            _max = X_cnn_train.max()
            X_cnn_train = (X_cnn_train - _min) / (_max - _min + 1e-8)
            X_cnn_val   = (X_cnn_val   - _min) / (_max - _min + 1e-8)

            cnn_model = build_cnn_model(
                input_shape=X_cnn_train.shape[1:],
                n_classes=config.N_CLASSES,
            )
            all_metrics['cnn'] = self.train_cnn_model(
                cnn_model, X_cnn_train, y_cnn_train, X_cnn_val, y_cnn_val
            )

            # LSTM — reshape to (N, T, F)
            X_lstm_train = X_cnn_train[..., 0].transpose(0, 2, 1)  # (N, T, N_MELS)
            X_lstm_val   = X_cnn_val[..., 0].transpose(0, 2, 1)

            lstm_model = build_lstm_model(
                input_shape=X_lstm_train.shape[1:],
                n_classes=config.N_CLASSES,
            )
            all_metrics['lstm'] = self.train_cnn_model(
                lstm_model, X_lstm_train, y_cnn_train, X_lstm_val, y_cnn_val
            )
        else:
            logger.warning(
                "CNN feature files not found — skipping CNN/LSTM training.\n"
                "Run feature extraction first: python main.py --mode extract"
            )

        # ── Save training log ──────────────────────────────────────────
        self._save_log(all_metrics)
        return all_metrics

    # ──────────────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────────────

    def _save_log(self, metrics: Dict[str, Any]) -> None:
        """Persist training metrics as JSON."""
        log_path = config.TRAINING_LOG_PATH
        # Strip non-serialisable values (e.g. full history lists > 1k items)
        serialisable = {}
        for name, m in metrics.items():
            serialisable[name] = {
                k: v for k, v in m.items() if k != 'history'
            }
            if 'history' in m:
                # Keep only final epoch values
                hist = m['history']
                serialisable[name]['final_train_acc'] = round(
                    float(hist['accuracy'][-1]), 4
                ) if 'accuracy' in hist else None
                serialisable[name]['final_val_acc'] = round(
                    float(hist['val_accuracy'][-1]), 4
                ) if 'val_accuracy' in hist else None

        with open(log_path, 'w') as fh:
            json.dump(serialisable, fh, indent=2)

        logger.info("Training log saved → %s", log_path)
