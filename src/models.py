"""
Model definitions for Human Audio Gender Recognition.

Provides:
- Traditional ML pipelines: SVM, Random Forest, Gradient Boosting
- Deep Learning models: CNN, LSTM (TensorFlow / Keras)
- ModelManager: save, load, and inventory trained models
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import joblib

# Scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Traditional ML pipelines (sklearn)
# ─────────────────────────────────────────────────────────────────────────────

def get_svm_model() -> Pipeline:
    """
    Build an SVM pipeline.

    StandardScaler → SVC(kernel='rbf', probability=True)

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(
            kernel=config.SVM_KERNEL,
            C=config.SVM_C,
            gamma=config.SVM_GAMMA,
            probability=True,
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
        )),
    ])


def get_random_forest_model() -> Pipeline:
    """
    Build a Random Forest pipeline.

    StandardScaler → RandomForestClassifier(n_estimators=200)
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1,
        )),
    ])


def get_gradient_boosting_model() -> Pipeline:
    """
    Build a Gradient Boosting pipeline.

    StandardScaler → GradientBoostingClassifier
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=config.GB_N_ESTIMATORS,
            learning_rate=config.GB_LEARNING_RATE,
            max_depth=config.GB_MAX_DEPTH,
            random_state=config.RANDOM_STATE,
        )),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Deep Learning models (TensorFlow / Keras)
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn_model(
    input_shape: Tuple[int, int, int] = config.CNN_INPUT_SHAPE,
    n_classes: int = config.N_CLASSES,
) -> Any:
    """
    Convolutional Neural Network for mel-spectrogram classification.

    Architecture
    ------------
    Input (N_MELS, CNN_TIME_FRAMES, 1)
    Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
    Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
    Conv2D(128, 3×3) → BatchNorm → ReLU → GlobalAveragePool
    Dense(256) → BatchNorm → ReLU → Dropout(0.5)
    Dense(n_classes, softmax)

    Parameters
    ----------
    input_shape : tuple (H, W, C)
    n_classes   : int

    Returns
    -------
    model : tf.keras.Model
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input

    inp = Input(shape=input_shape, name='mel_spectrogram')

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False, name='conv1')(inp)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(0.25, name='drop1')(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False, name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(0.25, name='drop2')(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False, name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('relu', name='relu3')(x)
    x = layers.GlobalAveragePooling2D(name='gap')(x)

    # Dense head
    x = layers.Dense(256, use_bias=False, name='dense1')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.Activation('relu', name='relu4')(x)
    x = layers.Dropout(0.5, name='drop3')(x)

    out = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = Model(inputs=inp, outputs=out, name='gender_cnn')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_lstm_model(
    input_shape: Tuple[int, int] = (config.CNN_TIME_FRAMES, config.N_MELS),
    n_classes: int = config.N_CLASSES,
) -> Any:
    """
    Bidirectional LSTM model for sequence-based gender classification.

    The model accepts the mel spectrogram as a time-step sequence:
    shape (CNN_TIME_FRAMES, N_MELS).

    Architecture
    ------------
    Input (T, N_MELS)
    Bidirectional LSTM(128, return_sequences=True) → Dropout(0.3)
    Bidirectional LSTM(64) → Dropout(0.3)
    Dense(128) → ReLU → Dropout(0.4)
    Dense(n_classes, softmax)

    Parameters
    ----------
    input_shape : tuple (T, features)
    n_classes   : int

    Returns
    -------
    model : tf.keras.Model
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input

    inp = Input(shape=input_shape, name='mel_sequence')

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True), name='bilstm1'
    )(inp)
    x = layers.Dropout(0.3, name='drop1')(x)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False), name='bilstm2'
    )(x)
    x = layers.Dropout(0.3, name='drop2')(x)

    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.4, name='drop3')(x)

    out = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = Model(inputs=inp, outputs=out, name='gender_lstm')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ModelManager
# ─────────────────────────────────────────────────────────────────────────────

class ModelManager:
    """
    Centralised model persistence and inventory.

    Supports both sklearn pipelines (saved with joblib) and Keras models
    (saved in SavedModel / .h5 format).
    """

    SKLEARN_MODELS = {'svm', 'random_forest', 'gradient_boosting'}
    KERAS_MODELS   = {'cnn', 'lstm'}

    def __init__(self, models_dir: Path = config.MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _sklearn_path(self, name: str) -> Path:
        return self.models_dir / f"{name}.joblib"

    def _keras_path(self, name: str) -> Path:
        return self.models_dir / f"{name}.h5"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_model(self, model: Any, name: str) -> Path:
        """
        Persist *model* under *name*.

        sklearn pipelines → joblib  ({name}.joblib)
        Keras models      → HDF5    ({name}.h5)

        Parameters
        ----------
        model : sklearn Pipeline or tf.keras.Model
        name  : str
            One of 'svm', 'random_forest', 'gradient_boosting', 'cnn', 'lstm'.

        Returns
        -------
        save_path : Path
        """
        name = name.lower()
        if name in self.SKLEARN_MODELS:
            save_path = self._sklearn_path(name)
            joblib.dump(model, save_path)
            logger.info("Sklearn model '%s' saved → %s", name, save_path)
        elif name in self.KERAS_MODELS:
            save_path = self._keras_path(name)
            model.save(str(save_path))
            logger.info("Keras model '%s' saved → %s", name, save_path)
        else:
            # Fall back to joblib for unknown model types
            save_path = self.models_dir / f"{name}.joblib"
            joblib.dump(model, save_path)
            logger.info("Unknown model type '%s' saved via joblib → %s", name, save_path)

        return save_path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_model(self, name: str) -> Any:
        """
        Load a previously saved model by *name*.

        Parameters
        ----------
        name : str

        Returns
        -------
        model
        """
        name = name.lower()
        if name in self.SKLEARN_MODELS:
            path = self._sklearn_path(name)
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {path}")
            model = joblib.load(path)
            logger.info("Sklearn model '%s' loaded from %s", name, path)
            return model
        elif name in self.KERAS_MODELS:
            import tensorflow as tf
            path = self._keras_path(name)
            if not path.exists():
                raise FileNotFoundError(f"Keras model not found: {path}")
            model = tf.keras.models.load_model(str(path))
            logger.info("Keras model '%s' loaded from %s", name, path)
            return model
        else:
            # Try joblib fallback
            path = self.models_dir / f"{name}.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {path}")
            return joblib.load(path)

    # ------------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------------

    def get_model_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Scan the models directory and return a dict of all saved models.

        Returns
        -------
        summary : dict
            {model_name: {'path': str, 'size_kb': float, 'type': str}}
        """
        summary: Dict[str, Dict[str, Any]] = {}

        for file in sorted(self.models_dir.iterdir()):
            if file.suffix in ('.joblib', '.h5') and file.stem != '.gitkeep':
                name = file.stem
                model_type = 'keras' if file.suffix == '.h5' else 'sklearn'
                size_kb = file.stat().st_size / 1024
                summary[name] = {
                    'path':     str(file),
                    'size_kb':  round(size_kb, 2),
                    'type':     model_type,
                }

        return summary
