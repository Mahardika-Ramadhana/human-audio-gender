"""
Configuration file for Human Audio Gender Recognition project.
All hyperparameters, paths, and constants are defined here.
"""

from pathlib import Path

# ─── Project Root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── Audio Parameters ────────────────────────────────────────────────────────
SAMPLE_RATE = 22050          # Hz
DURATION = 3                 # seconds
N_SAMPLES = SAMPLE_RATE * DURATION  # total samples per clip

# ─── MFCC / Spectrogram Parameters ───────────────────────────────────────────
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
N_CHROMA = 12
N_SPECTRAL_CONTRAST_BANDS = 6

# ─── Derived Feature Dimensions ──────────────────────────────────────────────
# MFCC:               N_MFCC * 2  = 80
# Chroma:             N_CHROMA * 2 = 24
# Mel spectrogram:    N_MELS * 2   = 256
# Spectral contrast:  (N_SPECTRAL_CONTRAST_BANDS + 1) * 2 = 14
# ZCR:                2
# RMS:                2
# Spectral rolloff:   2
# Total ~380

# ─── Directory Paths ─────────────────────────────────────────────────────────
DATA_RAW_DIR       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR         = PROJECT_ROOT / "results" / "models"
PLOTS_DIR          = PROJECT_ROOT / "results" / "plots"

# Ensure directories exist at import time
for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── Classes ─────────────────────────────────────────────────────────────────
CLASSES = ['male', 'female']
N_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# ─── Train / Val / Test Split ────────────────────────────────────────────────
TEST_SIZE      = 0.2
VAL_SIZE       = 0.1
RANDOM_STATE   = 42

# ─── Training Hyperparameters ────────────────────────────────────────────────
EPOCHS         = 50
BATCH_SIZE     = 32
LEARNING_RATE  = 0.001

# Early stopping / scheduler patience
EARLY_STOP_PATIENCE = 10
LR_PATIENCE         = 5
LR_FACTOR           = 0.5
MIN_LR              = 1e-6

# ─── CNN Input Shape ─────────────────────────────────────────────────────────
# mel spectrogram frame count at default settings:
#   frames = ceil(N_SAMPLES / HOP_LENGTH) + 1 ≈ 130
CNN_TIME_FRAMES = 130
CNN_INPUT_SHAPE = (N_MELS, CNN_TIME_FRAMES, 1)  # (height, width, channels)

# ─── SVM Parameters ──────────────────────────────────────────────────────────
SVM_KERNEL = 'rbf'
SVM_C      = 10.0
SVM_GAMMA  = 'scale'

# ─── Random Forest Parameters ────────────────────────────────────────────────
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH    = None
RF_MIN_SAMPLES_SPLIT = 2

# ─── Gradient Boosting Parameters ────────────────────────────────────────────
GB_N_ESTIMATORS  = 150
GB_LEARNING_RATE = 0.1
GB_MAX_DEPTH     = 4

# ─── Paths for Saved Features ────────────────────────────────────────────────
FEATURES_TRAIN_PATH = DATA_PROCESSED_DIR / "features_train.npz"
FEATURES_VAL_PATH   = DATA_PROCESSED_DIR / "features_val.npz"
FEATURES_TEST_PATH  = DATA_PROCESSED_DIR / "features_test.npz"
CNN_FEATURES_TRAIN_PATH = DATA_PROCESSED_DIR / "cnn_features_train.npz"
CNN_FEATURES_VAL_PATH   = DATA_PROCESSED_DIR / "cnn_features_val.npz"
CNN_FEATURES_TEST_PATH  = DATA_PROCESSED_DIR / "cnn_features_test.npz"

# ─── Training Log ────────────────────────────────────────────────────────────
TRAINING_LOG_PATH = MODELS_DIR / "training_log.json"
