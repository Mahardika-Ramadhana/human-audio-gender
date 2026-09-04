"""
Generate all plots required for the IEEE report.
Run from project root: python generate_plots.py
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import librosa
import librosa.display
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)

sys.path.insert(0, str(Path(__file__).parent))
from src import config

PLOTS_DIR = config.PLOTS_DIR
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {"male": "#2196F3", "female": "#E91E63"}
sns.set_theme(style="whitegrid", font_scale=1.1)

# ─── Load data ───────────────────────────────────────────────────────────────
print("Loading features...")
train = np.load(config.FEATURES_TRAIN_PATH)
X_train, y_train = train["features"], train["labels"]

test = np.load(config.FEATURES_TEST_PATH)
X_test, y_test = test["features"], test["labels"]

val = np.load(config.FEATURES_VAL_PATH)
X_val, y_val = val["features"], val["labels"]

X_all = np.vstack([X_train, X_val, X_test])
y_all = np.concatenate([y_train, y_val, y_test])

print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

# ─── Pick sample audio files ─────────────────────────────────────────────────
male_files   = sorted((config.DATA_RAW_DIR / "male").glob("*.wav"))
female_files = sorted((config.DATA_RAW_DIR / "female").glob("*.wav"))
male_path   = str(male_files[0])
female_path = str(female_files[0])

# ─── 1. Waveform comparison ───────────────────────────────────────────────────
print("  [1/10] Waveform comparison...")
def load_audio(path):
    y, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True, duration=config.DURATION)
    n = int(config.SAMPLE_RATE * config.DURATION)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y[:n]

audio_m = load_audio(male_path)
audio_f = load_audio(female_path)
t = np.linspace(0, config.DURATION, len(audio_m))

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
for ax, audio, gender, color in zip(
    axes, [audio_m, audio_f], ["Male", "Female"],
    [PALETTE["male"], PALETTE["female"]]
):
    ax.plot(t, audio, color=color, linewidth=0.5, alpha=0.9)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.set_title(f"{gender} Speaker", fontsize=11, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, config.DURATION)
axes[-1].set_xlabel("Time (s)", fontsize=10)
fig.suptitle("Waveform Comparison: Male vs. Female", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "01_waveform_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ─── 2. Mel spectrogram comparison ───────────────────────────────────────────
print("  [2/10] Mel spectrogram comparison...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, fpath, title, cmap in zip(
    axes, [male_path, female_path],
    ["Male Speaker", "Female Speaker"], ["Blues", "RdPu"]
):
    audio = load_audio(fpath)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=config.SAMPLE_RATE, n_mels=config.N_MELS,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(
        mel_db, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=ax, cmap=cmap,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title, fontsize=12, fontweight="bold")
fig.suptitle("Mel Spectrogram Comparison: Male vs. Female", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "02_spectrogram_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ─── 3. MFCC heatmap ─────────────────────────────────────────────────────────
print("  [3/10] MFCC heatmap...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, fpath, title in zip(
    axes, [male_path, female_path], ["Male Speaker", "Female Speaker"]
):
    audio = load_audio(fpath)
    mfcc = librosa.feature.mfcc(
        y=audio, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    img = librosa.display.specshow(
        mfcc, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
        x_axis="time", ax=ax, cmap="coolwarm",
    )
    fig.colorbar(img, ax=ax)
    ax.set_ylabel("MFCC Coefficient Index", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
fig.suptitle(f"MFCC Heatmap (first {config.N_MFCC} coefficients)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "03_mfcc_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ─── 4. Feature distributions (violin) ───────────────────────────────────────
print("  [4/10] Feature distributions...")
import pandas as pd
n_feat = 10
feature_names = [f"MFCC_{i+1}" for i in range(n_feat)]
X_sub = X_train[:, :n_feat]
df = pd.DataFrame(X_sub, columns=feature_names)
df["Gender"] = ["male" if yi == 0 else "female" for yi in y_train]
df_melt = df.melt(id_vars="Gender", var_name="Feature", value_name="Value")

fig, ax = plt.subplots(figsize=(14, 5))
sns.violinplot(
    data=df_melt, x="Feature", y="Value", hue="Gender",
    split=True, inner="quart",
    palette={"male": PALETTE["male"], "female": PALETTE["female"]},
    ax=ax,
)
ax.set_title(f"Distribution of Top {n_feat} MFCC Mean Coefficients by Gender",
             fontsize=13, fontweight="bold")
ax.set_xlabel("MFCC Coefficient", fontsize=11)
ax.set_ylabel("Mean Value", fontsize=11)
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "04_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ─── 5. F0 pitch distribution ─────────────────────────────────────────────────
print("  [5/10] F0 pitch distribution...")
import random
rng = random.Random(config.RANDOM_STATE)

all_audio_files = (
    [(str(p), 0) for p in male_files] +
    [(str(p), 1) for p in female_files]
)
sample = rng.sample(all_audio_files, min(80, len(all_audio_files)))

f0_male, f0_female = [], []
for fpath, label in sample:
    try:
        audio = load_audio(fpath)
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=50, fmax=500, sr=config.SAMPLE_RATE,
            hop_length=config.HOP_LENGTH,
        )
        f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
        if len(f0_voiced) > 0:
            med = float(np.median(f0_voiced))
            if label == 0:
                f0_male.append(med)
            else:
                f0_female.append(med)
    except Exception:
        pass

fig, ax = plt.subplots(figsize=(9, 4))
bins = np.linspace(50, 500, 50)
ax.hist(f0_male,   bins=bins, alpha=0.65, color=PALETTE["male"],
        label=f"Male (n={len(f0_male)})", edgecolor="white")
ax.hist(f0_female, bins=bins, alpha=0.65, color=PALETTE["female"],
        label=f"Female (n={len(f0_female)})", edgecolor="white")
ax.axvline(85,  color=PALETTE["male"],   linestyle="--", linewidth=1.2, label="Male range (85–180 Hz)")
ax.axvline(180, color=PALETTE["male"],   linestyle="--", linewidth=1.2)
ax.axvline(165, color=PALETTE["female"], linestyle=":",  linewidth=1.2, label="Female range (165–255 Hz)")
ax.axvline(255, color=PALETTE["female"], linestyle=":",  linewidth=1.2)
ax.set_xlabel("Median F0 (Hz)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Fundamental Frequency (F0) Distribution by Gender", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "05_pitch_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"    F0 computed: male={len(f0_male)}, female={len(f0_female)}")

# ─── 6. PCA scatter ───────────────────────────────────────────────────────────
print("  [6/10] PCA scatter...")
scaler = StandardScaler()
pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
X_pca = pca.fit_transform(scaler.fit_transform(X_all))
var1, var2 = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(7, 6))
for label_idx, label_name, color in [(0, "Male", PALETTE["male"]), (1, "Female", PALETTE["female"])]:
    mask = y_all == label_idx
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label_name,
               alpha=0.5, s=30, edgecolors="none")
ax.set_xlabel(f"PC 1 ({var1:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC 2 ({var2:.1f}% variance)", fontsize=11)
ax.set_title("PCA Projection of 380-D Feature Space", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.8)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "06_pca_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ─── 7 & 8. Load models, predict, confusion matrices ─────────────────────────
print("  [7/10] Loading models and computing predictions...")
import joblib

models_results = {}

# --- SVM ---
svm_model = joblib.load(config.MODELS_DIR / "svm.joblib")
y_pred_svm = svm_model.predict(X_test)
y_proba_svm = svm_model.predict_proba(X_test)
models_results["SVM"] = {
    "y_pred": y_pred_svm,
    "y_proba": y_proba_svm,
    "accuracy":  accuracy_score(y_test, y_pred_svm),
    "precision": precision_score(y_test, y_pred_svm, average="weighted"),
    "recall":    recall_score(y_test, y_pred_svm,    average="weighted"),
    "f1":        f1_score(y_test, y_pred_svm,        average="weighted"),
    "auc_roc":   roc_auc_score(y_test, y_proba_svm[:, 1]),
}

# --- Random Forest ---
rf_model = joblib.load(config.MODELS_DIR / "random_forest.joblib")
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)
models_results["Random Forest"] = {
    "y_pred": y_pred_rf,
    "y_proba": y_proba_rf,
    "accuracy":  accuracy_score(y_test, y_pred_rf),
    "precision": precision_score(y_test, y_pred_rf, average="weighted"),
    "recall":    recall_score(y_test, y_pred_rf,    average="weighted"),
    "f1":        f1_score(y_test, y_pred_rf,        average="weighted"),
    "auc_roc":   roc_auc_score(y_test, y_proba_rf[:, 1]),
}

# --- Gradient Boosting ---
gb_model = joblib.load(config.MODELS_DIR / "gradient_boosting.joblib")
y_pred_gb = gb_model.predict(X_test)
y_proba_gb = gb_model.predict_proba(X_test)
models_results["Gradient Boosting"] = {
    "y_pred": y_pred_gb,
    "y_proba": y_proba_gb,
    "accuracy":  accuracy_score(y_test, y_pred_gb),
    "precision": precision_score(y_test, y_pred_gb, average="weighted"),
    "recall":    recall_score(y_test, y_pred_gb,    average="weighted"),
    "f1":        f1_score(y_test, y_pred_gb,        average="weighted"),
    "auc_roc":   roc_auc_score(y_test, y_proba_gb[:, 1]),
}

# --- CNN ---
cnn_test = np.load(config.CNN_FEATURES_TEST_PATH)
X_test_cnn = cnn_test["features"]
y_test_cnn = cnn_test["labels"]
# Reshape for CNN: (N, H, W, 1)
if X_test_cnn.ndim == 3:
    X_test_cnn_4d = X_test_cnn[..., np.newaxis]
else:
    X_test_cnn_4d = X_test_cnn

# Normalize with train stats
cnn_norm = np.load(config.MODELS_DIR / "cnn_norm_stats.npz")
if "mean" in cnn_norm:
    cnn_mean = cnn_norm["mean"]
    cnn_std  = cnn_norm["std"]
    X_test_cnn_norm = (X_test_cnn_4d - cnn_mean) / (cnn_std + 1e-8)
elif "min" in cnn_norm:
    cnn_min = cnn_norm["min"]
    cnn_max = cnn_norm["max"]
    X_test_cnn_norm = (X_test_cnn_4d - cnn_min) / (cnn_max - cnn_min + 1e-8)
else:
    X_test_cnn_norm = X_test_cnn_4d

import tensorflow as tf
cnn_model = tf.keras.models.load_model(config.MODELS_DIR / "gender_cnn_best.h5")
y_proba_cnn = cnn_model.predict(X_test_cnn_norm, verbose=0)
y_pred_cnn = np.argmax(y_proba_cnn, axis=1)
models_results["CNN"] = {
    "y_pred": y_pred_cnn,
    "y_proba": y_proba_cnn,
    "accuracy":  accuracy_score(y_test_cnn, y_pred_cnn),
    "precision": precision_score(y_test_cnn, y_pred_cnn, average="weighted"),
    "recall":    recall_score(y_test_cnn, y_pred_cnn,    average="weighted"),
    "f1":        f1_score(y_test_cnn, y_pred_cnn,        average="weighted"),
    "auc_roc":   roc_auc_score(y_test_cnn, y_proba_cnn[:, 1]),
}

# --- BiLSTM --- (expects shape: (N, time_steps, mel_bins) = (N, 130, 128))
lstm_model = tf.keras.models.load_model(config.MODELS_DIR / "gender_lstm_best.h5")
# CNN input is (N, 128, 130, 1), LSTM needs (N, 130, 128)
X_test_lstm = X_test_cnn_norm[:, :, :, 0].transpose(0, 2, 1)
y_proba_lstm = lstm_model.predict(X_test_lstm, verbose=0)
y_pred_lstm = np.argmax(y_proba_lstm, axis=1)
models_results["BiLSTM"] = {
    "y_pred": y_pred_lstm,
    "y_proba": y_proba_lstm,
    "accuracy":  accuracy_score(y_test_cnn, y_pred_lstm),
    "precision": precision_score(y_test_cnn, y_pred_lstm, average="weighted"),
    "recall":    recall_score(y_test_cnn, y_pred_lstm,    average="weighted"),
    "f1":        f1_score(y_test_cnn, y_pred_lstm,        average="weighted"),
    "auc_roc":   roc_auc_score(y_test_cnn, y_proba_lstm[:, 1]),
}

# Print metrics table
print("\n  === Model Metrics ===")
print(f"  {'Model':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
for name, res in models_results.items():
    print(f"  {name:<20} {res['accuracy']*100:6.2f} {res['precision']*100:6.2f} "
          f"{res['recall']*100:6.2f} {res['f1']*100:6.2f} {res['auc_roc']:6.3f}")

# Confusion matrices for SVM and CNN (as referenced in LaTeX)
print("\n  [8/10] Confusion matrices...")
CLASS_LABELS = ["male", "female"]

def plot_cm(y_true, y_pred, model_name, save_name):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    annot = np.array([[f"{cm_norm[i,j]:.2f}\n({cm[i,j]})"
                       for j in range(2)] for i in range(2)])
    sns.heatmap(cm_norm, annot=annot, fmt="",
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                cmap="Blues", vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / save_name, dpi=150, bbox_inches="tight")
    plt.close(fig)

plot_cm(y_test, y_pred_svm, "SVM", "08_confusion_svm.png")
plot_cm(y_test_cnn, y_pred_cnn, "CNN", "08_confusion_cnn.png")

# ─── 9. ROC curves (all models) ───────────────────────────────────────────────
print("  [9/10] ROC curves...")

# Use common y_test (flat features) for SVM/RF/GB, cnn test for CNN/LSTM
# Both test sets are same samples (same split) so use cnn y_test for all
fig, ax = plt.subplots(figsize=(7, 6))
colors = plt.cm.tab10(np.linspace(0, 1, 5))

for (name, res), color in zip(models_results.items(), colors):
    if name in ["SVM", "Random Forest", "Gradient Boosting"]:
        yt = y_test
    else:
        yt = y_test_cnn
    fpr, tpr, _ = roc_curve(yt, res["y_proba"][:, 1])
    auc = res["auc_roc"]
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
fig.tight_layout()
fig.savefig(PLOTS_DIR / "09_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ─── 10 (→ 11). Model comparison bar chart ───────────────────────────────────
print("  [10/10] Model comparison bar chart...")
metrics  = ["accuracy", "precision", "recall", "f1"]
m_labels = ["Accuracy", "Precision", "Recall", "F1"]
model_names = list(models_results.keys())
n_models = len(model_names)
n_metrics = 4

x = np.arange(n_models)
width = 0.18
bar_colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]

fig, ax = plt.subplots(figsize=(max(10, n_models * 2), 5))
for i, (metric, mlabel, color) in enumerate(zip(metrics, m_labels, bar_colors)):
    values = [models_results[m][metric] * 100 for m in model_names]
    offset = i * width - (n_metrics / 2 - 0.5) * width
    bars = ax.bar(x + offset, values, width, label=mlabel, color=color, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=10)
ax.set_ylabel("Score (%)", fontsize=11)
ax.set_ylim(0, 115)
ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "11_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\n  All plots saved to:", PLOTS_DIR)
print("\n  Final metrics for LaTeX table:")
for name, res in models_results.items():
    print(f"  {name}: acc={res['accuracy']*100:.1f}  prec={res['precision']*100:.1f}  "
          f"rec={res['recall']*100:.1f}  f1={res['f1']*100:.1f}  auc={res['auc_roc']:.3f}")
