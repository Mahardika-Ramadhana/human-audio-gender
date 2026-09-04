"""
Visualization module for Human Audio Gender Recognition.
Mata kuliah: Pengenalan Pola

Provides publication-quality plots for the IEEE report:
  1. plot_waveform_comparison      — waveform male vs female
  2. plot_spectrogram_comparison   — mel spectrogram male vs female
  3. plot_mfcc_heatmap             — MFCC coefficients over time
  4. plot_feature_distributions    — violin plots per gender class
  5. plot_pitch_distribution       — F0 histogram male vs female
  6. plot_pca_2d                   — PCA projection of feature space
  7. plot_correlation_heatmap      — top feature correlations
  8. plot_confusion_matrix         — per model
  9. plot_roc_curves               — all models on one axes
 10. plot_training_history         — loss/accuracy curves (CNN/LSTM)
 11. plot_model_comparison         — bar chart accuracy all models
 12. generate_all_visualizations   — run all of the above
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")                    # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import librosa
import librosa.display

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src import config
from src.data_loader import load_audio

logger = logging.getLogger(__name__)

# ── Consistent style ─────────────────────────────────────────────────────────
PALETTE   = {"male": "#2196F3", "female": "#E91E63"}
CLASS_MAP = {0: "male", 1: "female"}
sns.set_theme(style="whitegrid", font_scale=1.1)
PLOTS_DIR = config.PLOTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# 1. Waveform Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_waveform_comparison(
    male_path: str,
    female_path: str,
    save_name: str = "01_waveform_comparison.png",
) -> Path:
    """
    Side-by-side waveform plot for one male and one female sample.
    Shows amplitude over time. Male speakers typically exhibit lower
    frequency oscillation visible to the naked eye.
    """
    audio_m, sr = load_audio(male_path,   config.SAMPLE_RATE, config.DURATION)
    audio_f, _  = load_audio(female_path, config.SAMPLE_RATE, config.DURATION)
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
    fig.suptitle("Waveform Comparison: Male vs. Female", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.tight_layout()

    save_path = PLOTS_DIR / save_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mel Spectrogram Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectrogram_comparison(
    male_path: str,
    female_path: str,
    save_name: str = "02_spectrogram_comparison.png",
) -> Path:
    """
    Log-mel spectrograms for one male and one female sample.
    Female voices show higher-frequency energy concentration.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, fpath, title, cmap in zip(
        axes,
        [male_path, female_path],
        ["Male Speaker", "Female Speaker"],
        ["Blues", "RdPu"],
    ):
        audio, sr = load_audio(fpath, config.SAMPLE_RATE, config.DURATION)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=config.N_MELS,
            n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        img = librosa.display.specshow(
            mel_db, sr=sr, hop_length=config.HOP_LENGTH,
            x_axis="time", y_axis="mel", ax=ax, cmap=cmap,
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.suptitle("Mel Spectrogram Comparison: Male vs. Female",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    save_path = PLOTS_DIR / save_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 3. MFCC Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_mfcc_heatmap(
    male_path: str,
    female_path: str,
    save_name: str = "03_mfcc_heatmap.png",
) -> Path:
    """
    MFCC coefficient values over time as a heatmap.
    Lower MFCCs (C1-C5) carry most of the gender-discriminative information.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, fpath, title in zip(
        axes,
        [male_path, female_path],
        ["Male Speaker", "Female Speaker"],
    ):
        audio, sr = load_audio(fpath, config.SAMPLE_RATE, config.DURATION)
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=config.N_MFCC,
            n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        )
        img = librosa.display.specshow(
            mfcc, sr=sr, hop_length=config.HOP_LENGTH,
            x_axis="time", ax=ax, cmap="coolwarm",
        )
        fig.colorbar(img, ax=ax)
        ax.set_ylabel("MFCC Coefficient Index", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.suptitle(f"MFCC Heatmap (first {config.N_MFCC} coefficients)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    save_path = PLOTS_DIR / save_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature Distribution (Violin plots)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_distributions(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int = 10,
    save_name: str = "04_feature_distributions.png",
) -> Path:
    """
    Violin plots comparing the distribution of the top *n_features* MFCC
    means between male and female samples.
    These distributions reveal which cepstral coefficients are most
    discriminative for gender.
    """
    # Use first n_features (MFCC means: indices 0..N_MFCC-1)
    feature_names = [f"MFCC_{i+1}" for i in range(n_features)]
    X_sub = X[:, :n_features]

    import pandas as pd
    df = pd.DataFrame(X_sub, columns=feature_names)
    df["Gender"] = [CLASS_MAP[int(yi)] for yi in y]
    df_melt = df.melt(id_vars="Gender", var_name="Feature", value_name="Value")

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.violinplot(
        data=df_melt,
        x="Feature", y="Value", hue="Gender",
        split=True, inner="quart",
        palette={"male": PALETTE["male"], "female": PALETTE["female"]},
        ax=ax,
    )
    ax.set_title(
        f"Distribution of Top {n_features} MFCC Mean Coefficients by Gender",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("MFCC Coefficient", fontsize=11)
    ax.set_ylabel("Mean Value", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    save_path = PLOTS_DIR / save_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pitch (F0) Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_pitch_distribution(
    data: List[Tuple[str, int]],
    n_samples: int = 80,
    save_name: str = "05_pitch_distribution.png",
) -> Path:
    """
    Histogram of estimated fundamental frequency (F0) for male and female
    samples. Demonstrates the classic 85-180 Hz (male) vs 165-255 Hz (female)
    separation — the primary acoustic correlate of gender.
    """
    import random
    rng = random.Random(config.RANDOM_STATE)
    sample = rng.sample(data, min(n_samples, len(data)))

    f0_male, f0_female = [], []

    for fpath, label in sample:
        try:
            audio, sr = load_audio(fpath, config.SAMPLE_RATE, config.DURATION)
            f0, voiced_flag, _ = librosa.pyin(
                audio, fmin=50, fmax=500, sr=sr,
                hop_length=config.HOP_LENGTH,
            )
            f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
            if len(f0_voiced) > 0:
                median_f0 = float(np.median(f0_voiced))
                if label == 0:
                    f0_male.append(median_f0)
                else:
                    f0_female.append(median_f0)
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(50, 500, 50)
    ax.hist(f0_male,   bins=bins, alpha=0.65, color=PALETTE["male"],
            label=f"Male (n={len(f0_male)})", edgecolor="white")
    ax.hist(f0_female, bins=bins, alpha=0.65, color=PALETTE["female"],
            label=f"Female (n={len(f0_female)})", edgecolor="white")

    ax.axvline(85,  color=PALETTE["male"],   linestyle="--", linewidth=1.2,
               label="Male range (85–180 Hz)")
    ax.axvline(180, color=PALETTE["male"],   linestyle="--", linewidth=1.2)
    ax.axvline(165, color=PALETTE["female"], linestyle=":",  linewidth=1.2,
               label="Female range (165–255 Hz)")
    ax.axvline(255, color=PALETTE["female"], linestyle=":",  linewidth=1.2)

    ax.set_xlabel("Median F0 (Hz)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Fundamental Frequency (F0) Distribution by Gender",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()

    save_path = PLOTS_DIR / save_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 6. PCA 2D Scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_2d(
    X: np.ndarray,
    y: np.ndarray,
    save_name: str = "06_pca_scatter.png",
) -> Path:
    """
    Project 380-dim feature vectors onto 2 principal components.
    Good class separation in PCA space confirms feature discriminability.
    """
    scaler = StandardScaler()
    pca    = PCA(n_components=2, random_state=config.RANDOM_STATE)
    X_pca  = pca.fit_transform(scaler.fit_transform(X))

    var1, var2 = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    for label_idx, label_name in CLASS_MAP.items():
        mask = y == label_idx
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=PALETTE[label_name],
            label=label_name.capitalize(),
            alpha=0.6, s=40, edgecolors="none",
        )

    ax.set_xlabel(f"PC 1 ({var1:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC 2 ({var2:.1f}% variance)", fontsize=11)
    ax.set_title("PCA Projection of 380-D Feature Space",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.8)
    fig.tight_layout()

    save_path = PLOTS_DIR / save_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 7. Feature Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    X: np.ndarray,
    n_top: int = 20,
    save_name: str = "07_correlation_heatmap.png",
) -> Path:
    """
    Pearson correlation heatmap of the top *n_top* most-variance features.
    Reveals redundancy among features (e.g., adjacent MFCC coefficients).
    """
    # Select top-variance features
    variances  = np.var(X, axis=0)
    top_idx    = np.argsort(variances)[::-1][:n_top]
    X_top      = X[:, top_idx]

    labels = []
    for i in top_idx:
        if i < 80:
            labels.append(f"MFCC_{i+1}")
        elif i < 104:
            labels.append(f"Chr_{i-80+1}")
        elif i < 360:
            labels.append(f"Mel_{i-104+1}")
        elif i < 374:
            labels.append(f"SC_{i-360+1}")
        elif i < 376:
            labels.append("ZCR" if i == 374 else "ZCR_std")
        elif i < 378:
            labels.append("RMS" if i == 376 else "RMS_std")
        else:
            labels.append("Rol" if i == 378 else "Rol_std")

    corr = np.corrcoef(X_top.T)

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask,
        xticklabels=labels, yticklabels=labels,
        annot=False, cmap="coolwarm", center=0,
        vmin=-1, vmax=1, linewidths=0.3,
        ax=ax,
    )
    ax.set_title(f"Feature Correlation Matrix (top {n_top} by variance)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    save_path = PLOTS_DIR / save_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 8. Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_name: Optional[str] = None,
) -> Path:
    """
    Normalised confusion matrix with counts and percentages.
    """
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels  = [CLASS_MAP[i] for i in range(len(CLASS_MAP))]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_norm,
        annot=np.array([[f"{cm_norm[i,j]:.2f}\n({cm[i,j]})"
                         for j in range(len(labels))]
                        for i in range(len(labels))]),
        fmt="",
        xticklabels=labels, yticklabels=labels,
        cmap="Blues", vmin=0, vmax=1, linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name.replace('_',' ').title()}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    fname     = save_name or f"08_confusion_{model_name}.png"
    save_path = PLOTS_DIR / fname
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 9. ROC Curves (all models on one figure)
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    results: Dict[str, Dict],
    save_name: str = "09_roc_curves.png",
) -> Path:
    """
    Overlay ROC curves for all models.

    Parameters
    ----------
    results : dict  { model_name: {'fpr': array, 'tpr': array, 'auc': float} }
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, res), color in zip(results.items(), colors):
        label = (
            f"{name.replace('_',' ').title()} "
            f"(AUC = {res.get('auc_roc', res.get('auc', 0)):.3f})"
        )
        fpr = res.get("fpr", [0, 1])
        tpr = res.get("tpr", [0, 1])
        ax.plot(fpr, tpr, lw=2, color=color, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()

    save_path = PLOTS_DIR / save_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 10. Training History (CNN / LSTM)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(
    history: Dict,
    model_name: str = "CNN",
    save_name: Optional[str] = None,
) -> Path:
    """
    Loss and accuracy curves from Keras training history dict.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, metric, ylabel in zip(
        axes,
        [("loss", "val_loss"), ("accuracy", "val_accuracy")],
        ["Loss (Cross-Entropy)", "Accuracy"],
    ):
        train_key, val_key = metric
        if train_key in history:
            ax.plot(history[train_key], label="Train", color="#1a73e8", lw=2)
        if val_key in history:
            ax.plot(history[val_key], label="Validation",
                    color="#fbbc04", lw=2, linestyle="--")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=10)
        ax.set_title(ylabel, fontsize=11)

    model_title = model_name.replace("_", " ").upper()
    fig.suptitle(f"Training History — {model_title}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    fname     = save_name or f"10_training_history_{model_name}.png"
    save_path = PLOTS_DIR / fname
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 11. Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(
    results: Dict[str, Dict],
    save_name: str = "11_model_comparison.png",
) -> Path:
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, and F1
    across all trained models.
    """
    metrics  = ["accuracy", "precision", "recall", "f1"]
    m_labels = ["Accuracy", "Precision", "Recall", "F1"]
    models   = list(results.keys())
    n_models = len(models)
    n_metrics = len(metrics)

    x      = np.arange(n_models)
    width  = 0.18
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]

    fig, ax = plt.subplots(figsize=(max(8, n_models * 2), 5))

    for i, (metric, mlabel, color) in enumerate(zip(metrics, m_labels, colors)):
        values = [results[m].get(metric, 0) * 100 for m in models]
        bars = ax.bar(x + i * width - (n_metrics / 2 - 0.5) * width,
                      values, width, label=mlabel, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("_", "\n").title() for m in models],
        fontsize=10,
    )
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()

    save_path = PLOTS_DIR / save_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 12. Full visualization pipeline
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_visualizations(
    X_train: np.ndarray,
    y_train: np.ndarray,
    data_all: List[Tuple[str, int]],
    eval_results: Optional[Dict] = None,
    cnn_history: Optional[Dict] = None,
) -> List[Path]:
    """
    Run all static visualizations that depend only on the dataset and features.
    Evaluation plots (confusion matrix, ROC) require *eval_results*.

    Parameters
    ----------
    X_train    : feature matrix (n_samples, 380)
    y_train    : label vector
    data_all   : list of (path, label) for the full dataset
    eval_results : dict {model_name: metrics_dict} (optional)
    cnn_history  : Keras history dict for CNN (optional)

    Returns
    -------
    saved_paths : list of Path
    """
    saved = []

    # Pick one representative male and female path
    male_paths   = [p for p, l in data_all if l == 0]
    female_paths = [p for p, l in data_all if l == 1]

    if not male_paths or not female_paths:
        logger.warning("Not enough class samples to generate audio comparisons.")
    else:
        m_path = male_paths[0]
        f_path = female_paths[0]

        print("  [1/7] Waveform comparison …")
        saved.append(plot_waveform_comparison(m_path, f_path))

        print("  [2/7] Spectrogram comparison …")
        saved.append(plot_spectrogram_comparison(m_path, f_path))

        print("  [3/7] MFCC heatmap …")
        saved.append(plot_mfcc_heatmap(m_path, f_path))

        print("  [4/7] Pitch distribution …")
        try:
            saved.append(plot_pitch_distribution(data_all))
        except Exception as exc:
            logger.warning("Pitch plot failed: %s", exc)

    print("  [5/7] Feature distributions …")
    saved.append(plot_feature_distributions(X_train, y_train))

    print("  [6/7] PCA scatter …")
    saved.append(plot_pca_2d(X_train, y_train))

    print("  [7/7] Correlation heatmap …")
    saved.append(plot_correlation_heatmap(X_train))

    if eval_results:
        print("  [+]  Model comparison bar chart …")
        saved.append(plot_model_comparison(eval_results))

        print("  [+]  ROC curves …")
        saved.append(plot_roc_curves(eval_results))

        for name, res in eval_results.items():
            if "y_true" in res and "y_pred" in res:
                print(f"  [+]  Confusion matrix — {name} …")
                saved.append(plot_confusion_matrix(
                    np.array(res["y_true"]),
                    np.array(res["y_pred"]),
                    name,
                ))

    if cnn_history:
        print("  [+]  CNN training history …")
        saved.append(plot_training_history(cnn_history, model_name="CNN"))

    print(f"\n  All visualizations saved to: {PLOTS_DIR}")
    return [p for p in saved if p is not None]
