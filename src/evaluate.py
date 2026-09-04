"""
Evaluation utilities for Human Audio Gender Recognition.

Provides:
- Per-model metric computation (accuracy, precision, recall, F1, AUC-ROC)
- Confusion matrix plot
- ROC curve plot
- Training history plot (Keras)
- Multi-model comparison bar chart
- Textual classification report
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from src import config

logger = logging.getLogger(__name__)

# Consistent plot style
sns.set_theme(style='whitegrid', palette='muted')
FIGSIZE_DEFAULT = (8, 6)


class Evaluator:
    """
    Centralised evaluation and visualisation for all trained models.
    """

    def __init__(self, plots_dir: Path = config.PLOTS_DIR):
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    # Core metric computation
    # ──────────────────────────────────────────────────────────────────────

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute a comprehensive set of classification metrics.

        Parameters
        ----------
        y_true  : int array, shape (n,)
        y_pred  : int array, shape (n,)
        y_proba : float array, shape (n, n_classes) or (n,), optional

        Returns
        -------
        metrics : dict
            accuracy, precision, recall, f1, auc_roc (if y_proba given)
        """
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_true, y_pred,    average='weighted', zero_division=0)
        f1   = f1_score(y_true, y_pred,        average='weighted', zero_division=0)

        metrics = {
            'accuracy':  round(float(acc),  4),
            'precision': round(float(prec), 4),
            'recall':    round(float(rec),  4),
            'f1':        round(float(f1),   4),
        }

        if y_proba is not None:
            try:
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
                metrics['auc_roc'] = round(float(auc), 4)
            except Exception as exc:
                logger.warning("Could not compute AUC-ROC: %s", exc)
                metrics['auc_roc'] = float('nan')

        return metrics

    # ──────────────────────────────────────────────────────────────────────
    # Full model evaluation
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        is_keras: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference on *X_test* and return metrics + generate plots.

        Parameters
        ----------
        model      : sklearn Pipeline or tf.keras.Model
        X_test     : np.ndarray
        y_test     : np.ndarray
        model_name : str
        is_keras   : bool
            Set True for Keras models to use model.predict() correctly.

        Returns
        -------
        results : dict
        """
        # ── Predictions ───────────────────────────────────────────────
        if is_keras:
            proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(proba, axis=1)
            y_proba = proba
        else:
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
            else:
                y_proba = None

        # ── Metrics ───────────────────────────────────────────────────
        metrics = self.compute_metrics(y_test, y_pred, y_proba)

        logger.info(
            "[%s] acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
            model_name,
            metrics['accuracy'], metrics['precision'],
            metrics['recall'],   metrics['f1'],
            metrics.get('auc_roc', float('nan')),
        )

        # ── Plots ─────────────────────────────────────────────────────
        self.plot_confusion_matrix(y_test, y_pred, model_name)
        if y_proba is not None:
            self.plot_roc_curve(y_test, y_proba, model_name)

        return {
            'model_name': model_name,
            **metrics,
            'n_test_samples': len(y_test),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Confusion matrix
    # ──────────────────────────────────────────────────────────────────────

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
    ) -> Path:
        """
        Plot and save a confusion matrix heatmap.

        Returns
        -------
        save_path : Path
        """
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=config.CLASSES,
            yticklabels=config.CLASSES,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label',      fontsize=12)
        ax.set_title(f'Confusion Matrix — {model_name.replace("_", " ").title()}', fontsize=14)
        fig.tight_layout()

        save_path = self.plots_dir / f"confusion_matrix_{model_name}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Confusion matrix saved → %s", save_path)
        return save_path

    # ──────────────────────────────────────────────────────────────────────
    # ROC curve
    # ──────────────────────────────────────────────────────────────────────

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str,
    ) -> Path:
        """
        Plot and save an ROC curve (binary classification).

        Returns
        -------
        save_path : Path
        """
        if y_proba.ndim == 2:
            scores = y_proba[:, 1]
        else:
            scores = y_proba

        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
        except Exception as exc:
            logger.warning("ROC curve could not be computed for %s: %s", model_name, exc)
            return None

        fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
        ax.plot(fpr, tpr, lw=2, color='steelblue',
                label=f'ROC (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
        ax.fill_between(fpr, tpr, alpha=0.15, color='steelblue')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate',  fontsize=12)
        ax.set_title(f'ROC Curve — {model_name.replace("_", " ").title()}', fontsize=14)
        ax.legend(loc='lower right', fontsize=11)
        fig.tight_layout()

        save_path = self.plots_dir / f"roc_curve_{model_name}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("ROC curve saved → %s", save_path)
        return save_path

    # ──────────────────────────────────────────────────────────────────────
    # Training history (Keras)
    # ──────────────────────────────────────────────────────────────────────

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        model_name: str,
    ) -> Path:
        """
        Plot accuracy and loss curves from Keras training history.

        Parameters
        ----------
        history : dict
            history.history from model.fit() or equivalent dict.
        model_name : str

        Returns
        -------
        save_path : Path
        """
        # Support both raw Keras History object and plain dict
        if hasattr(history, 'history'):
            hist = history.history
        else:
            hist = history

        epochs = range(1, len(hist['loss']) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # ── Accuracy ──────────────────────────────────────────────────
        ax1.plot(epochs, hist['accuracy'],     lw=2, label='Train accuracy', color='steelblue')
        ax1.plot(epochs, hist['val_accuracy'], lw=2, label='Val accuracy',   color='coral',
                 linestyle='--')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title(f'{model_name.upper()} — Accuracy', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # ── Loss ──────────────────────────────────────────────────────
        ax2.plot(epochs, hist['loss'],     lw=2, label='Train loss', color='steelblue')
        ax2.plot(epochs, hist['val_loss'], lw=2, label='Val loss',   color='coral',
                 linestyle='--')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title(f'{model_name.upper()} — Loss', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f'Training History — {model_name.replace("_", " ").title()}', fontsize=15)
        fig.tight_layout()

        save_path = self.plots_dir / f"training_history_{model_name}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Training history plot saved → %s", save_path)
        return save_path

    # ──────────────────────────────────────────────────────────────────────
    # Multi-model comparison
    # ──────────────────────────────────────────────────────────────────────

    def compare_models(
        self,
        results_dict: Dict[str, Dict[str, float]],
    ) -> Path:
        """
        Bar chart comparing accuracy, F1, and AUC-ROC across all models.

        Parameters
        ----------
        results_dict : dict
            {model_name: {'accuracy': ..., 'f1': ..., 'auc_roc': ...}}

        Returns
        -------
        save_path : Path
        """
        model_names = list(results_dict.keys())
        metrics_keys = ['accuracy', 'f1', 'auc_roc']
        metric_labels = ['Accuracy', 'F1 Score', 'AUC-ROC']
        colors = ['#2196F3', '#4CAF50', '#FF9800']

        x = np.arange(len(model_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, (key, label, color) in enumerate(zip(metrics_keys, metric_labels, colors)):
            values = [
                results_dict[m].get(key, 0.0) for m in model_names
            ]
            bars = ax.bar(x + i * width, values, width,
                          label=label, color=color, alpha=0.85, edgecolor='white')
            # Value labels on bars
            for bar, val in zip(bars, values):
                if val and not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                    )

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Comparison — Test Set Performance', fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels(
            [n.replace('_', '\n').title() for n in model_names],
            fontsize=10,
        )
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()

        save_path = self.plots_dir / "model_comparison.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Model comparison chart saved → %s", save_path)
        return save_path

    # ──────────────────────────────────────────────────────────────────────
    # Text report
    # ──────────────────────────────────────────────────────────────────────

    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = '',
    ) -> str:
        """
        Return a formatted scikit-learn classification report string.

        Parameters
        ----------
        y_true     : int array
        y_pred     : int array
        model_name : str (optional, used in header)

        Returns
        -------
        report : str
        """
        header = f"\n{'='*60}\nClassification Report — {model_name}\n{'='*60}\n"
        report = classification_report(
            y_true,
            y_pred,
            target_names=config.CLASSES,
            zero_division=0,
        )
        full_report = header + report
        logger.info(full_report)
        return full_report
