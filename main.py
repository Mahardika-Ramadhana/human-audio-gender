#!/usr/bin/env python3
"""
Human Audio Gender Recognition — Main Orchestration Script
===========================================================

Usage
-----
  python main.py --mode download                        # Download RAVDESS dataset from Zenodo
  python main.py --mode extract                         # Extract audio features
  python main.py --mode train                           # Train all models
  python main.py --mode evaluate                        # Evaluate all models
  python main.py --mode predict --audio path/to/file.wav
  python main.py --mode full                            # Run full pipeline end-to-end
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── Ensure project root is on path ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s — %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('main')


# ─────────────────────────────────────────────────────────────────────────────
# ASCII header
# ─────────────────────────────────────────────────────────────────────────────

HEADER = r"""
╔══════════════════════════════════════════════════════════════╗
║       Human Audio Gender Recognition System v1.0            ║
║       ML + Deep Learning Pipeline                           ║
╚══════════════════════════════════════════════════════════════╝
"""

def print_header() -> None:
    print(HEADER)


def print_section(title: str) -> None:
    bar = '─' * 60
    print(f'\n{bar}')
    print(f'  {title}')
    print(f'{bar}')


def print_summary_table(results: dict) -> None:
    """Print a formatted table of model evaluation results."""
    print_section('Final Results Summary')
    header = f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10}"
    print(header)
    print('─' * len(header))
    for name, metrics in results.items():
        acc  = metrics.get('accuracy',  float('nan'))
        prec = metrics.get('precision', float('nan'))
        rec  = metrics.get('recall',    float('nan'))
        f1   = metrics.get('f1',        float('nan'))
        auc  = metrics.get('auc_roc',   float('nan'))
        print(
            f"{name.replace('_', ' ').title():<22}"
            f"  {acc:>8.4f}  {prec:>9.4f}  {rec:>8.4f}  {f1:>8.4f}  {auc:>9.4f}"
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Generate synthetic data
# ─────────────────────────────────────────────────────────────────────────────

def run_download(force: bool = False) -> None:
    print_section('Step 1: Downloading RAVDESS Dataset (Zenodo)')
    from download_dataset import download_ravdess
    t0 = time.time()
    download_ravdess(force_download=force)
    elapsed = time.time() - t0
    print(f'\n  Download complete in {elapsed:.1f}s')
    print(f'  Output: {config.DATA_RAW_DIR}')


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def run_extract() -> None:
    print_section('Step 2: Feature Extraction')
    from src.data_loader import DataLoader
    from src.features import FeatureExtractor

    t0 = time.time()
    loader    = DataLoader()
    extractor = FeatureExtractor()

    print(f'  Scanning data directory: {config.DATA_RAW_DIR}')
    all_data = loader.load_dataset(config.DATA_RAW_DIR)
    print(f'  Total audio files found: {len(all_data)}')

    train_data, val_data, test_data = loader.split_dataset(all_data)
    print(f'  Split — train: {len(train_data)}  val: {len(val_data)}  test: {len(test_data)}')

    # Flat features (for sklearn models)
    print('\n  [Flat features — traditional ML]')
    extractor.extract_and_save(train_data, config.FEATURES_TRAIN_PATH,
                               mode='flat', desc='  Train (flat)')
    extractor.extract_and_save(val_data,   config.FEATURES_VAL_PATH,
                               mode='flat', desc='  Val   (flat)')
    extractor.extract_and_save(test_data,  config.FEATURES_TEST_PATH,
                               mode='flat', desc='  Test  (flat)')

    # CNN features (mel spectrograms)
    print('\n  [CNN features — mel spectrograms]')
    extractor.extract_and_save(train_data, config.CNN_FEATURES_TRAIN_PATH,
                               mode='cnn', desc='  Train (cnn) ')
    extractor.extract_and_save(val_data,   config.CNN_FEATURES_VAL_PATH,
                               mode='cnn', desc='  Val   (cnn) ')
    extractor.extract_and_save(test_data,  config.CNN_FEATURES_TEST_PATH,
                               mode='cnn', desc='  Test  (cnn) ')

    # Save CNN normalisation stats from training data
    import numpy as np
    cnn_train_features, _ = loader.load_features(config.CNN_FEATURES_TRAIN_PATH)
    _min = float(cnn_train_features.min())
    _max = float(cnn_train_features.max())
    stats_path = config.MODELS_DIR / 'cnn_norm_stats.npz'
    np.savez(str(stats_path), min=_min, max=_max)
    print(f'\n  CNN norm stats saved → {stats_path}')

    elapsed = time.time() - t0
    print(f'\n  Feature extraction complete in {elapsed:.1f}s')


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Training
# ─────────────────────────────────────────────────────────────────────────────

def run_train() -> dict:
    print_section('Step 3: Training All Models')
    from src.train import TrainingPipeline

    t0       = time.time()
    pipeline = TrainingPipeline()
    metrics  = pipeline.train_all_models()
    elapsed  = time.time() - t0

    print(f'\n  Training complete in {elapsed:.1f}s')
    print(f'  Training log → {config.TRAINING_LOG_PATH}')

    print_section('Training Results (Validation Set)')
    for name, m in metrics.items():
        print(
            f"  {name.replace('_', ' ').title():<25}"
            f"train_acc={m.get('train_acc', 0):.4f}  "
            f"val_acc={m.get('val_acc', 0):.4f}  "
            f"time={m.get('training_time_s', 0):.1f}s"
        )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluate() -> dict:
    print_section('Step 4: Evaluating All Models')
    import numpy as np
    from src.data_loader import DataLoader
    from src.models import ModelManager
    from src.evaluate import Evaluator

    loader  = DataLoader()
    manager = ModelManager()
    evaluator = Evaluator()

    X_test, y_test = loader.load_features(config.FEATURES_TEST_PATH)
    print(f'  Test set: {len(y_test)} samples')

    # Load CNN test features
    cnn_features_exist = config.CNN_FEATURES_TEST_PATH.exists()
    X_cnn_test = y_cnn_test = None
    if cnn_features_exist:
        X_cnn_test, y_cnn_test = loader.load_features(config.CNN_FEATURES_TEST_PATH)
        # Normalise CNN features
        stats_path = config.MODELS_DIR / 'cnn_norm_stats.npz'
        if stats_path.exists():
            stats = np.load(str(stats_path))
            _min, _max = float(stats['min']), float(stats['max'])
        else:
            _min, _max = X_cnn_test.min(), X_cnn_test.max()
        X_cnn_test = (X_cnn_test - _min) / (_max - _min + 1e-8)

    all_results = {}
    saved_models = manager.get_model_summary()

    for name, info in saved_models.items():
        if name.endswith('_best'):
            continue  # skip checkpoint copies

        try:
            model = manager.load_model(name)
        except Exception as exc:
            logger.warning("Could not load model '%s': %s", name, exc)
            continue

        is_keras = info['type'] == 'keras'
        if is_keras and X_cnn_test is None:
            logger.warning("CNN test features not available — skipping '%s'.", name)
            continue

        if is_keras:
            if 'lstm' in name:
                X_eval = X_cnn_test[..., 0].transpose(0, 2, 1)  # (N, T, F)
            else:
                X_eval = X_cnn_test[..., np.newaxis] if X_cnn_test.ndim == 3 else X_cnn_test
            y_eval = y_cnn_test
        else:
            X_eval = X_test
            y_eval = y_test

        print(f'\n  Evaluating: {name}')
        result = evaluator.evaluate_model(model, X_eval, y_eval, name, is_keras=is_keras)
        all_results[name] = result

        report = evaluator.generate_classification_report(
            y_eval,
            model.predict(X_eval) if not is_keras else np.argmax(model.predict(X_eval, verbose=0), axis=1),
            model_name=name,
        )
        print(report)

        # Plot training history for keras models
        log_path = config.TRAINING_LOG_PATH
        if is_keras and log_path.exists():
            with open(log_path) as fh:
                log = json.load(fh)
            clean_name = name.replace('gender_', '')
            if clean_name in log and 'history' in log.get(clean_name, {}):
                evaluator.plot_training_history(
                    log[clean_name]['history'], model_name=clean_name
                )

    if all_results:
        evaluator.compare_models(all_results)
        print_summary_table(all_results)

        # Save evaluation results
        results_path = config.MODELS_DIR / 'evaluation_results.json'
        with open(results_path, 'w') as fh:
            json.dump(all_results, fh, indent=2, default=str)
        print(f'  Evaluation results saved → {results_path}')
        print(f'  Plots saved → {config.PLOTS_DIR}')
    else:
        print('  No models found. Run training first: python main.py --mode train')

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Single file prediction
# ─────────────────────────────────────────────────────────────────────────────

def run_predict(audio_path: str, model_name: str = 'svm') -> None:
    print_section(f'Predicting: {audio_path}')
    from src.predict import AudioPredictor

    predictor = AudioPredictor(model_name=model_name)

    result = predictor.predict(audio_path)
    print(f'\n  File       : {audio_path}')
    print(f'  Prediction : {result["class"].upper()}')
    print(f'  Confidence : {result["confidence"]:.1%}')
    print('  Probabilities:')
    for cls, prob in result['probabilities'].items():
        bar = '█' * int(prob * 30) + '░' * (30 - int(prob * 30))
        print(f'    {cls:6s}  {bar}  {prob:.1%}')

    # Also save visualisation
    try:
        vis_path = predictor.visualize_prediction(audio_path)
        print(f'\n  Visualisation saved → {vis_path}')
    except Exception as exc:
        logger.warning("Could not create visualisation: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_visualize() -> None:
    """Generate all visualizations from extracted features + dataset."""
    print_section('Visualizations — Pengenalan Pola')
    import numpy as np
    from src.data_loader import DataLoader
    from src.visualize import generate_all_visualizations

    loader   = DataLoader()
    data_all = loader.load_dataset(config.DATA_RAW_DIR)

    if not config.FEATURES_TRAIN_PATH.exists():
        print('  Feature files not found. Run --mode extract first.')
        return

    X_train, y_train = loader.load_features(config.FEATURES_TRAIN_PATH)
    print(f'  Training samples : {len(y_train)}')
    print(f'  Feature dim      : {X_train.shape[1]}')

    # Load eval results if available
    eval_results = None
    results_path = config.MODELS_DIR / 'evaluation_results.json'
    if results_path.exists():
        with open(results_path) as fh:
            eval_results = json.load(fh)

    # Load CNN history if available
    cnn_history = None
    if config.TRAINING_LOG_PATH.exists():
        with open(config.TRAINING_LOG_PATH) as fh:
            log = json.load(fh)
        cnn_history = log.get('cnn', {}).get('history')

    t0 = time.time()
    saved = generate_all_visualizations(
        X_train, y_train, data_all, eval_results, cnn_history
    )
    elapsed = time.time() - t0

    print(f'\n  Generated {len(saved)} plots in {elapsed:.1f}s')
    print(f'  Saved to: {config.PLOTS_DIR}')
    for p in saved:
        print(f'    {p.name}')


def run_full_pipeline() -> None:
    print_section('Running Full Pipeline')
    run_download()
    run_extract()
    run_train()
    run_evaluate()
    run_visualize()
    print_section('Full Pipeline Complete')
    print(f'  Models saved to  : {config.MODELS_DIR}')
    print(f'  Plots saved to   : {config.PLOTS_DIR}')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Human Audio Gender Recognition — Pengenalan Pola',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--mode',
        choices=['download', 'extract', 'train', 'evaluate',
                 'visualize', 'predict', 'full'],
        required=True,
        help='Pipeline stage to run.',
    )
    parser.add_argument(
        '--audio',
        type=str,
        default=None,
        help='Path to audio file (required for --mode predict).',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='svm',
        help='Model name to use for prediction (default: svm).',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if dataset exists (for --mode download).',
    )
    return parser.parse_args()


def main() -> None:
    print_header()
    args = parse_args()

    t_start = time.time()

    if args.mode == 'download':
        run_download(force=args.force)

    elif args.mode == 'extract':
        run_extract()

    elif args.mode == 'train':
        run_train()

    elif args.mode == 'evaluate':
        run_evaluate()

    elif args.mode == 'visualize':
        run_visualize()

    elif args.mode == 'predict':
        if args.audio is None:
            print('  ERROR: --audio <path> is required for predict mode.')
            sys.exit(1)
        run_predict(args.audio, model_name=args.model)

    elif args.mode == 'full':
        run_full_pipeline()

    total = time.time() - t_start
    print(f'\n  Total wall-clock time: {total:.1f}s\n')


if __name__ == '__main__':
    main()
