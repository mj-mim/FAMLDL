# metrics.py — AuPRC and FPR@95 for anomaly segmentation evaluation
import os, glob
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_curve


def load_pairs(scores_dir, masks_dir, method, mask_ext=None):
    """
    Load matching score maps (.npy) and ground-truth masks.

    Mask convention (SMIYC / FS datasets):
      255  → void / ignore
      > 0  → anomaly  (positive)
      0    → normal   (negative)
    """
    npy_files = sorted(glob.glob(os.path.join(scores_dir, f'*_{method}.npy')))
    if not npy_files:
        raise FileNotFoundError(
            f'No *_{method}.npy files in {scores_dir}')

    # Auto-detect mask extension if not given
    if mask_ext is None:
        for ext in ('png', 'jpg', 'jpeg', 'webp'):
            sample = os.path.join(
                masks_dir,
                os.path.basename(npy_files[0]).replace(f'_{method}.npy', f'.{ext}'))
            if os.path.isfile(sample):
                mask_ext = ext
                break
        if mask_ext is None:
            mask_ext = 'png'

    S, L = [], []
    for npy_path in npy_files:
        stem      = os.path.basename(npy_path).replace(f'_{method}.npy', '')
        mask_path = os.path.join(masks_dir, f'{stem}.{mask_ext}')

        if not os.path.isfile(mask_path):
            print(f'  [SKIP] mask not found: {mask_path}')
            continue

        score = np.load(npy_path).astype(np.float32)
        mask  = np.array(Image.open(mask_path))

        # Resize score to mask resolution if they differ
        if score.shape != mask.shape:
            score = np.array(
                Image.fromarray(score).resize(
                    (mask.shape[1], mask.shape[0]), Image.BILINEAR))

        valid = mask != 255
        if valid.sum() == 0:
            print(f'  [SKIP] all void: {mask_path}')
            continue

        S.append(score[valid].ravel())
        L.append((mask[valid] > 0).ravel().astype(np.int32))

    if not S:
        raise RuntimeError(f'No valid (score, mask) pairs loaded from {scores_dir}')

    return np.concatenate(S), np.concatenate(L)


def compute_auprc(scores, labels):
    """Area under Precision-Recall Curve (0–1)."""
    return float(average_precision_score(labels, scores))


def fpr95(scores, labels):
    """False Positive Rate at 95 % True Positive Rate (0–1)."""
    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr_arr, 0.95)
    return float(fpr_arr[min(idx, len(fpr_arr) - 1)])
