import json
from typing import Optional, Dict, Tuple
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def dice_coefficient(mask_fixed: sitk.Image, mask_moving_resampled: sitk.Image) -> float:
    fixed_arr = sitk.GetArrayFromImage(mask_fixed) > 0
    moving_arr = sitk.GetArrayFromImage(mask_moving_resampled) > 0
    intersection = np.logical_and(fixed_arr, moving_arr).sum()
    size_sum = fixed_arr.sum() + moving_arr.sum()
    return 2.0 * intersection / size_sum if size_sum > 0 else 0.0


def _extract_slice_pair(fixed: sitk.Image, moving_resampled: sitk.Image, slice_index: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    fixed_arr = sitk.GetArrayFromImage(fixed)
    mov_arr = sitk.GetArrayFromImage(moving_resampled)
    if fixed_arr.ndim == 2:
        return fixed_arr, mov_arr
    depth = fixed_arr.shape[0]
    if slice_index is None:
        slice_index = depth // 2
    return fixed_arr[slice_index], mov_arr[slice_index]


def _crop_arrays_to_moving_extent(fixed_arr: np.ndarray, moving_arr: np.ndarray, min_margin: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    abs_mov = np.abs(moving_arr)
    if abs_mov.max() <= 0:
        return fixed_arr, moving_arr
    threshold = 0.01 * abs_mov.max()
    mask = abs_mov > threshold
    if not mask.any():
        return fixed_arr, moving_arr
    rows, cols = np.nonzero(mask)
    min_r = max(rows.min() - min_margin, 0)
    max_r = min(rows.max() + min_margin + 1, moving_arr.shape[0])
    min_c = max(cols.min() - min_margin, 0)
    max_c = min(cols.max() + min_margin + 1, moving_arr.shape[1])
    return fixed_arr[min_r:max_r, min_c:max_c], moving_arr[min_r:max_r, min_c:max_c]


def save_overlay(
    fixed: sitk.Image,
    moving_resampled: sitk.Image,
    out_png: str,
    slice_index: Optional[int] = None,
    crop_to_moving: bool = False,
    overlay_cmap: str = "Reds",
) -> None:
    f, m = _extract_slice_pair(fixed, moving_resampled, slice_index)
    if crop_to_moving:
        f, m = _crop_arrays_to_moving_extent(f, m)
    plt.figure(figsize=(6, 6))
    plt.imshow(f, cmap='gray')
    plt.imshow(m, cmap=overlay_cmap, alpha=0.4)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def display_overlay(
    fixed: sitk.Image,
    moving_resampled: sitk.Image,
    slice_index: Optional[int] = None,
    crop_to_moving: bool = False,
    overlay_cmap: str = "jet",
) -> None:
    f, m = _extract_slice_pair(fixed, moving_resampled, slice_index)
    if crop_to_moving:
        f, m = _crop_arrays_to_moving_extent(f, m)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title('Fixed')
    plt.imshow(f, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Registered Moving')
    plt.imshow(m, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Overlay')
    plt.imshow(f, cmap='gray')
    plt.imshow(m, cmap=overlay_cmap, alpha=0.4)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def save_deformation_field_magnitude(transform: sitk.Transform, fixed: sitk.Image, out_png: str) -> None:
    # For non-deformable transforms this will be near zero; included for pipeline completeness.
    grid = sitk.TransformToDisplacementField(transform, sitk.sitkVectorFloat64, fixed.GetSize(), fixed.GetOrigin(), fixed.GetSpacing(), fixed.GetDirection())
    mag = sitk.VectorMagnitude(grid)
    arr = sitk.GetArrayFromImage(mag)
    if arr.ndim == 2:
        slice_data = arr
    else:
        slice_index = arr.shape[0] // 2
        slice_data = arr[slice_index]
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_data, cmap='viridis')
    plt.colorbar(label='Displacement Magnitude')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def write_metrics_json(metrics: Dict, out_path: str) -> None:
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def compute_intensity_metrics(fixed: sitk.Image, moving_resampled: sitk.Image) -> Dict[str, float]:
    f = sitk.GetArrayFromImage(fixed).astype(np.float64)
    m = sitk.GetArrayFromImage(moving_resampled).astype(np.float64)
    if f.shape != m.shape:
        raise ValueError("Intensity metrics require images of matching shape")
    f_flat = f.ravel()
    m_flat = m.ravel()
    mse = float(np.mean((f_flat - m_flat) ** 2))
    mae = float(np.mean(np.abs(f_flat - m_flat)))
    f_mean = np.mean(f_flat)
    m_mean = np.mean(m_flat)
    f_std = np.std(f_flat) if np.std(f_flat) > 0 else 1.0
    m_std = np.std(m_flat) if np.std(m_flat) > 0 else 1.0
    ncc = float(np.mean(((f_flat - f_mean) / f_std) * ((m_flat - m_mean) / m_std)))
    return {"mse": mse, "mae": mae, "ncc": ncc}
