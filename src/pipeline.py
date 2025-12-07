import os
import json
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import SimpleITK as sitk

from .io_utils import read_image_auto, read_optional_mask, write_image
from .preprocess import normalize_intensity, histogram_match, n4_bias_correction, ensure_isotropic_spacing
from .registration import register_rigid_affine, resample_moving_to_fixed
from .evaluate import save_overlay, save_deformation_field_magnitude, write_metrics_json, dice_coefficient, compute_intensity_metrics


def ensure_dirs(base: str) -> None:
    subdirs = ["preprocessed", "transforms", "registered", "eval"]
    for sd in subdirs:
        os.makedirs(os.path.join(base, sd), exist_ok=True)


def _parse_spacing(spacing: Optional[Any], dimension: int) -> Tuple[float, ...]:
    if spacing is None:
        values = [1.0]
    elif isinstance(spacing, str):
        values = [float(s.strip()) for s in spacing.split(",") if s.strip()]
    elif isinstance(spacing, (list, tuple)):
        values = [float(v) for v in spacing]
    else:
        values = [1.0]

    if not values:
        values = [1.0]

    if len(values) == 1:
        values = values * dimension
    elif len(values) < dimension:
        values = values + [values[-1]] * (dimension - len(values))
    else:
        values = values[:dimension]
    return tuple(values)


def _resize_2d_image(img: sitk.Image, target_size: int, interpolator) -> sitk.Image:
    if img.GetDimension() != 2:
        return img
    target_size = int(target_size)
    if target_size <= 0:
        return img
    current_size = img.GetSize()
    if current_size[0] == target_size and current_size[1] == target_size:
        return img
    spacing = img.GetSpacing()
    new_spacing = [spacing[0] * current_size[0] / target_size, spacing[1] * current_size[1] / target_size]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([target_size, target_size])
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(img)


def _resize_optional_image(img: Optional[sitk.Image], target_size: int, interpolator) -> Optional[sitk.Image]:
    if img is None:
        return None
    return _resize_2d_image(img, target_size, interpolator)


def _squeeze_image(img: sitk.Image) -> sitk.Image:
    while img.GetDimension() > 2 and 1 in img.GetSize():
        size = list(img.GetSize())
        axis = size.index(1)
        size[axis] = 0
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize(size)
        extractor.SetIndex([0] * img.GetDimension())
        img = extractor.Execute(img)
    return img


def _squeeze_optional_image(img: Optional[sitk.Image]) -> Optional[sitk.Image]:
    if img is None:
        return None
    return _squeeze_image(img)


@dataclass
class PipelineConfig:
    fixed_path: str
    moving_path: str
    fixed_mask_path: Optional[str] = None
    moving_mask_path: Optional[str] = None
    outdir: Optional[str] = None
    target_spacing: Optional[Any] = "1,1,1"
    histogram_matching: bool = True
    fixed_series_uid: Optional[str] = None
    moving_series_uid: Optional[str] = None
    save_outputs: bool = True
    force_size_2d: Optional[int] = None
    crop_2d_to_moving: bool = False
    overlay_cmap: str = "Reds"


@dataclass
class PipelineResult:
    metrics: Dict[str, float]
    output_dir: str
    overlay_path: str
    deformation_path: str
    registered_image_path: str
    rigid_transform_path: str
    affine_transform_path: str
    fixed_image: sitk.Image
    registered_image: sitk.Image
    fixed_series_ids: Optional[List[str]]
    moving_series_ids: Optional[List[str]]
    crop_2d_to_moving: bool


def run_registration_pipeline(config: PipelineConfig) -> PipelineResult:
    work_dir = config.outdir or tempfile.mkdtemp(prefix="mmreg_")
    ensure_dirs(work_dir)

    fixed, fixed_series = read_image_auto(config.fixed_path, series_uid=config.fixed_series_uid, pixel_type=sitk.sitkFloat32)
    moving, moving_series = read_image_auto(config.moving_path, series_uid=config.moving_series_uid, pixel_type=sitk.sitkFloat32)

    fixed = _squeeze_image(fixed)
    moving = _squeeze_image(moving)

    fixed_mask = read_optional_mask(config.fixed_mask_path)
    moving_mask = read_optional_mask(config.moving_mask_path)
    fixed_mask = _squeeze_optional_image(fixed_mask)
    moving_mask = _squeeze_optional_image(moving_mask)

    if fixed.GetDimension() != moving.GetDimension():
        raise ValueError("Fixed and moving images must have the same dimensionality")

    spacing = _parse_spacing(config.target_spacing, fixed.GetDimension())
    fixed_iso = ensure_isotropic_spacing(fixed, target_spacing=spacing, interp=sitk.sitkLinear)
    moving_iso = ensure_isotropic_spacing(moving, target_spacing=spacing, interp=sitk.sitkLinear)

    if config.force_size_2d and fixed_iso.GetDimension() == 2:
        fixed_iso = _resize_2d_image(fixed_iso, config.force_size_2d, sitk.sitkLinear)
        moving_iso = _resize_2d_image(moving_iso, config.force_size_2d, sitk.sitkLinear)
        fixed_mask = _resize_optional_image(fixed_mask, config.force_size_2d, sitk.sitkNearestNeighbor)
        moving_mask = _resize_optional_image(moving_mask, config.force_size_2d, sitk.sitkNearestNeighbor)

    fixed_norm = normalize_intensity(fixed_iso, method="zscore")
    moving_norm = normalize_intensity(moving_iso, method="zscore")

    if config.histogram_matching:
        moving_norm = histogram_match(moving_norm, fixed_norm)

    fixed_n4 = n4_bias_correction(fixed_norm, fixed_mask)
    moving_n4 = n4_bias_correction(moving_norm, moving_mask)

    fixed_pre_path = os.path.join(work_dir, "preprocessed", "fixed_preprocessed.nii.gz")
    moving_pre_path = os.path.join(work_dir, "preprocessed", "moving_preprocessed.nii.gz")
    write_image(fixed_n4, fixed_pre_path)
    write_image(moving_n4, moving_pre_path)

    rigid_tfm, affine_tfm = register_rigid_affine(fixed_n4, moving_n4, fixed_mask, moving_mask)

    rigid_path = os.path.join(work_dir, "transforms", "rigid.tfm")
    affine_path = os.path.join(work_dir, "transforms", "affine.tfm")
    sitk.WriteTransform(rigid_tfm, rigid_path)
    sitk.WriteTransform(affine_tfm, affine_path)

    moving_reg = resample_moving_to_fixed(moving_n4, fixed_n4, affine_tfm, interp=sitk.sitkLinear)
    registered_path = os.path.join(work_dir, "registered", "moving_registered_to_fixed.nii.gz")
    write_image(moving_reg, registered_path)

    metrics: Dict[str, float] = {}
    metrics.update(compute_intensity_metrics(fixed_n4, moving_reg))
    if fixed_mask is not None and moving_mask is not None:
        moving_mask_reg = resample_moving_to_fixed(moving_mask, fixed_n4, affine_tfm, interp=sitk.sitkNearestNeighbor)
        metrics["dice_coefficient"] = float(dice_coefficient(fixed_mask, moving_mask_reg))

    overlay_path = os.path.join(work_dir, "eval", "overlay_mid_slice.png")
    deformation_path = os.path.join(work_dir, "eval", "deformation_magnitude.png")
    metrics_path = os.path.join(work_dir, "eval", "metrics.json")

    crop_overlay = bool(config.crop_2d_to_moving and fixed_n4.GetDimension() == 2)
    save_overlay(
        fixed_n4,
        moving_reg,
        overlay_path,
        crop_to_moving=crop_overlay,
        overlay_cmap=config.overlay_cmap,
    )
    save_deformation_field_magnitude(affine_tfm, fixed_n4, deformation_path)
    write_metrics_json(metrics, metrics_path)

    return PipelineResult(
        metrics=metrics,
        output_dir=work_dir,
        overlay_path=overlay_path,
        deformation_path=deformation_path,
        registered_image_path=registered_path,
        rigid_transform_path=rigid_path,
        affine_transform_path=affine_path,
        fixed_image=fixed_n4,
        registered_image=moving_reg,
        fixed_series_ids=fixed_series,
        moving_series_ids=moving_series,
        crop_2d_to_moving=config.crop_2d_to_moving,
    )