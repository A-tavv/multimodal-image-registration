import SimpleITK as sitk
from typing import Optional, Any


def n4_bias_correction(
    img: sitk.Image,
    mask: Optional[sitk.Image] = None,
    conv_thresh: float = 0.001,
    max_iters=(50, 50, 30, 20),
) -> sitk.Image:
    """Run N4 bias correction without shrinking to avoid zero-spacing issues on thin slices."""
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetConvergenceThreshold(conv_thresh)
    corrector.SetMaximumNumberOfIterations(list(max_iters))
    if mask is None:
        mask = sitk.OtsuThreshold(img, 0, 1)
    return corrector.Execute(img, mask)


def normalize_intensity(img: sitk.Image, method: str = "zscore") -> sitk.Image:
    stats = sitk.StatisticsImageFilter()
    stats.Execute(img)
    if method == "zscore":
        mean = stats.GetMean()
        std = stats.GetSigma() if stats.GetSigma() > 0 else 1.0
        return sitk.ShiftScale(img, -mean, 1.0 / std)
    elif method == "minmax":
        min_val = stats.GetMinimum()
        max_val = stats.GetMaximum()
        scale = 1.0 / (max_val - min_val) if max_val > min_val else 1.0
        return sitk.ShiftScale(img, -min_val, scale)
    else:
        return img


def histogram_match(moving: sitk.Image, fixed: sitk.Image, bins: int = 64, points: int = 5000, threshold: float = 0.01) -> sitk.Image:
    hm = sitk.HistogramMatchingImageFilter()
    hm.SetNumberOfHistogramLevels(bins)
    hm.SetNumberOfMatchPoints(points)
    hm.ThresholdAtMeanIntensityOn()
    return hm.Execute(moving, fixed)


def ensure_isotropic_spacing(img: sitk.Image, target_spacing=(1.0,), interp: Any = sitk.sitkLinear) -> sitk.Image:
    dimension = img.GetDimension()
    if len(target_spacing) == 1:
        spacing = target_spacing * dimension
    elif len(target_spacing) < dimension:
        spacing = target_spacing + target_spacing[-1:] * (dimension - len(target_spacing))
    else:
        spacing = target_spacing[:dimension]

    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_size = [int(round(osz * osp / tsp)) for osz, osp, tsp in zip(original_size, original_spacing, spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(interp)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(img)
