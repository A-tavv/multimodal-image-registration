import SimpleITK as sitk
from typing import Optional, Tuple


def centered_initial_transform(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    dimension = fixed.GetDimension()
    if dimension == 2:
        transform = sitk.Euler2DTransform()
    else:
        transform = sitk.VersorRigid3DTransform()
    return sitk.CenteredTransformInitializer(
        fixed,
        moving,
        transform,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )


def _extract_leaf_transform(transform: sitk.Transform) -> sitk.Transform:
    if hasattr(transform, "GetNumberOfTransforms"):
        count = transform.GetNumberOfTransforms()
        if count > 0:
            return _extract_leaf_transform(transform.GetNthTransform(count - 1))
    return transform


def rigid_to_affine(transform: sitk.Transform) -> sitk.AffineTransform:
    base = _extract_leaf_transform(transform)
    dimension = base.GetDimension()
    affine = sitk.AffineTransform(dimension)
    affine.SetMatrix(base.GetMatrix())
    affine.SetTranslation(base.GetTranslation())
    affine.SetCenter(base.GetCenter())
    return affine


def register_rigid_affine(
    fixed: sitk.Image,
    moving: sitk.Image,
    fixed_mask: Optional[sitk.Image] = None,
    moving_mask: Optional[sitk.Image] = None,
    metric_bins: int = 50,
    sampling_pct: float = 0.2,
    levels: Tuple[int, int, int] = (4, 2, 1),
) -> Tuple[sitk.Transform, sitk.Transform]:
    def _safe_levels(image: sitk.Image, desired: Tuple[int, ...]) -> Tuple[int, ...]:
        min_dim = max(1, min(image.GetSize()))
        if min_dim < 4:
            return (1,)
        max_factor = max(1, min_dim // 4)
        return tuple(max(1, min(level, max_factor)) for level in desired)

    safe_levels = _safe_levels(fixed, levels)
    if len(safe_levels) == 1 and safe_levels[0] == 1:
        smoothing_sigmas = [0.0]
    else:
        smoothing_sigmas = list(reversed(safe_levels))

    init = centered_initial_transform(fixed, moving)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(metric_bins)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling_pct)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel(list(safe_levels))
    reg.SetSmoothingSigmasPerLevel(smoothing_sigmas)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    if fixed_mask is not None:
        reg.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        reg.SetMetricMovingMask(moving_mask)

    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4, numberOfIterations=200, relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()

    reg.SetInitialTransform(init, inPlace=False)
    rigid_out = reg.Execute(fixed, moving)
    initial_affine = rigid_to_affine(rigid_out)

    reg2 = sitk.ImageRegistrationMethod()
    reg2.SetMetricAsMattesMutualInformation(metric_bins)
    reg2.SetMetricSamplingStrategy(reg2.RANDOM)
    reg2.SetMetricSamplingPercentage(sampling_pct)
    reg2.SetInterpolator(sitk.sitkLinear)
    reg2.SetShrinkFactorsPerLevel(list(safe_levels))
    reg2.SetSmoothingSigmasPerLevel(smoothing_sigmas)
    reg2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    if fixed_mask is not None:
        reg2.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        reg2.SetMetricMovingMask(moving_mask)

    reg2.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=200, relaxationFactor=0.7)
    reg2.SetOptimizerScalesFromPhysicalShift()

    reg2.SetInitialTransform(initial_affine, inPlace=False)
    affine_out = reg2.Execute(fixed, moving)

    return rigid_out, affine_out


def resample_moving_to_fixed(moving: sitk.Image, fixed: sitk.Image, transform: sitk.Transform, interp=sitk.sitkLinear) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(interp)
    resampler.SetTransform(transform)
    return resampler.Execute(moving)
