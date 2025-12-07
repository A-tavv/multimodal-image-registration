import os
import SimpleITK as sitk
from typing import Optional, Any, Tuple, List


# PixelIDValueEnum is not present in some SimpleITK builds; we avoid referencing it directly.
def read_image(path: str, pixel_type: Any = sitk.sitkFloat32) -> sitk.Image:
    img = sitk.ReadImage(path)
    if img.GetNumberOfComponentsPerPixel() > 1:
        img = sitk.VectorMagnitude(img)
    if img.GetPixelID() != pixel_type:
        img = sitk.Cast(img, pixel_type)
    return img


def list_dicom_series(directory: str) -> List[str]:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(directory)
    return series_ids if series_ids else []


def read_dicom_series(directory: str, series_uid: Optional[str] = None, pixel_type: Any = sitk.sitkFloat32) -> Tuple[sitk.Image, List[str]]:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(directory)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {directory}")
    if series_uid is None:
        # choose the series with the largest number of files
        chosen = max(series_ids, key=lambda uid: len(reader.GetGDCMSeriesFileNames(directory, uid)))
    else:
        if series_uid not in series_ids:
            raise RuntimeError(f"Series UID {series_uid} not found in directory. Found: {series_ids}")
        chosen = series_uid
    file_names = reader.GetGDCMSeriesFileNames(directory, chosen)
    reader.SetFileNames(file_names)
    img = reader.Execute()
    if img.GetPixelID() != pixel_type:
        img = sitk.Cast(img, pixel_type)
    return img, series_ids


def read_image_auto(path: str, series_uid: Optional[str] = None, pixel_type: Any = sitk.sitkFloat32) -> Tuple[sitk.Image, Optional[List[str]]]:
    """Read either a single file (NIfTI/MHA/etc.) or a DICOM directory.
    Returns image and list of series IDs if DICOM.
    """
    if os.path.isdir(path):
        img, series_ids = read_dicom_series(path, series_uid=series_uid, pixel_type=pixel_type)
        return img, series_ids
    else:
        return read_image(path, pixel_type), None


def write_image(img: sitk.Image, path: str) -> None:
    sitk.WriteImage(img, path)


def read_optional_mask(path: Optional[str]) -> Optional[sitk.Image]:
    if path is None:
        return None
    return sitk.ReadImage(path)


def resample_to_reference(img: sitk.Image, reference: sitk.Image, interp: Any) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interp)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(img)
