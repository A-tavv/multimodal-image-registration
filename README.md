# Multimodal CT/MRI Registration (SimpleITK)

Complete SimpleITK/ITK pipeline for aligning CT and MRI volumes. The workflow covers preprocessing, rigid and affine registration, and quantitative/visual evaluation so you can drop the project straight onto GitHub or into a report.

- **Registration** – Mattes Mutual Information with multi-resolution rigid + affine stages, saved transforms, and resampled moving volumes.
- **Preprocessing** – isotropic resampling, intensity normalization, histogram matching, and N4 bias correction.
- **Evaluation** – intensity metrics (MSE/MAE/NCC) for every run, optional Dice when masks are provided, deformation magnitude PNGs, and overlay figures.

## Features

- Automatic loading of NIfTI/MHA volumes or DICOM directories (largest series by default, with UID filtering support).
- Optional resizing/cropping helpers for 2D PNG/JPG slices so mismatched FOVs still produce readable overlays.
- Config-driven workflow via `config.json`, with a simple runner (`mir.py`).
- Organized outputs (preprocessed volumes, transforms, registered image, metrics, overlays) ready for downstream processing.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

## Quick Start

1. Point `config.json` at your fixed/moving volumes.
2. Run the main entry point:

   ```bash
   python mir.py
   ```

The run writes all intermediate artifacts to `outputs/`. For ad-hoc experiments, call the verbose CLI:

```bash
python mir.py --fixed patient_ct.nii.gz --moving patient_mri.nii.gz --target-spacing 1,1,1 --show
```

## CLI Highlights (`mir.py`)

- `--fixed`, `--moving`: file paths or DICOM directories.
- `--fixed-mask`, `--moving-mask`: enable Dice evaluation.
- `--target-spacing`: isotropic resampling size (default `1,1,1`).
- `--no-hist-match`: turn off histogram matching if it washes out CT contrast.
- `--force-size-2d`: resize 2D inputs to an `N×N` canvas before registration (set `0` to disable).
- `--crop-2d-to-moving`: crop overlays to the moving-image footprint for mismatched FOVs.
- `--overlay-cmap`: choose the Matplotlib colormap for the moving overlay (default `jet`).
- `--show`: pop up the Matplotlib viewer with fixed, registered, and overlay slices.

## Using DICOM Directories

You can pass directories for `--fixed`/`--moving`; the loader automatically selects the largest series. To lock onto a specific Series Instance UID:

```bash
python mini.py \
  --fixed DICOM_CT_DIR --fixed-series-uid "1.2.840...." \
  --moving DICOM_MR_DIR --moving-series-uid "1.2.840...."
```

## Notes

- Supported formats: NIfTI (`.nii/.nii.gz`), MHD/MHA, DICOM directories, and lightweight PNG/JPG slices.
- DICOM IO relies on SimpleITK’s GDCM reader; ensure your wheel includes GDCM (standard Windows wheels do).
- Defaults in `config.json` let you run `python mir.py` with zero flags; tweak any key there or via CLI overrides.
- Consider disabling histogram matching (`--no-hist-match`) for lung CTs where air/bone contrast is critical.
- For GitHub testing, simply edit `config.json` with the two volumes you want to align (any format supported above) and re-run `python mir.py`; all derived artifacts update automatically.