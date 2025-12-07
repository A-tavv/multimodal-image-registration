import argparse
import json
import os

from src.pipeline import PipelineConfig, run_registration_pipeline
from src.evaluate import display_overlay


def main():
    parser = argparse.ArgumentParser(description="CT/MRI registration pipeline")
    parser.add_argument("--fixed", help="Path to fixed image file OR DICOM directory (e.g., CT)")
    parser.add_argument("--moving", help="Path to moving image file OR DICOM directory (e.g., MRI)")
    parser.add_argument("--fixed-mask", default=None, help="Optional fixed mask path")
    parser.add_argument("--moving-mask", default=None, help="Optional moving mask path")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--no-hist-match", action="store_true", help="Disable histogram matching")
    parser.add_argument("--config", default="config.json", help="Path to default configuration JSON")
    parser.add_argument("--target-spacing", default=None, help="Target isotropic spacing, comma-separated")
    parser.add_argument("--fixed-series-uid", default=None, help="Specific DICOM series UID for fixed (if directory)")
    parser.add_argument("--moving-series-uid", default=None, help="Specific DICOM series UID for moving (if directory)")
    parser.add_argument("--show", action="store_true", help="Display overlay figure at the end of the run")
    parser.add_argument("--force-size-2d", type=int, default=None, help="Resize 2D images to NxN pixels before registration (0 to disable)")
    parser.add_argument("--crop-2d-to-moving", action="store_true", help="Crop overlays to the moving extent for clearer visualization")
    parser.add_argument("--overlay-cmap", default=None, help="Matplotlib colormap used to colorize the moving slice in saved/displayed overlays")

    args = parser.parse_args()
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)

    fixed_path = args.fixed or config.get("fixed")
    moving_path = args.moving or config.get("moving")
    if fixed_path is None or moving_path is None:
        parser.error("--fixed and --moving are required (either via CLI or config)")

    fixed_mask_path = args.fixed_mask or config.get("fixed_mask")
    moving_mask_path = args.moving_mask or config.get("moving_mask")
    outdir = args.outdir or config.get("outdir", "outputs")
    target_spacing_str = args.target_spacing or config.get("target_spacing") or "1,1,1"
    args.no_hist_match = args.no_hist_match or bool(config.get("no_hist_match", False))
    fixed_series_uid = args.fixed_series_uid or config.get("fixed_series_uid")
    moving_series_uid = args.moving_series_uid or config.get("moving_series_uid")
    args.show = args.show or bool(config.get("show", False))
    overlay_cmap = "Reds"
    if args.force_size_2d is None:
        args.force_size_2d = config.get("force_size_2d")
        args.crop_2d_to_moving = args.crop_2d_to_moving or bool(config.get("crop_2d_to_moving", False))
        overlay_cmap = args.overlay_cmap or config.get("overlay_cmap") or overlay_cmap
    if isinstance(args.force_size_2d, str):
        try:
            args.force_size_2d = int(args.force_size_2d)
        except ValueError:
            args.force_size_2d = None

    args.fixed = fixed_path
    args.moving = moving_path
    args.fixed_mask = fixed_mask_path
    args.moving_mask = moving_mask_path
    args.outdir = outdir
    args.target_spacing = target_spacing_str
    args.fixed_series_uid = fixed_series_uid
    args.moving_series_uid = moving_series_uid
    pipeline_config = PipelineConfig(
        fixed_path=args.fixed,
        moving_path=args.moving,
        fixed_mask_path=args.fixed_mask,
        moving_mask_path=args.moving_mask,
        outdir=args.outdir,
        target_spacing=args.target_spacing,
        histogram_matching=not args.no_hist_match,
        fixed_series_uid=args.fixed_series_uid,
        moving_series_uid=args.moving_series_uid,
        save_outputs=True,
        force_size_2d=args.force_size_2d,
        crop_2d_to_moving=args.crop_2d_to_moving,
        overlay_cmap=overlay_cmap,
    )

    result = run_registration_pipeline(pipeline_config)

    if result.fixed_series_ids is not None:
        print(f"Fixed DICOM series detected. Available series UIDs: {result.fixed_series_ids}")
    if result.moving_series_ids is not None:
        print(f"Moving DICOM series detected. Available series UIDs: {result.moving_series_ids}")

    if args.show:
        display_overlay(
            result.fixed_image,
            result.registered_image,
            crop_to_moving=result.crop_2d_to_moving,
            overlay_cmap=overlay_cmap,
        )

    print("Registration complete. Outputs saved to:", result.output_dir)


if __name__ == "__main__":
    main()
