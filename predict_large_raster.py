"""Apply a trained three-stage MSRL4SSC model to large multiband GeoTIFFs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

from model import (CONVENTIONAL_BANDS, S1_BANDS, S2_BANDS, CrossModalSSL,
                   TaskAdapter, band_indexes, load_stage3, predict_stage3)

ROOT = Path(__file__).resolve().parent
NODATA = -9999.0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(ROOT / "outputs" / "training"),
                        help="Directory containing stage2_final.pt and stage3_*.joblib")
    parser.add_argument("--input", nargs="+", default=[str(ROOT / "data" / "input_geotiffs")],
                        help="Input GeoTIFF file(s) or directories")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "prediction_maps"))
    parser.add_argument("--path", choices=["auto", "s1", "s2", "both"], default="auto",
                        help="auto selects both/S1/S2 separately for each map cell after nodata checks")
    parser.add_argument("--stride", type=int, default=16,
                        help="Map-cell size in pixels; use 1 for pointwise sliding inference")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument(
        "--invalid-values", nargs="*", type=float, default=[-9999.0, -32768.0],
        help="Additional numeric NoData sentinels when the GeoTIFF metadata does not declare them",
    )
    parser.add_argument("--smoke-test", action="store_true",
                        help="Use bundled synthetic raster and the model produced by train.py --smoke-test")
    return parser.parse_args()


def discover(items: list[str]) -> list[Path]:
    files: list[Path] = []
    for value in items:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(path.glob("*.tif"))); files.extend(sorted(path.glob("*.tiff")))
        elif path.is_file(): files.append(path)
        else: raise FileNotFoundError(path)
    unique = list(dict.fromkeys(files))
    if not unique: raise FileNotFoundError("No input GeoTIFFs were found")
    return unique


def load_model(model_dir: Path, device: torch.device):
    checkpoint = torch.load(model_dir / "stage2_final.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    ssl = CrossModalSSL(**config)
    model = TaskAdapter(ssl, config["dim"], config["heads"], config["dropout"])
    model.load_state_dict(checkpoint["model"]); model.to(device).eval()
    stage3 = {path: load_stage3(model_dir / f"stage3_{path}.joblib")
              for path in ("s1", "s2", "both")}
    return model, stage3, checkpoint, config


def batches(sequence: list[tuple], size: int):
    for start in range(0, len(sequence), size): yield sequence[start:start + size]


@torch.inference_mode()
def predict_raster(source_path: Path, destination_path: Path, model: TaskAdapter,
                   stage3: dict[str, dict], checkpoint: dict, config: dict, path: str,
                   stride: int, batch_size: int, device: torch.device,
                   clip: bool, invalid_values: list[float]) -> dict:
    image_size = int(config["image_size"]); half = image_size // 2
    means, stds = checkpoint["means"], checkpoint["stds"]
    with rasterio.open(source_path) as src:
        if path == "auto" and src.count != 44:
            raise ValueError(
                f"Automatic modality routing expects a 44-band GeoTIFF, found {src.count}: {source_path}"
            )
        s2_idx = band_indexes(src.descriptions, S2_BANDS)
        s1_idx = band_indexes(src.descriptions, S1_BANDS)
        conventional_idx = {
            route: band_indexes(src.descriptions, bundle["conventional_bands"])
            for route, bundle in stage3.items()
        }

        def invalid_mask(values: np.ndarray | np.ma.MaskedArray) -> np.ndarray:
            masked = np.ma.getmaskarray(values)
            array = np.asarray(np.ma.filled(values, np.nan), dtype=np.float32)
            invalid = masked | ~np.isfinite(array)
            for sentinel in invalid_values:
                invalid |= array == np.float32(sentinel)
            return invalid

        def clean(values: np.ndarray | np.ma.MaskedArray) -> np.ndarray:
            array = np.asarray(np.ma.filled(values, np.nan), dtype=np.float32)
            invalid = invalid_mask(values)
            array[invalid] = np.nan
            return array

        def valid_center(indexes: list[int], row: int, col: int) -> bool:
            values = src.read(indexes, window=Window(col, row, 1, 1), masked=True)
            return not invalid_mask(values).any()

        def choose_route(row: int, col: int) -> str | None:
            s1_valid, s2_valid = valid_center(s1_idx, row, col), valid_center(s2_idx, row, col)
            if path == "auto":
                if s1_valid and s2_valid: return "both"
                if s1_valid: return "s1"
                if s2_valid: return "s2"
                return None
            required_valid = {"s1": s1_valid, "s2": s2_valid,
                              "both": s1_valid and s2_valid}[path]
            return path if required_valid else None

        jobs = []
        for row0 in range(0, src.height, stride):
            for col0 in range(0, src.width, stride):
                height, width = min(stride, src.height - row0), min(stride, src.width - col0)
                jobs.append((row0, col0, height, width, row0 + height // 2, col0 + width // 2))
        output = np.full((src.height, src.width), NODATA, np.float32)
        predicted = 0
        route_pixels = {"s1": 0, "s2": 0, "both": 0, "nodata": 0}
        for group in batches(jobs, batch_size):
            s2_list, s1_list, raw_list, routes, valid_jobs = [], [], [], [], []
            for job in group:
                row0, col0, height, width, row, col = job
                route = choose_route(row, col)
                if route is None:
                    route_pixels["nodata"] += height * width
                    continue
                patch_window = Window(col - half, row - half, image_size, image_size)
                s2 = clean(src.read(s2_idx, window=patch_window, boundless=True,
                                    masked=True))
                s1 = clean(src.read(s1_idx, window=patch_window, boundless=True,
                                    masked=True))
                raw_masked = src.read(conventional_idx[route],
                                      window=Window(col, row, 1, 1), masked=True)
                raw = clean(raw_masked).reshape(-1)
                if not np.isfinite(raw).all():
                    route_pixels["nodata"] += height * width
                    continue
                s2 = np.nan_to_num((s2 - means["s2"][:, None, None]) / stds["s2"][:, None, None], nan=0).clip(-8, 8)
                s1 = np.nan_to_num((s1 - means["s1"][:, None, None]) / stds["s1"][:, None, None], nan=0).clip(-8, 8)
                if route == "s1": s2.fill(0)
                if route == "s2": s1.fill(0)
                s2_list.append(s2); s1_list.append(s1); raw_list.append(raw)
                routes.append(route); valid_jobs.append(job)
            if not valid_jobs: continue
            s2_tensor = torch.from_numpy(np.stack(s2_list)).to(device)
            s1_tensor = torch.from_numpy(np.stack(s1_list)).to(device)
            local = torch.zeros((len(valid_jobs), 2), device=device)
            model_output = model(s2_tensor, s1_tensor, local)
            values = np.full(len(valid_jobs), np.nan, np.float32)
            for route in ("s1", "s2", "both"):
                take = [i for i, selected in enumerate(routes) if selected == route]
                if not len(take): continue
                states = model_output[route]["state"][take].cpu().numpy()
                raw = np.stack([raw_list[i] for i in take])
                route_values = predict_stage3(stage3[route], states, raw).astype(np.float32)
                if clip:
                    route_values = np.clip(route_values, stage3[route]["target_min"],
                                           stage3[route]["target_max"])
                values[take] = route_values
            for job, route, value in zip(valid_jobs, routes, values):
                row0, col0, height, width, _, _ = job
                output[row0:row0 + height, col0:col0 + width] = value
                predicted += height * width; route_pixels[route] += height * width
        profile = src.profile.copy(); profile.update(count=1, dtype="float32", nodata=NODATA, compress="deflate")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(destination_path, "w", **profile) as dst:
            dst.write(output, 1); dst.set_band_description(1, f"MSRL4SSC_{path}_SSC")
            dst.update_tags(model="MSRL4SSC", modality_path=path, inference_stride=stride,
                            route_s1_pixels=route_pixels["s1"],
                            route_s2_pixels=route_pixels["s2"],
                            route_both_pixels=route_pixels["both"],
                            route_nodata_pixels=route_pixels["nodata"],
                            invalid_values=",".join(str(value) for value in invalid_values),
                            note="auto checks centre-pixel nodata per modality; stride>1 fills each map cell")
    return {"input": str(source_path), "output": str(destination_path),
            "path": path, "stride": stride, "predicted_pixels": predicted,
            "route_pixels": route_pixels, "invalid_values": invalid_values}


def main() -> None:
    args = arguments()
    if args.stride < 1: raise ValueError("--stride must be at least 1")
    if args.smoke_test:
        args.model_dir = str(ROOT / "outputs" / "smoke_training")
        args.input = [str(ROOT / "data" / "smoke_test" / "input_raster")]
        args.output_dir = str(ROOT / "outputs" / "smoke_prediction")
    model_dir, output_dir = Path(args.model_dir).resolve(), Path(args.output_dir).resolve()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, stage3, checkpoint, config = load_model(model_dir, device)
    records = []
    for source in discover(args.input):
        destination = output_dir / f"{source.stem}_MSRL4SSC_{args.path}.tif"
        records.append(predict_raster(source, destination, model, stage3, checkpoint, config,
                                      args.path, args.stride, args.batch_size, device,
                                      not args.no_clip, args.invalid_values))
        print(f"Saved {destination}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prediction_manifest.json").write_text(
        json.dumps({"model_dir": str(model_dir), "rasters": records}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
