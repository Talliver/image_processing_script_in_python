from __future__ import annotations
import os
import io
import json
import base64
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image, ExifTags
import argparse

"""read.py

Improvements made:
- Fixed broken import/try-except
- Added `load_image(path: Optional[str]) -> (Image.Image, metadata)` which will auto-detect an image
  in the current working directory when `path` is omitted.
- CLI will auto-include pixel data for small files if the user doesn't request pixels explicitly.

Primary API:
  - read_image(path, include_pixels=False, pixel_format='base64', max_pixels=1000000) -> dict
  - load_image(path: Optional[str] = None) -> Tuple[Image.Image, Dict[str, Any]]

"""


def _safe_to_text(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__type__": "bytes/base64", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (tuple, list)):
        return [_safe_to_text(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _parse_exif(img: Image.Image) -> Dict[str, Any]:
    exif_data: Dict[str, Any] = {}
    raw = img.getexif()
    if not raw:
        return exif_data
    for tag_id, val in raw.items():
        tag = ExifTags.TAGS.get(tag_id, str(tag_id))
        exif_data[tag] = _safe_to_text(val)
    try:
        raw_bytes = raw.tobytes()
        if raw_bytes:
            exif_data["__raw_exif_bytes_base64"] = base64.b64encode(raw_bytes).decode("ascii")
    except Exception:
        pass
    return exif_data


def _sample_pixels(img: Image.Image, max_pixels: int) -> Tuple[Tuple[int, int], List[Any]]:
    w, h = img.size
    total = w * h
    pix_iter = img.getdata()
    sample = []
    limit = min(int(max_pixels), total)
    for i, p in enumerate(pix_iter):
        if i >= limit:
            break
        if isinstance(p, (tuple, list)):
            sample.append([int(x) if isinstance(x, (int,)) else _safe_to_text(x) for x in p])
        else:
            sample.append(int(p) if isinstance(p, (int,)) else _safe_to_text(p))
    return (w, h), sample


def detect_image_files(search_dir: Path = Path.cwd()) -> List[Path]:
    """Return candidate image files in the directory (non-recursive)."""
    exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.gif", "*.webp"]
    results: List[Path] = []
    for e in exts:
        results.extend(list(search_dir.glob(e)))
    # sort by name (stable) and return files only
    return [p for p in sorted(results) if p.is_file()]


def load_image(path: Optional[str] = None) -> Tuple[Image.Image, Dict[str, Any]]:
    """Load and return a PIL Image plus lightweight metadata.

    If path is None, this will try to auto-detect an image file in the current working
    directory and open the first candidate it finds.
    """
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
    else:
        candidates = detect_image_files(Path.cwd())
        if not candidates:
            raise FileNotFoundError("No image path provided and no image files found in the current directory")
        p = candidates[0]

    img = Image.open(p)
    metadata = {
        "path": str(p),
        "filename": p.name,
        "format": img.format,
        "mode": img.mode,
        "size": {"width": img.width, "height": img.height},
        "filesize": p.stat().st_size,
    }
    return img, metadata


def read_image(path: str,
                include_pixels: Optional[bool] = False,
                pixel_format: str = "base64",
                max_pixels: int = 1000000) -> Dict[str, Any]:
    """Read an image and return JSON-serializable metadata and optional pixel content.

    Auto-detection behavior: if include_pixels is None, small files (<1MB) will include
    pixels by default; larger files will not.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)

    filesize = p.stat().st_size
    # sensible default: include pixels for small files unless caller explicitly set False
    if include_pixels is None:
        include_pixels = filesize < 1_000_000

    result: Dict[str, Any] = {}
    result["filename"] = p.name
    result["path"] = str(p.resolve())
    result["filesize"] = filesize

    with open(p, "rb") as f:
        raw_file = f.read()
    # do not always include entire original file base64 by default (can be large), but keep for compatibility
    result["original_file_base64"] = base64.b64encode(raw_file).decode("ascii")

    with Image.open(io.BytesIO(raw_file)) as img:
        result["format"] = img.format
        result["mode"] = img.mode
        result["size"] = {"width": img.width, "height": img.height}
        info_text: Dict[str, Any] = {}
        for k, v in img.info.items():
            if isinstance(v, bytes):
                info_text[k] = {"__type__": "bytes/base64", "data": base64.b64encode(v).decode("ascii")}
            else:
                try:
                    json.dumps(v)
                    info_text[k] = v
                except TypeError:
                    info_text[k] = str(v)
        result["info"] = info_text

        icc = img.info.get("icc_profile")
        if icc:
            result["icc_profile_base64"] = base64.b64encode(icc if isinstance(icc, bytes) else icc.encode("latin1")).decode("ascii")

        result["exif"] = _parse_exif(img)

        if include_pixels:
            total_pixels = img.width * img.height
            if pixel_format == "base64":
                raw_pixels = img.tobytes()
                result["pixels"] = {
                    "format": "raw_bytes_base64",
                    "mode": img.mode,
                    "size": {"width": img.width, "height": img.height},
                    "total_pixels": total_pixels,
                    "data_base64": base64.b64encode(raw_pixels).decode("ascii")
                }
            elif pixel_format == "hex":
                raw_pixels = img.tobytes()
                result["pixels"] = {
                    "format": "raw_bytes_hex",
                    "mode": img.mode,
                    "size": {"width": img.width, "height": img.height},
                    "total_pixels": total_pixels,
                    "data_hex": raw_pixels.hex()
                }
            elif pixel_format == "csv":
                (w, h), sample = _sample_pixels(img, max_pixels)
                result["pixels"] = {
                    "format": "sampled_csv_like",
                    "mode": img.mode,
                    "size": {"width": w, "height": h},
                    "total_pixels": total_pixels,
                    "sampled_pixels_count": len(sample),
                    "sampled_pixels": sample
                }
            else:
                raise ValueError("pixel_format must be 'base64', 'hex' or 'csv'")

    return result


def _cli():
    parser = argparse.ArgumentParser(description="Read an image and emit all data as JSON (text).")
    parser.add_argument("path", nargs="?", default=None, help="Path to image file (optional, auto-detects in cwd if omitted)")
    parser.add_argument("--pixels", action="store_true", help="Include pixel data (can be very large)")
    parser.add_argument("--no-pixels", dest="no_pixels", action="store_true", help="Do not include pixel data")
    parser.add_argument("--pixel-format", choices=["base64", "csv", "hex"], default="base64",
                        help="How to encode pixel data when included")
    parser.add_argument("--max-pixels", type=int, default=1000000, help="Max pixels to sample for CSV mode")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    if args.path is None:
        # attempt to auto-detect an image file in cwd
        try:
            img_obj, meta = load_image(None)
            path = meta["path"]
        except FileNotFoundError:
            parser.error("No path provided and no image files detected in current directory")
    else:
        path = args.path

    include_pixels = None
    if args.pixels:
        include_pixels = True
    if args.no_pixels:
        include_pixels = False

    # If include_pixels is None, read_image() will decide based on file size (default behavior)
    data = read_image(path, include_pixels=include_pixels, pixel_format=args.pixel_format, max_pixels=args.max_pixels)
    if args.pretty:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(data, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    _cli()