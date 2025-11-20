"""CLI wrapper for the image processing pipeline.

Usage examples:
    python3 main.py                # auto-detect input, apply default filters, save output_image.jpg
    python3 main.py -i in.jpg -o out.png -f sepia -f "unsharp:1.2,2" --quality 85
    python3 main.py --dry-run -v
"""

import argparse
import sys
import os
import shutil
from datetime import datetime
from typing import List, Optional, Tuple

import read
import process as proc
import write as save


def _parse_filters(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        if not v:
            continue
    parts = [p.strip() for p in v.split(',') if p.strip()]
    out.extend(parts)
    return out


def _maybe_backup(path: str, no_backup: bool, verbose: bool) -> Optional[str]:
    if not os.path.exists(path):
        return None
    if no_backup:
        if verbose:
            print(f"Skipping backup for existing file: {path}")
            return None
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    bak = f"{path}.bak.{ts}"
    shutil.copy2(path, bak)
    if verbose:
        print(f"Backed up existing file {path} -> {bak}")
    return bak


def run_pipeline(input_path: Optional[str], output_path: str, filters: List[str],
        overwrite: bool, no_backup: bool, dry_run: bool, verbose: bool,
        fmt: Optional[str], quality: Optional[int], optimize: bool,
        progressive: bool, png_compress_level: Optional[int], png_optimize: bool,
        webp_quality: Optional[int], webp_lossless: Optional[bool], dpi: Optional[Tuple[int,int]]):
    if input_path:
        img, meta = read.load_image(input_path)
    else:
        img, meta = read.load_image(None)

    if verbose:
        print(f"Loaded image: {meta['filename']} ({meta.get('format')}, {meta.get('size')})")

    if dry_run:
        print("Dry run mode: the following actions would be taken:")
        print(f" - Apply filters: {filters}")
        print(f" - Save to: {output_path} (format={fmt or 'auto'})")
        return {"dry_run": True, "input": meta, "filters": filters, "output": output_path}

    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output exists and overwrite is False: {output_path}")

    if os.path.exists(output_path):
        _maybe_backup(output_path, no_backup=no_backup, verbose=verbose)

    processed = proc.apply_filters(img, filters)


    meta_out = save.save_image(
        processed,
        output_path,
        format=fmt,
        overwrite=True,
        quality=quality,
        optimize=optimize,
        progressive=progressive,
        png_compress_level=png_compress_level,
        png_optimize=png_optimize,
        webp_quality=webp_quality,
        webp_lossless=webp_lossless,
        dpi=dpi,
    )

    if verbose:
        print(f"Saved processed image: {meta_out['path']} ({meta_out['format']}, {meta_out['filesize']} bytes)")

    return {"input": meta, "filters": filters, "output": meta_out}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Image processing CLI: read -> process -> write with safeguards")
    p.add_argument('-i', '--input', help='Input image path (optional, auto-detect in cwd if omitted)')
    p.add_argument('-o', '--output', default='output_image.jpg', help='Output image path')
    p.add_argument('-f', '--filter', action='append', default=[], help='Filter to apply (can be repeated or comma-separated)')
    p.add_argument('--overwrite', action='store_true', help='Allow overwriting existing output file')
    p.add_argument('--no-backup', action='store_true', help='Do not create a backup of existing output')
    p.add_argument('--dry-run', action='store_true', help='Print planned actions without writing files')
    p.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    p.add_argument('--format', help='Force output format (e.g. JPEG, PNG, WEBP)')
    # format-specific options (common)
    p.add_argument('--quality', type=int, help='Quality for lossy formats (JPEG/WEBP)')
    p.add_argument('--optimize', action='store_true', help='Optimize output (where supported)')
    p.add_argument('--progressive', action='store_true', help='Create progressive JPEG where supported')
    p.add_argument('--png-compress-level', type=int, help='PNG compress level 0-9')
    p.add_argument('--png-optimize', action='store_true', help='PNG optimize flag')
    p.add_argument('--webp-quality', type=int, help='WEBP quality 0-100')
    p.add_argument('--webp-lossless', action='store_true', help='WEBP lossless')
    p.add_argument('--dpi', nargs=2, type=int, metavar=('X','Y'), help='DPI to embed in output')
    return p


def main(argv: Optional[List[str]] = None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    filters = _parse_filters(args.filter) if args.filter else ['grayscale','blur','sharpen']

    try:
        result = run_pipeline(
            input_path=args.input,
            output_path=args.output,
            filters=filters,
            overwrite=args.overwrite,
            no_backup=args.no_backuP if False else args.no_backup,
            dry_run=args.dry_run,
            verbose=args.verbose,
            fmt=args.format,
            quality=args.quality,
            optimize=args.optimize,
            progressive=args.progressive,
            png_compress_level=args.png_compress_level,
            png_optimize=args.png_optimize,
            webp_quality=args.webp_quality,
            webp_lossless=(True if args.webp_lossless else None),
            dpi=tuple(args.dpi) if args.dpi else None,
        )
        if args.verbose:
            print('Pipeline result:', result)
    except Exception as e:
        print('Error:', e, file=sys.stderr)
        sys.exit(2)

    sys.exit(0)


if __name__ == '__main__':
    main()