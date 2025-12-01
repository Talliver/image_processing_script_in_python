from PIL import Image
from typing import Optional, Dict, Any, Tuple
import io
import os
from pathlib import Path
import tempfile
import shutil


def _format_from_path(path: str, fmt: Optional[str]) -> str:
    if fmt:
        return fmt.upper()
    ext = Path(path).suffix.lower().lstrip('.')
    mapping = {
        'jpg': 'JPEG', 'jpeg': 'JPEG', 'png': 'PNG', 'webp': 'WEBP', 'tif': 'TIFF', 'tiff': 'TIFF',
        'bmp': 'BMP', 'gif': 'GIF'
    }
    return mapping.get(ext, ext.upper())


def save_image_bytes(img: Image.Image, format: str, **options) -> Tuple[bytes, Dict[str, Any]]:
    bio = io.BytesIO()
    save_kwargs = {}
    for k in ['quality', 'optimize', 'progressive', 'compress_level', 'compress_level', 'dpi', 'icc_profile', 'exif', 'lossless']:
        if k in options and options[k] is not None:
            save_kwargs[k] = options[k]
    img.save(bio, format=format, **save_kwargs)
    data = bio.getvalue()
    meta = {
        'format': format,
        'size': len(data),
        'options_used': {k: v for k, v in save_kwargs.items()}
    }
    return data, meta


def save_image(img: Image.Image,
               path: str,
               format: Optional[str] = None,
               overwrite: bool = True,
               # JPEG options
               quality: Optional[int] = None,
               optimize: bool = False,
               progressive: bool = False,
               subsampling: Optional[int] = None,
               # PNG options
               png_compress_level: Optional[int] = None,
               png_optimize: bool = False,
               # WEBP options
               webp_quality: Optional[int] = None,
               webp_lossless: Optional[bool] = None,
               # general
               dpi: Optional[Tuple[int, int]] = None,
               icc_profile: Optional[bytes] = None,
               exif: Optional[bytes] = None,
               ) -> Dict[str, Any]:
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"File exists and overwrite disabled: {path}")

    p.parent.mkdir(parents=True, exist_ok=True)

    fmt = _format_from_path(path, format)

    save_opts: Dict[str, Any] = {}
    if fmt == 'JPEG':
        if quality is not None:
            save_opts['quality'] = max(1, min(95, int(quality)))
        if optimize:
            save_opts['optimize'] = True
        if progressive:
            save_opts['progressive'] = True
        if subsampling is not None:
            save_opts['subsampling'] = subsampling
        if dpi is not None:
            save_opts['dpi'] = tuple(dpi)
        if icc_profile is not None:
            save_opts['icc_profile'] = icc_profile
        if exif is not None:
            save_opts['exif'] = exif
    elif fmt == 'PNG':
        if png_compress_level is not None:
            save_opts['compress_level'] = max(0, min(9, int(png_compress_level)))
        if png_optimize:
            save_opts['optimize'] = True
        if icc_profile is not None:
            save_opts['icc_profile'] = icc_profile
        if exif is not None:
            save_opts['exif'] = exif
    elif fmt == 'WEBP':
        if webp_quality is not None:
            save_opts['quality'] = max(0, min(100, int(webp_quality)))
        if webp_lossless is not None:
            save_opts['lossless'] = bool(webp_lossless)
        if icc_profile is not None:
            save_opts['icc_profile'] = icc_profile
        if exif is not None:
            save_opts['exif'] = exif
    else:
        if quality is not None:
            save_opts['quality'] = quality
        if icc_profile is not None:
            save_opts['icc_profile'] = icc_profile
        if exif is not None:
            save_opts['exif'] = exif

    if dpi is not None and 'dpi' not in save_opts:
        save_opts['dpi'] = tuple(dpi)

    temp_dir = str(p.parent) if p.parent.exists() else None
    fd, tmp_path = tempfile.mkstemp(prefix='.tmp_image_', dir=temp_dir)
    os.close(fd)
    try:
        data, meta = save_image_bytes(img, fmt, **save_opts)
        with open(tmp_path, 'wb') as tf:
            tf.write(data)

        shutil.move(tmp_path, str(p))
        filesize = p.stat().st_size
        result = {
            'path': str(p.resolve()),
            'format': fmt,
            'filesize': filesize,
            'options_used': save_opts,
        }
        return result
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise
