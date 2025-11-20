# imageprocessing — Quick examples

This README gives short, copy-pastable examples showing how to use the `imageprocessing` helper modules in this folder (`read.py`, `process.py`, `write.py`, and `main.py`). Replace the placeholder `Image` type with whatever your project uses (e.g. `PIL.Image.Image` or a NumPy array).

**Example Functions**

- **read_image:** `read_image(path: str) -> Image` : Read an image from disk and return an image object/array.
- **read_image_bytes:** `read_image_bytes(data: bytes) -> Image` : Load an image from raw bytes (HTTP uploads, in-memory pipelines).
- **get_image_metadata:** `get_image_metadata(path: str) -> dict` : Extract basic metadata (size, mode, format) without loading full pixels.
- **resize_image:** `resize_image(img: Image, size: tuple[int,int], resample: str='bilinear') -> Image` : Resize to `(width, height)`.
- **crop_image:** `crop_image(img: Image, box: tuple[int,int,int,int]) -> Image` : Crop to `(left, top, right, bottom)`.
- **convert_to_grayscale:** `convert_to_grayscale(img: Image) -> Image` : Convert to grayscale.
- **adjust_contrast:** `adjust_contrast(img: Image, factor: float) -> Image` : Change contrast; `1.0` = no change.
- **apply_threshold:** `apply_threshold(img: Image, threshold: int) -> Image` : Binary mask from threshold (0-255).
- **detect_edges:** `detect_edges(img: Image, method: str='sobel') -> Image` : Return an edge map.
- **annotate_image:** `annotate_image(img: Image, annotations: list[dict]) -> Image` : Draw boxes/text given annotation specs.
- **write_image:** `write_image(img: Image, path: str, format: str|None=None, quality: int|None=None) -> None` : Save to disk.
- **write_image_bytes:** `write_image_bytes(img: Image, format: str='PNG') -> bytes` : Return encoded image bytes.
- **pipeline_process_and_save:** `pipeline_process_and_save(src_path: str, dst_path: str, ops: list[Callable]) -> None` : Read, apply ops, write result.
- **batch_process_directory:** `batch_process_directory(input_dir: str, output_dir: str, op: Callable, recursive: bool=False) -> dict` : Apply `op` to all images; return report.

**Usage Snippets**

- Single-file pipeline

```python
# If the imageprocessing folder is on PYTHONPATH or installed as a package:
from imageprocessing.read import read_image
from imageprocessing.process import resize_image, convert_to_grayscale
from imageprocessing.write import write_image

img = read_image("input.jpg")
img = resize_image(img, (800, 600))
img = convert_to_grayscale(img)
write_image(img, "output.png", format="PNG")
```

- In-memory web upload (bytes -> process -> bytes)

```python
from imageprocessing.read import read_image_bytes
from imageprocessing.process import apply_threshold
from imageprocessing.write import write_image_bytes

# `upload_bytes` could come from a Flask/Django request.files read() call
img = read_image_bytes(upload_bytes)
img = apply_threshold(img, 128)
out_bytes = write_image_bytes(img, format="JPEG")
```

- Batch convert + watermark (conceptual)

```python
from imageprocessing.write import write_image
from imageprocessing.read import read_image
from imageprocessing.process import annotate_image
from imageprocessing.process import batch_process_directory

def add_watermark(img):
    return annotate_image(img, [{"text": "© Me", "pos": (10, 10), "color": "white"}])

stats = batch_process_directory("photos/", "photos_out/", add_watermark, recursive=True)
print(stats)
```

**Imports / package notes**

- When running scripts from inside this folder, you may import as `from read import read_image` (local import).
- When the folder is used as a package (added to `PYTHONPATH` or installed), prefer `from imageprocessing.read import read_image`.

**Quick try**

Run a simple example from the folder root (adjust import style if necessary):

```bash
python -c "from read import read_image; from write import write_image; img=read_image('input.jpg'); write_image(img,'out.png')"
```

**Next steps**

- I can implement these helper stubs directly into `read.py`, `process.py`, and `write.py` if you want runnable helpers.
- I can also add tests or example images to try the snippets.

---

File: `README.md` for the `imageprocessing` folder — concise examples to get started.
