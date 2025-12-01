# Read Module Documentation

The `read` module encapsulates file-input responsibilities: open files, validate formats, and convert file contents into an in-memory image representation suitable for `process`.

## Responsibilities

- Detect and validate supported input formats (PNG, JPEG, TIFF, BMP, and optionally others).
- Provide a single entry point (`load(path, *, config)`) that returns a consistent image object (for example, a NumPy array, Pillow `Image`, or a small wrapper object).
- Fail fast on invalid files and raise consistent exceptions that `main` translates into user-facing messages.
- Offer memory-friendly options: e.g., a streaming mode or lazy loading for very large files (if supported).

## Behaviour and Return Types

- The loader should return a documented image object type. If the project uses Pillow, return a `PIL.Image.Image` instance; if numpy, return an `ndarray` with shape `(H, W[, C])`.
- Always return image data plus metadata when available (mode, format, original DPI, color profile).

## Validation and Errors

- Validate that the file exists and is readable before attempting to parse it.
- Detect corrupted files early and raise `ReadError` (or a similarly named exception) with a short message and an optional `cause` attribute for debugging.

## Performance Considerations

- When reading many images or very large images, prefer an API that supports chunked reads or streaming to reduce peak memory usage.
- Provide an optional `mode` parameter in `load` that determines whether to return an in-memory copy (`mode='eager'`) or a lazy/streaming object (`mode='stream'`).

## Integration

- Call sites (typically `main`) should handle the returned image object uniformly. Convert or adapt the image only once at the pipeline boundary; do not spread conversion code throughout `process`.
