# Write Module Documentation

The `write` module is responsible for persisting processed images to disk. It provides safe, consistent writes and preserves necessary metadata.

## Responsibilities

- Provide a single `save(image, path, *, config)` entry point that writes the image to `path` in the desired format.
- Support common output formats (PNG, JPEG, TIFF) and allow format-specific options (quality, compression level, metadata embedding).
- Perform safe writes (atomic where possible) to avoid corrupting existing files on failure.
- Validate output path and raise a `WriteError` with a concise message when writes fail.

## Behaviour and Safety

- Default behavior should avoid overwriting existing files unless `--overwrite` is explicitly provided.
- Implement atomic writes by writing to a temporary file in the same directory and then renaming it to the final filename.

## Metadata and Format Options

- Preserve relevant metadata when possible (for example, color profile, DPI, EXIF data), but make the behavior explicit in `save` options.
- Expose format options via `config` (e.g., `quality` for JPEG, `compression` for PNG/TIFF).

## Example

`write.save(image, 'out.png', config={'overwrite': False, 'format': 'PNG'})`
