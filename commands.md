# Commands (Flags) Reference

This file is the canonical reference for command-line flags and recommended default values. Other docs reference this file rather than duplicating flag-level details.

## IO Flags

- `--input <path>`: Path to the input image. Recommended: an absolute or repo-relative path to a PNG/JPEG/TIFF file.
- `--output <path>`: Path to the output file. Recommended: include the extension to indicate desired output format (e.g., `out.png`).
- `--overwrite`: Optional boolean flag. Recommended default: disabled. Only enable for batch or scripted runs where overwriting is intended.

## Processing Flags

- `--filter <name>`: Apply a named filter; common values: `blur`, `sharpen`, `edge-detect`. Recommended: `blur` for softening and `edge-detect` for analysis.
- `--resize <WxH>`: Resize to width `W` and height `H` in pixels, written as `800x600`. Recommended: preserve aspect ratio when possible (provide one dimension or calculate both).
- `--quality <0-100>`: Output quality for lossy formats (JPEG). Recommended: `85` for a good quality/size tradeoff.

## Performance Flags

- `--threads <N>`: Number of worker threads for applicable operations. Recommended: number of CPU cores or `2` for small images.
- `--mode <eager|stream>`: Choose loading behaviour. Recommended: `eager` for small files, `stream` for very large inputs.

## Logging & Verbosity

- `--verbose`: Increase logging verbosity (info -> debug). Recommended to pass when troubleshooting.
- `--quiet`: Reduce non-essential output; still print errors.

## Examples

- Basic processing: `python main.py --input img.png --output out.png --filter blur`
- Resize and high-quality JPEG: `python main.py --input img.png --output out.jpg --resize 1600x1200 --quality 90`
- Batch safe run (no overwrite): `python main.py --input img.png --output out.png`

Refer to `cli.md` for how flags are parsed and how the CLI maps flags into the runtime configuration.
