# imageprocessing — CLI examples

This README shows how to use the project's CLI wrapper (`main.py`) to run the imageprocessing pipeline from the command line. Examples below use `python3 main.py` from the `imageprocessing` folder root.

The CLI (see `main.py`) supports these common flags (summary):

- `-i, --input` : Input image path (optional — auto-detect in cwd if omitted)
- `-o, --output` : Output image path (default `output_image.jpg`)
- `-f, --filter` : Filter(s) to apply. Can be repeated or comma-separated.
- `--overwrite` : Allow overwriting existing output.
- `--no-backup` : Do not backup existing output file.
- `--dry-run` : Print planned actions without writing files.
- `-v, --verbose` : Verbose output.
- `--format` : Force output format (JPEG, PNG, WEBP, ...)
- `--quality` : Quality for lossy formats (JPEG/WEBP).
- `--optimize`, `--progressive`, `--png-compress-level`, `--png-optimize`, `--webp-quality`, `--webp-lossless`, `--dpi` : Additional output controls.

## Single-file pipeline

Apply default filters and write output (auto-detect input if omitted):

```bash
python3 main.py
```

Specify input, output, and filters (example forces quality):

```bash
python3 main.py -i in.jpg -o out.png -f sepia -f "unsharp:1.2,2" --quality 85
```

Dry-run with verbose output to see what will happen:

```bash
python3 main.py -i in.jpg -o out.png -f grayscale --dry-run -v
```

Force output format and optimization flags:

```bash
python3 main.py -i photo.heic -o photo.jpg --format JPEG --quality 90 --optimize --progressive
```

## Batch processing (shell examples)

`main.py` processes a single image per invocation. Use shell loops or `find`/`xargs` to process many files.

Simple bash loop (preserves filenames):

```bash
mkdir -p out
for f in photos/*.{jpg,png}; do
    outfile="out/$(basename "${f%.*}").jpg"
    python3 main.py -i "$f" -o "$outfile" -f "resize:800x600" --quality 80
done
```

Using `find` + `xargs` (parallelizable):

```bash
mkdir -p out
find photos -type f \( -iname '*.jpg' -o -iname '*.png' \) -print0 \
    | xargs -0 -n1 -P4 -I{} bash -c 'outfile="out/$(basename "${1%.*}").jpg"; python3 main.py -i "$1" -o "$outfile" -f "grayscale" --quality 75' -- {}
```

## Auto-detect input

If you omit `-i/--input`, `main.py` will try to auto-detect a suitable input image in the current working directory (see `main.py` docstring).

```bash
# Run auto-detect + default filters
python3 main.py -o result.png
```

## Quick try (one-liner)

From the `imageprocessing` folder (adjust path to `main.py` if running from elsewhere):

```bash
python3 main.py -i tests/example.jpg -o out/example_out.jpg -f "sharpen" -v
```

## Notes & next steps

- `main.py` is the recommended entrypoint for CLI use — it calls `read.load_image`, `process.apply_filters`, and `write.save_image` with safe defaults and backup behavior.
- If you want, I can also add a `--batch` mode to `main.py` to process entire directories directly (so you don't need external loops), or add a `--config` option to load filter chains from YAML/JSON.

---

File: `README.md` updated to show CLI-first usage for `main.py`.
