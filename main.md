# Main Module Documentation

The `main` module is the application's runtime coordinator. It receives the parsed CLI options, builds a configuration object, wires together the `read`, `process`, and `write` modules, and controls the execution lifecycle.

## Responsibilities

- Accept a single parsed-options object from the CLI and normalize it into an internal configuration.
- Validate cross-parameter constraints (for example: `filter` requires `input` to exist).
- Initialize logging, temporary folders, and other runtime resources.
- Sequence calls: Read → Process → Write, while propagating configuration and metadata.
- Catch and translate low-level exceptions to user-facing error messages and non-zero exit codes.

## Lifecycle and Example Flow

1. Parse CLI (delegated to `cli`), receive options.
2. Create a runtime `config` object containing: IO paths, processing parameters, concurrency settings, and logging level.
3. Initialize logging according to `config.log_level`.
4. Call `read.load(input_path, config)` to obtain an image buffer or stream.
5. Call `process.apply(image, config)` to obtain a processed image object.
6. Call `write.save(processed_image, output_path, config)`.
7. Handle cleanup and return appropriate exit code.

## Error Handling

- Prefer short, actionable error messages for the user. Include a suggestion when possible (e.g., "file not found — check path or use `--input` to point to a valid file").
- Log detailed tracebacks only when `--verbose` is set; otherwise keep logs concise.

## Extensibility

- Keep orchestration logic minimal and testable. Business logic belongs in `process`, file I/O logic belongs in `read`/`write`.
- To add a processing step, add it to the `process` module and call it from the pipeline in a single place within `main`.
