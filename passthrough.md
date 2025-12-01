# Program Passthrough Structure

This document gives a concise, single-place view of the program's control and data flow. It intentionally avoids repeating flag-level details or implementation notes that live in `commands.md`, `cli.md`, or the per-module docs.

## High-level Flow

1. User invokes program with flags.
2. `cli` parses flags into a validated options object.
3. `main` converts options into an internal `config` and initializes runtime resources.
4. `main` calls `read.load(input_path, config)` to obtain the input image object.
5. `main` calls `process.apply(image, config)` to perform the requested pipeline of operations.
6. `main` calls `write.save(result, output_path, config)` to persist the result.
7. `main` performs cleanup and exits with an appropriate status code.

## Responsibilities (one-line mapping)

- CLI: parse and validate raw flags (see `cli.md`).
- Main: orchestration, logging, and error translation (see `main.md`).
- Read: input decoding and metadata extraction (see `read.md`).
- Process: transformation pipeline and parameter validation (see `process.md`).
- Write: safe persistence and metadata preservation (see `write.md`).

## Error propagation

- Modules should prefer raising module-specific exceptions (e.g., `ReadError`, `ProcessingError`, `WriteError`).
- `main` is responsible for catching these, emitting user-facing messages, and returning non-zero exit codes.

## Notes

- For CLI flag details and recommended values, consult `commands.md`. This file focuses on sequencing and responsibilities only.
