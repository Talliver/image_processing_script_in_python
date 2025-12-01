# CLI Module Documentation

This document explains the responsibilities and behaviour of the project's CLI adapter. The CLI module's job is to convert raw command-line input into a validated, structured set of options that the rest of the program (primarily `main.py`) consumes.

## Responsibilities

- Parse command-line arguments into a structured config object.
- Validate required values and provide clear error messages for invalid input.
- Provide user-friendly help/usage text.
- Normalize and sanitize values (file paths, numeric ranges, enumerations).
- Forward the parsed arguments to `main` in a single structure (for example, a dict or simple dataclass).

## Design Notes

- Implementation should be minimal: parsing + validation only. Keep parsing logic decoupled from processing logic.
- Do not hard-code processing defaults here; defaults that affect algorithms belong in `commands.md` or `process` module configuration. See `commands.md` for a canonical list of CLI flags and recommended values.

## Behaviour and Integration

- On success: return a parsed options object and exit with status `0` only when the program runs to completion.
- On invalid input: print a single-line summary of the error and the `--help` hint, then exit with a non-zero status.
- For interactive or verbose modes: expose a `--verbose` switch that increases log verbosity in `main` rather than duplicating verbose handling in the CLI.

## Example usage

`python main.py --input input.png --output out.png --filter blur --resize 800x600`

Refer to `commands.md` for full flag descriptions, recommended values, and examples of common workflows.
