"""CLI wrapper that uses the core pipeline in `main.py`.

This file provides a standalone CLI entrypoint while keeping the
implementation in `main.py` untouched.
"""
import argparse
import sys
from typing import Optional, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from main import run_pipeline, _parse_filters


console = Console()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Image processing CLI wrapper")
    p.add_argument('-i', '--input', help='Input image path (optional, auto-detect in cwd if omitted)')
    p.add_argument('-o', '--output', default='output_image.jpg', help='Output image path')
    p.add_argument('-f', '--filter', action='append', default=[], help='Filter to apply (can be repeated or comma-separated)')
    p.add_argument('--overwrite', action='store_true', help='Allow overwriting existing output file')
    p.add_argument('--no-backup', action='store_true', help='Do not create a backup of existing output')
    p.add_argument('--dry-run', action='store_true', help='Print planned actions without writing files')
    p.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    p.add_argument('--format', help='Force output format (e.g. JPEG, PNG, WEBP)')
    p.add_argument('--quality', type=int, help='Quality for lossy formats (JPEG/WEBP)')
    p.add_argument('--optimize', action='store_true', help='Optimize output (where supported)')
    p.add_argument('--progressive', action='store_true', help='Create progressive JPEG where supported')
    p.add_argument('--png-compress-level', type=int, help='PNG compress level 0-9')
    p.add_argument('--png-optimize', action='store_true', help='PNG optimize flag')
    p.add_argument('--webp-quality', type=int, help='WEBP quality 0-100')
    p.add_argument('--webp-lossless', action='store_true', help='WEBP lossless')
    p.add_argument('--dpi', nargs=2, type=int, metavar=('X','Y'), help='DPI to embed in output')
    return p


def main(argv=None):

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    def process_args(args):
        filters = _parse_filters(args.filter) if args.filter else ['grayscale', 'blur', 'sharpen']
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="cyan", no_wrap=True)
        table.add_column()
        table.add_row("Input:", str(args.input or "(auto-detect)"))
        table.add_row("Output:", args.output)
        table.add_row("Filters:", ", ".join(filters))
        table.add_row("Overwrite:", str(args.overwrite))
        table.add_row("Dry run:", str(args.dry_run))
        console.print(Panel(table, title="Image Processing", subtitle="Starting...", expand=False))
        try:
            with console.status("Processing image…", spinner="dots"):
                result = run_pipeline(
                    input_path=args.input,
                    output_path=args.output,
                    filters=filters,
                    overwrite=args.overwrite,
                    no_backup=args.no_backup,
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
            console.print(Panel("[green]Pipeline finished successfully[/green]", expand=False))
            console.print(result)
        except Exception as e:
            console.print(Panel(f"[bold red]Error:[/bold red] {e}"))
            return False
        return True

    # If input is missing, enter interactive CLI mode
    if not args.input:
        help_text = (
            "[bold yellow]Interactive CLI Mode Help[/bold yellow]\n"
            "Type your image processing command as you would after 'cli.py'.\n"
            "For example: [green]-i input.jpg -o out.png -f grayscale -f blur[/green]\n"
            "Type [cyan]help[/cyan] to see this message again.\n"
            "Type [cyan]exit[/cyan] or [cyan]quit[/cyan] to leave interactive mode.\n"
            "Use [cyan]-h[/cyan] or [cyan]--help[/cyan] after any command to see available options."
        )
        console.print(Panel("[yellow]No input image path provided. Entering interactive CLI mode.[/yellow]", expand=False))
        console.print(Panel(help_text, title="How to Use Interactive CLI", expand=False))
        while True:
            try:
                user_input = console.input("[bold cyan]cli> [/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                console.print("\nExiting CLI.")
                break
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            if user_input.strip().lower() == "help":
                console.print(Panel(help_text, title="How to Use Interactive CLI", expand=False))
                continue
            if not user_input.strip():
                continue
            try:
                user_args = parser.parse_args(user_input.split())
                process_args(user_args)
            except SystemExit:
                # argparse throws this on error
                pass
            except Exception as e:
                console.print(Panel(f"[bold red]Error:[/bold red] {e}"))
        sys.exit(0)
    else:
        success = process_args(args)
        sys.exit(0 if success else 2)


if __name__ == '__main__':
    main()
