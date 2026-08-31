"""Command-line interface for the PDF translator."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer

from .orchestrator import Orchestrator, PageRangeError, parse_page_spec
from .translate import Translator

app = typer.Typer(add_completion=False, help="Layout-preserving PDF translator.")


@app.command()
def main(
    input_file: Path = typer.Option(
        ..., "--input", "-i", exists=True, dir_okay=False, readable=True,
        help="Source PDF.",
    ),
    output_file: Path = typer.Option(..., "--output", "-o", help="Destination PDF."),
    target_lang: str = typer.Option("English", "--target", "-t", help="Target language."),
    model: str = typer.Option(
        "openrouter/google/gemini-2.5-flash-lite", "--model", "-m",
        help=(
            "LiteLLM model string. Examples: "
            "'openrouter/google/gemini-2.5-flash-lite', "
            "'anthropic/claude-haiku-4-5-20251001', "
            "'ollama/llama3', "
            "'openai/gpt-4o'."
        ),
    ),
    mode: str = typer.Option(
        "side-by-side", "--mode",
        help="'side-by-side' (original | translation) or 'substitute' (translation only).",
    ),
    pages: str = typer.Option("all", "--pages", "-p", help="e.g. '1-5,7' or 'all'."),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", envvar="PDF_TRANSLATE_API_KEY",
        help="API key override. Usually not needed — LiteLLM reads the provider's own env var "
             "(ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, etc.) automatically. "
             "Use this only to override. Also read from PDF_TRANSLATE_API_KEY.",
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base",
        help="Override the provider base URL, e.g. 'http://localhost:11434' for Ollama "
             "or 'http://localhost:1234/v1' for LM Studio.",
    ),
    cache_dir: Path = typer.Option(
        Path(".translation_cache"), "--cache-dir", help="Translation cache directory."
    ),
    ignore_cache: bool = typer.Option(
        False, "--ignore-cache", help="Clear the translation cache before running."
    ),
    request_pause: float = typer.Option(
        0.0, "--request-pause",
        help="Seconds to wait before each API call. Use to stay under a "
             "provider's rate limit.",
    ),
    max_tokens: int = typer.Option(
        8192, "--max-tokens",
        help="Maximum tokens per model response. Raise it if dense pages come "
             "back truncated; lower it to cap per-call cost.",
    ),
    max_attempts: int = typer.Option(
        3, "--max-attempts",
        help="Attempts per API call before giving up on a page, and retries of "
             "a batch that parses incompletely.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if mode not in {"side-by-side", "substitute"}:
        typer.secho(f"Unknown --mode {mode!r}", fg="red")
        raise typer.Exit(1)

    try:
        _validate_output(input_file, output_file)
    except ValueError as exc:
        typer.secho(str(exc), fg="red")
        raise typer.Exit(1)

    # Validate the page spec before doing any work, so a typo costs nothing.
    try:
        parse_page_spec(pages)
    except PageRangeError as exc:
        typer.secho(str(exc), fg="red")
        raise typer.Exit(1)

    if max_attempts < 1:
        typer.secho("--max-attempts must be at least 1", fg="red")
        raise typer.Exit(1)
    if request_pause < 0:
        typer.secho("--request-pause cannot be negative", fg="red")
        raise typer.Exit(1)

    translator = Translator(
        model=model,
        target_lang=target_lang,
        api_key=api_key,
        api_base=api_base,
        cache_dir=str(cache_dir),
        request_pause=request_pause,
        max_tokens=max_tokens,
        max_attempts=max_attempts,
    )
    if ignore_cache:
        translator.clear_cache()

    orchestrator = Orchestrator(translator, mirror=(mode == "side-by-side"))
    try:
        orchestrator.run(input_file, output_file, pages)
    except PageRangeError as exc:
        typer.secho(str(exc), fg="red")
        raise typer.Exit(1)
    finally:
        translator.close()

    typer.secho(f"Done -> {output_file}", fg="green")


def _validate_output(input_file: Path, output_file: Path) -> None:
    """Reject an unusable --output before a single API call is paid for.

    Everything here used to surface only at the final writer.save(), i.e. after
    the whole document had been translated.
    """
    if output_file.is_dir():
        raise ValueError(
            f"Output path {str(output_file)!r} is a directory. Give a file path."
        )

    try:
        same = input_file.resolve() == output_file.resolve()
    except OSError:
        same = False
    if same:
        raise ValueError(
            "Output path is the same file as the input. Choose a different "
            "--output so the source PDF is not overwritten."
        )

    parent = output_file.parent if str(output_file.parent) else Path(".")
    if not parent.is_dir():
        raise ValueError(
            f"Output directory {str(parent)!r} does not exist. Create it first."
        )
    if not os.access(parent, os.W_OK):
        raise ValueError(f"Output directory {str(parent)!r} is not writable.")


if __name__ == "__main__":
    app()
