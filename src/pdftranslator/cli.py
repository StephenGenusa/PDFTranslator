"""Command-line interface for the PDF translator."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from .orchestrator import Orchestrator
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

    translator = Translator(
        model=model,
        target_lang=target_lang,
        api_key=api_key,
        api_base=api_base,
        cache_dir=str(cache_dir),
    )
    if ignore_cache:
        translator.clear_cache()

    orchestrator = Orchestrator(translator, mirror=(mode == "side-by-side"))
    try:
        orchestrator.run(input_file, output_file, pages)
    finally:
        translator.close()

    typer.secho(f"Done -> {output_file}", fg="green")


if __name__ == "__main__":
    app()
