# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Run the MCP server, browser workbench, or an explicit LLM rewriting job."""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.resources import files
from pathlib import Path

from dotenv import load_dotenv

from .research_rules import validate_manifest


def load_manifest(path: Path | None = None) -> dict:
    """Load and verify a custom manifest or the packaged default pipeline."""
    resource = (
        path
        if path is not None
        else files("fakehuman").joinpath("data/current_rules.json")
    )
    manifest = json.loads(resource.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main() -> None:
    load_dotenv(".env")
    parser = argparse.ArgumentParser(
        description="Run text post-processing over MCP, a browser UI, or remote LLM rewriting."
    )
    parser.add_argument(
        "--pipeline",
        type=Path,
        help="custom pipeline manifest; defaults to the bundled audited pipeline",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use cached models only; disable model lookups and remote LLM requests",
    )
    commands = parser.add_subparsers(dest="command")
    download = commands.add_parser(
        "download-models",
        help="prefetch all three pinned model checkpoints without loading them",
    )
    download.add_argument(
        "--offline",
        action="store_true",
        default=argparse.SUPPRESS,
        help="verify the required model files are cached without network access",
    )
    browser = commands.add_parser("ui", help="open the optional NiceGUI test interface")
    browser.add_argument(
        "--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)"
    )
    browser.add_argument(
        "--port", type=_port, default=8080, help="HTTP port (default: 8080)"
    )
    browser.add_argument(
        "--offline",
        action="store_true",
        default=argparse.SUPPRESS,
        help="use cached models only; disable model lookups and remote LLM requests",
    )
    browser.add_argument(
        "--pipeline",
        type=Path,
        default=argparse.SUPPRESS,
        help="custom pipeline manifest; defaults to the bundled audited pipeline",
    )
    rewrite = commands.add_parser(
        "rewrite",
        help="rewrite with remote LLMs, preservation checks and local scoring",
    )
    source = rewrite.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="UTF-8 file, or - to read standard input")
    source.add_argument("--text", help="literal text to rewrite")
    rewrite.add_argument("--output", type=Path, help="run directory; reuse to resume")
    rewrite.add_argument("--rounds", type=int, choices=(1, 2, 3), default=3)
    rewrite.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    rewrite.add_argument("--section-tokens", type=int, default=360)
    rewrite.add_argument("--request-timeout", type=float, default=120)
    rewrite.add_argument(
        "--strategy",
        choices=("adaptive", "technical-update-experimental"),
        default="adaptive",
    )
    rewrite.add_argument(
        "--input-format", choices=("auto", "plain", "markdown", "html"), default="auto"
    )
    rewrite.add_argument("--editor-model", help="adaptive editor model ID")
    rewrite.add_argument("--reviewer-model", help="adaptive reviewer model ID")
    rewrite.add_argument(
        "--review-policy",
        choices=("strict", "contextual"),
        default="strict",
        help="sentence editing: strict local checks or experimental source-quoted context review",
    )
    rewrite.add_argument(
        "--model",
        action="append",
        help="technical generator model ID; repeat for a portfolio",
    )
    rewrite.add_argument(
        "--reviewer",
        action="append",
        help="technical reviewer model ID; provide at least three",
    )
    rewrite.add_argument(
        "--offline",
        action="store_true",
        default=argparse.SUPPRESS,
        help="disable remote LLM requests (rewriting will report an error)",
    )
    args = parser.parse_args()
    if args.offline:
        # These libraries read their environment on import. Keep their imports
        # below this point so CLI offline mode applies to the entire process.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["FASTMCP_CHECK_FOR_UPDATES"] = "off"
        os.environ["FAKEHUMAN_OFFLINE"] = "1"

    if args.command == "rewrite":
        from .llm_client import validate_configuration
        from .process import process_text
        from .run_storage import new_run_directory

        try:
            validate_configuration()
            if args.strategy == "technical-update-experimental" and (
                args.editor_model or args.reviewer_model
            ):
                raise ValueError(
                    "Use --model and --reviewer for the technical strategy"
                )
            text = (
                args.text
                if args.text is not None
                else sys.stdin.read()
                if args.input == "-"
                else Path(args.input).read_text(encoding="utf-8", newline="")
            )
            import torch

            torch.set_num_threads(4)
            options = {}
            if args.editor_model is not None:
                options["editor_model"] = args.editor_model
            if args.reviewer_model is not None:
                options["reviewer_model"] = args.reviewer_model
            if args.review_policy != "strict":
                options["review_policy"] = args.review_policy
            report = process_text(
                text,
                output_dir=args.output or new_run_directory(),
                rounds=args.rounds,
                workers=args.workers,
                section_tokens=args.section_tokens,
                request_timeout=args.request_timeout,
                strategy=args.strategy,
                input_format=args.input_format,
                models=args.model,
                reviewers=args.reviewer,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
                **options,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            parser.exit(1, f"{parser.prog}: {exc}\n")
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        if report.get("status") == "provider_error":
            parser.exit(1)
        return

    manifest = load_manifest(args.pipeline)

    if args.command == "download-models":
        from .models import download_models

        try:
            report = download_models(offline=args.offline)
        except RuntimeError as exc:
            parser.exit(1, f"{parser.prog}: {exc}\n")
        print(json.dumps({"models": report}, indent=2))
        return

    if args.command == "ui":
        from .ui import run_ui

        try:
            run_ui(manifest, host=args.host, port=args.port)
        except RuntimeError as exc:
            parser.exit(1, f"{parser.prog}: {exc}\n")
        return

    import torch

    from .research_mcp import make_server

    torch.set_num_threads(4)
    make_server(manifest).run(transport="stdio")
