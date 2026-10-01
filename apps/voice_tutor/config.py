"""The only place that reads the environment."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from dotenv import dotenv_values


@dataclass(frozen=True)
class Keys:
    anthropic: str = field(repr=False)
    gemini: str = field(repr=False)
    elevenlabs: str = field(repr=False)


@dataclass(frozen=True)
class Settings:
    port: int
    keys: Keys | None
    """None in fake mode, which needs no keys."""
    diagram_model: Literal["gemini", "anthropic"]


def load_settings(argv: Sequence[str]) -> Settings:
    parser = argparse.ArgumentParser(prog="tutor", description="A voice tutor with a whiteboard.")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="run without API keys: scripted models, silent speech, typed speech",
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    for name, value in dotenv_values().items():
        if value and not os.environ.get(name):
            os.environ[name] = value

    diagram_model = os.environ.get("DIAGRAM_MODEL", "gemini")
    if diagram_model not in ("gemini", "anthropic"):
        parser.error("DIAGRAM_MODEL must be gemini or anthropic")
    if args.fake:
        return Settings(args.port, None, diagram_model)

    needed = ["ANTHROPIC_API_KEY", "ELEVENLABS_API_KEY"]
    if diagram_model == "gemini":
        needed.append("GEMINI_API_KEY")
    missing = [name for name in needed if not os.environ.get(name)]
    if missing:
        parser.error(f"missing {', '.join(missing)} in .env (see .env.example), or run with --fake")
    keys = Keys(
        anthropic=os.environ["ANTHROPIC_API_KEY"],
        gemini=os.environ.get("GEMINI_API_KEY", ""),
        elevenlabs=os.environ["ELEVENLABS_API_KEY"],
    )
    return Settings(args.port, keys, diagram_model)
