from __future__ import annotations

import logging
import sys

from nicegui import app, ui

from apps.voice_tutor.compose import build
from apps.voice_tutor.config import load_settings
from apps.voice_tutor.ui import install


def main() -> None:
    settings = load_settings(sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    voice_tutor = build(settings)
    install(voice_tutor)
    app.on_startup(voice_tutor.start)
    app.on_shutdown(voice_tutor.aclose)
    ui.run(port=settings.port, title="Voice tutor", reload=False, show=True)


if __name__ == "__main__":
    main()
