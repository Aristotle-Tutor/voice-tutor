# Voice tutor

A tutor you talk to out loud. It listens through your browser's microphone,
answers with a voice, and can ask a second agent to draw diagrams on a shared
whiteboard while it keeps talking. You can interrupt it at any time.

Everything is Python except two small scripts that move audio between the
browser and the server.

## Run it

You need [uv](https://docs.astral.sh/uv/) and three API keys: Anthropic (the
tutor), Gemini (the diagram agent), and ElevenLabs (speech to text and text to
speech).

```sh
cp .env.example .env      # then fill in the keys
uv run tutor
```

A browser tab opens. Click **Voice on** and allow the microphone. Wear
headphones: on laptop speakers the tutor can hear itself and stop talking.

You can also type. **Send** delivers text as a typed message. **Say it aloud**
plays your text as if you had spoken it, at talking speed, which is handy
without a microphone.

`uv run tutor --fake` runs the whole app with no keys and no sound. The
scripted tutor repeats back what you say, so how much you type sets how long
it talks. The scripted diagram agent takes about 8 seconds.

## Where things are

| Folder | What it holds |
| --- | --- |
| `events/` | The bus and every message type that crosses between services |
| `providers/` | One adapter per vendor (Anthropic, Gemini, ElevenLabs), the shared types they speak, and test fakes |
| `packages/voice/` | Audio plumbing: microphone to transcript, text to speech, the browser connection |
| `services/` | The tutor agent, the diagram agent, and the whiteboard, plus the agent loop they share |
| `apps/voice_tutor/` | The entry point: reads `.env`, picks vendors, wires everything up, serves the page |

[ARCHITECTURE.md](ARCHITECTURE.md) explains how the pieces fit together.

## Checks

```sh
uv run pytest               # offline tests
uv run pytest -m live       # the same provider tests against the real vendors (needs .env)
uv run ruff check .
uv run basedpyright
uv run lint-imports         # the import rules between folders
```
