# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**age** (Android GUI Executor) is a Python framework for AI-powered Android/web GUI automation. It uses LLM vision models to perform tasks on Android devices via ADB, or web environments via ScaleWoB.

## Common Development Commands

```bash
# Run all checks (format, lint, type check)
poe check

# Format code and sort imports
poe format

# Auto-fix linting issues
poe fix

# Run tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_brownie_agent.py -v
```

## Architecture

The codebase has three layers: API client, devices, and agents.

### Package Structure

```
age/
├── __init__.py          # Re-exports all public types; initializes logging on import
├── api.py               # APIClient — HTTP and Vertex AI LLM backends
├── tracker.py           # ActionTracker — screenshots + execution traces
├── video_recorder.py    # Scrcpy-based video recording
├── devices/
│   ├── base.py          # Device ABC + Direction enum
│   ├── android.py       # AndroidDevice — ADB via adbutils
│   └── scalewob.py      # ScaleWoBDevice — web automation via scalewob package
└── agents/
    ├── base.py          # Agent ABC — tracking, step limiting, context manager
    ├── doubao.py        # DoubaoAgent — Doubao/Ark models, Android
    ├── mai.py           # MAIAgent — XML tool-call format, Android
    ├── rgm.py           # RGMAgent — function-call JSON format, Android
    ├── brownie.py       # BrownieAgent — device.* code-style prompt, Android
    ├── doubao_scalewob.py     # DoubaoScaleWoBAgent — ScaleWoB variant
    ├── mai_scalewob.py        # MAIScaleWoBAgent
    ├── rgm_scalewob.py        # RGMScaleWoBAgent
    ├── brownie_scalewob.py    # BrownieScaleWoBAgent
    ├── general_scalewob.py    # GeneralScaleWoBAgent
    ├── open_autoglm_scalewob.py
    └── mobileagent35_scalewob.py
```

### Device Layer

`Device` (ABC) defines the interface all devices implement:
- `capture_screenshot(format: "PIL" | "B64")` — returns PIL Image or base64 string
- `size` property — returns `(width, height)` in pixels

**`AndroidDevice`** (`devices/android.py`):
- Connects via ADB using `adbutils`; auto-selects first device if `device_id=None`
- On context manager entry (`__enter__`), switches IME to `com.android.adbkeyboard/.AdbIME`; restores original IME on exit
- Resolves app names to packages via a content provider (`com.mobilellm.appscan.provider`), with case-insensitive and fuzzy matching fallback

**`ScaleWoBDevice`** (`devices/scalewob.py`):
- Wraps `ScaleWoBAutomation` from the `scalewob` package
- Used for web-based task evaluation; video recording always returns `False`
- Hardcoded `size`: `(3840, 2400)` desktop / `(1170, 2532)` mobile

### Agent Layer

`Agent` (ABC in `agents/base.py`) provides:
- Tracking lifecycle: `_init_tracker()`, `_record_action()`, `_finalize_tracking()`
- Step limiting: `_check_step_limit()` — returns a forced `finished` Action when `max_steps` exceeded
- Context manager: delegates device `__enter__`/`__exit__` and finalizes tracking on exit

**Always use agents as context managers** to ensure IME restoration and tracking finalization:
```python
with DoubaoAgent(model_name="...", task_name="...") as agent:
    while True:
        action = agent.predict()
        if action["action_type"] == "finished":
            break
```

#### Agent Variants and Prompt Formats

| Agent | Prompt format | History strategy |
|-------|--------------|-----------------|
| `DoubaoAgent` | `Thought: / Action: func(args)` | Sliding window (last 5 screenshots) |
| `MAIAgent` | `<thinking>...</thinking> <tool_call>{json}</tool_call>` | Configurable `history_n` (default 3) |
| `RGMAgent` | System prompt with tool schema; `<tool_call>` in response | Stateless per-turn (no history) |
| `BrownieAgent` | `Thought: / Action: \`device.method()\`` | Step log (thought+action strings) |

`*ScaleWoBAgent` variants are identical in prompt format but use `ScaleWoBDevice` and call `device.eval(task_id, params)` on `finished` to get a success/failure grade.

### API Client (`api.py`)

`APIClient` supports two backends, selected per-instance:
- **HTTP** (default): OpenAI-compatible REST endpoint; reads `{PREFIX}_BASE_URL` and `{PREFIX}_API_KEY` from env (e.g. `DOUBAO_BASE_URL`, `DOUBAO_API_KEY`)
- **Vertex AI** (`use_vertex_api=True`): routes through LiteLLM; requires `GOOGLE_APPLICATION_CREDENTIALS`, `VERTEX_PROJECT`, optionally `VERTEX_LOCATION`

Each agent passes its `name` as `prefix` to `APIClient`, so different agents can use different API endpoints from the same `.env`.

### Coordinate System

All agents use a **0–1000 normalized coordinate scale** in prompts (except `MAIAgent` which uses 0–999). `_normalize_coordinates(x, y)` converts to device pixels. The abstraction lets the same prompt templates work across devices with different resolutions.

### Action Tracking System

When `enable_tracking=True`, `ActionTracker` captures annotated screenshots and a video (via scrcpy) for every step. Output is saved to:
```
tracks/task_YYYYMMDD_HHMMSS_<name>/
├── trace.md, actions.json, trace.json
├── recording.mp4
└── screenshots/step_NNN_<action>.png
```
Tracking is finalized automatically on `finished` action or on context manager exit.

## Environment Setup

Required in `.env`:
```
# Per-agent prefix pattern (e.g., DOUBAO_BASE_URL, RGM_API_KEY)
{AGENT_NAME}_BASE_URL=https://...
{AGENT_NAME}_API_KEY=your_key

# Or generic fallback
BASE_URL=https://api.openai.com/v1/chat/completions
API_KEY=your_api_key

# For Vertex AI
GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json
VERTEX_PROJECT=my-project
VERTEX_LOCATION=us-central1
```

## ADB Requirements

- Android device connected via ADB
- `com.android.adbkeyboard/.AdbIME` installed for text input (auto-enabled by `AndroidDevice`)
- `com.mobilellm.appscan.provider` content provider installed for app name resolution
- **scrcpy** (optional) — required for video recording; gracefully degrades if absent
