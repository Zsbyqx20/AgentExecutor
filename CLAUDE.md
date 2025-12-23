# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**age** (Android GUI Executor) is a Python framework for AI-powered Android GUI automation. It uses computer vision and LLMs to perform tasks on Android devices via ADB.

## Common Development Commands

```bash
# Run all checks (format, lint, type check)
poe check

# Format code and sort imports
poe format

# Auto-fix linting issues
poe fix

# Apply unsafe fixes
poe fix_unsafe

# Type check only
pyright
```

## Architecture

The codebase follows a three-layer architecture:

### Core Modules (`age/`)

1. **`device.py`** - Device Interface Layer
   - Manages ADB communication with Android devices
   - Auto-detects devices or uses specified `device_id`
   - Handles screenshots (PIL Image or base64), touch input, text entry, navigation
   - Automatically sets ADB keyboard as IME for text input

2. **`api.py`** - API Client Layer
   - Generic LLM API client (OpenAI-compatible)
   - Reads `BASE_URL` and `API_KEY` from environment variables
   - Built-in retry logic (max 3 retries)

3. **`agent.py`** - Agent Logic Layer
   - `Agent` (abstract base class) - extensible for different LLM providers
   - `DoubaoAgent` - concrete implementation for Doubao/Ark models
   - Orchestrates the predict-execute loop: screenshot → LLM → parse action → execute

### Execution Flow

```
DoubaoAgent.predict()
  ├─ capture_screenshot() → base64 image
  ├─ client.query(model, messages) → LLM response
  ├─ _parse_action(response_text) → Action dict
  └─ _execute(action) → Device interaction
```

### Coordinate System

The LLM uses normalized coordinates (0-1000 scale) which are converted to actual device pixels in `_normalize_coordinates()`. This abstraction allows the same prompts to work across different screen sizes.

### Action Space

Defined in `DOUBAO_PROMPT`:
- `click(point)`, `long_press(point)`
- `type(content)` - use `\n` for submit
- `scroll(point, direction)`
- `drag(start_point, end_point)`
- `press_home()`, `press_back()`
- `finished(content)` - terminates the task loop

### History Management

The `_compress_history()` method implements a sliding window for screenshots, keeping only the last 5 screenshots to manage token usage while maintaining conversation context.

## Environment Setup

Required in `.env`:
```
BASE_URL=https://api.openai.com/v1/chat/completions  # or custom endpoint
API_KEY=your_api_key_here
```

## ADB Requirements

- Android device connected via ADB
- ADB keyboard (`com.android.adbkeyboard/.AdbIME`) will be automatically enabled
- Device auto-selection uses the first available device if `device_id` is not specified

## Dependencies

- `adbutils` - ADB communication
- `PIL` - Image processing
- `requests` - HTTP client for LLM API
- `python-dotenv` - Environment configuration
- `rich` - Terminal output (available but not yet utilized)
