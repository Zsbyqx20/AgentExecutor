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

### Supporting Modules (`age/`)

- **`logging_config.py`** - Centralized logging configuration with dual-output setup
- **`logger.py`** - Logger factory for consistent logger instances
- **`icons.py`** - Nerd Font icons for terminal output
- **`tracker.py`** - Action tracking system for generating execution traces with annotated screenshots

### Core Modules (`age/`)

1. **`device.py`** - Device Interface Layer
   - Manages ADB communication with Android devices
   - Auto-detects devices or uses specified `device_id`
   - Handles screenshots (PIL Image or base64), touch input, text entry, navigation
   - Automatically sets ADB keyboard as IME for text input
   - Maintains app name-to-package mapping with fuzzy matching for flexible app launching

2. **`api.py`** - API Client Layer
   - Generic LLM API client (OpenAI-compatible)
   - Reads `BASE_URL` and `API_KEY` from environment variables
   - Built-in retry logic (max 3 retries)

3. **`agent.py`** - Agent Logic Layer
   - `Agent` (abstract base class) - extensible for different LLM providers
   - `DoubaoAgent` - concrete implementation for Doubao/Ark models
   - Orchestrates the predict-execute loop: screenshot → LLM → parse action → execute
   - Implements notepad functionality for persistent note-taking during task execution
   - Includes wait action for pausing during loading screens
   - Optional action tracking for generating execution traces with annotated screenshots

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

Defined in the prompt components (`_BASE_ACTIONS`, `_NOTE_ACTIONS`, `_FINISHED_ACTION`):
- `click(point)`, `long_press(point)`
- `type(content)` - use `\n` for submit
- `scroll(point, direction)`
- `drag(start_point, end_point)`
- `press_home()`, `press_back()`
- `open_app(app_name)` - with fuzzy matching support
- `wait(seconds)` - for pausing during loading screens
- `note_create/edit/remove(content)` - for persistent note-taking (optional feature)
- `finished(content)` - terminates the task loop

### Logging System

The project uses a professional dual-output logging system with Rich:
- **Console**: Clean INFO+ level output with colored Nerd Font icons
- **Files**: Detailed DEBUG+ logs with full context in `logs/` directory
  - `agent_debug.log` - All debug logs (10MB per file, 5 backups)
  - `agent_errors.log` - Error-only logs (10MB per file, 10 backups)
- Configurable via `setup_logging()` in `logging_config.py`

### History Management

The `_compress_history()` method implements a sliding window for screenshots, keeping only the last 5 screenshots to manage token usage while maintaining conversation context.

### Action Tracking System

The `ActionTracker` class provides automatic action execution tracing with annotated screenshots and video recording.

**Features:**
- Records each action with timestamp, parameters, and LLM thought
- Annotates screenshots with visual markers showing where actions occurred
- Records video of entire task execution
- Generates markdown execution trace documents

**Usage:**
```python
from age.agent import DoubaoAgent

agent = DoubaoAgent(
    model_name="doubao-seed-1-6-vision-250815",
    task_name="在设置中开启深色模式",
    enable_tracking=True,  # Enable action tracking (includes video)
    tracking_output_dir="./tracks",  # Optional: custom output directory
)

# Run the task loop
while True:
    action = agent.predict()
    if action["action_type"] == "finished":
        break

# Trace and video automatically saved to tracks/task_YYYYMMDD_HHMMSS_<name>/
```

**Output Structure:**
```
tracks/
└── task_20241224_201530_open_settings/
    ├── trace.md              # Markdown execution trace with video section
    ├── recording.mp4         # Video of entire task execution
    └── screenshots/
        ├── step_001_open_app.png
        ├── step_002_click.png
        └── ...
```

**Video Recording:**
- Starts automatically when tracking is enabled
- Records entire task execution from start to finish
- Saved as MP4 format in task directory
- Referenced in markdown trace with embedded HTML5 video player
- Gracefully handles recording failures without disrupting task execution

**Action Visualizations:**
- Click/Long Press: Green circle with center dot
- Scroll: Yellow circle with directional arrow
- Drag: Green start point, red end point, purple connecting line
- Type: Blue keyboard area with text preview
- System Actions (Home/Back/Open App): Centered text overlay
- Wait: Clock icon with seconds label
- Notepad: Note icon with content preview

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
- `aiohttp` - Async HTTP client
- `httpx[socks]` - HTTP client with SOCKS support
- `PIL` - Image processing
- `requests` - HTTP client for LLM API
- `python-dotenv` - Environment configuration
- `rich` - Professional terminal output and logging

The project uses `uv` for dependency management with a locked `uv.lock` file.
