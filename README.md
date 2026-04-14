# age — Android GUI Executor

A Python framework for AI-powered Android and web GUI automation. Vision-language models observe screenshots and emit structured actions executed on a real Android device (via ADB) or a web environment (via ScaleWoB).

## Requirements

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- For Android tasks: ADB-connected device with `com.android.adbkeyboard/.AdbIME` installed
- For video recording: `scrcpy` (optional, gracefully skipped if absent)
- For ScaleWoB tasks: `scalewob` package (included in dependencies)

## Installation

```bash
# Clone and install with uv (uses uv.lock for reproducible installs)
git clone <repo-url>
cd AgentExecutor
uv sync

# Or install with pip
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and fill in the API credentials for each agent you plan to use:

```bash
cp .env.example .env
```

```dotenv
# Per-agent credentials — variable prefix matches the agent name
DOUBAO_BASE_URL=https://...
DOUBAO_API_KEY=sk-...

RGM_BASE_URL=https://...
RGM_API_KEY=sk-...

BROWNIE_BASE_URL=https://...
BROWNIE_API_KEY=sk-...

MAI_BASE_URL=https://...
MAI_API_KEY=sk-...
```

Each agent reads `{AGENT_NAME}_BASE_URL` and `{AGENT_NAME}_API_KEY` from the environment. You can also set generic `BASE_URL` / `API_KEY` as a fallback.

For Vertex AI backends add:

```dotenv
GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json
VERTEX_PROJECT=my-gcp-project
VERTEX_LOCATION=us-central1   # optional, defaults to us-central1
```

## Running a single Android task

`single_task_run.py` shows the minimal usage pattern:

```python
from age import BrownieAgent

task = "Open NetEase Music, search for a song, open the playback screen and pause"

with BrownieAgent(
    model_name="BrownieGUI-8B",
    task_name=task,
    language="EN",
    enable_tracking=True,
) as agent:
    while True:
        action = agent.predict()
        if action["action_type"] == "finished":
            break
```

```bash
python single_task_run.py
```

Execution traces and annotated screenshots are written to `tracks/` and logs to `logs/`.

## Running the ScaleWoB benchmark

`scalewob-eval-benchmark-parallel.py` evaluates agents against a CSV benchmark in parallel worker processes:

```bash
python scalewob-eval-benchmark-parallel.py \
  --input-csv data/benchmark_source.csv \
  --output-json data/benchmark_eval.json \
  --agent brownie \
  --model-name BrownieGUI-8B \
  --max-steps 40 \
  --workers 4
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--agent` | `doubao` | Agent backend: `doubao`, `brownie`, `mai`, `rgm`, `general`, `open-autoglm`, `mobileagent35` |
| `--model-name` | `doubao-seed-1-8-251228` | Model name passed to the agent |
| `--max-steps` | `40` | Maximum steps per task |
| `--workers` | auto | Number of parallel worker processes |
| `--max-workers-cap` | `4` | Cap for auto worker count |
| `--headless` / `--no-headless` | headless | Browser headless mode |
| `--rerun-failed` / `--no-rerun-failed` | rerun | Re-run previously failed tasks |
| `--no-param-task` | — | Skip tasks with non-null ScaleWoB params |
| `--vertex-api` | — | Use Vertex AI instead of OpenAI-compatible endpoint |

Results are written incrementally to `--output-json` (crash-safe via per-worker shard files).

## Analyzing benchmark results

`scripts/analyze_ckpt_avg_steps.py` computes step statistics for successful tasks:

```bash
python scripts/analyze_ckpt_avg_steps.py \
  --input-csv data/benchmark_source.csv \
  --input-ckpt data/benchmark_eval.json \
  --track-dir tracks
```

Repeat `--track-dir` to aggregate across multiple track roots:

```bash
python scripts/analyze_ckpt_avg_steps.py \
  --input-csv data/benchmark_source.csv \
  --input-ckpt data/benchmark_eval.json \
  --track-dir tracks/worker_1234 \
  --track-dir tracks/worker_5678
```

## Available agents

| Class | Target | Prompt format |
|---|---|---|
| `DoubaoAgent` | Android | `Thought: / Action: func(args)` |
| `MAIAgent` | Android | `<thinking>` + `<tool_call>` JSON |
| `RGMAgent` | Android | System tool schema + `<tool_call>` JSON |
| `BrownieAgent` | Android | `Thought: / Action: \`device.method()\`` |
| `DoubaoScaleWoBAgent` | ScaleWoB | Same as Doubao |
| `MAIScaleWoBAgent` | ScaleWoB | Same as MAI |
| `RGMScaleWoBAgent` | ScaleWoB | Same as RGM |
| `BrownieScaleWoBAgent` | ScaleWoB | Same as Brownie |
| `GeneralScaleWoBAgent` | ScaleWoB | General |
| `OpenAutoGLMScaleWoBAgent` | ScaleWoB | OpenAutoGLM |
| `MobileAgent35ScaleWoBAgent` | ScaleWoB | MobileAgent v3.5 |

All agents and core types are importable directly from `age`:

```python
from age import BrownieAgent, DoubaoAgent, MAIAgent, RGMAgent, AndroidDevice
```

## Development

```bash
# Format code and sort imports
poe format

# Run all checks (format, lint, type check)
poe check

# Auto-fix linting issues
poe fix

# Run tests
python -m pytest tests/ -v
```
