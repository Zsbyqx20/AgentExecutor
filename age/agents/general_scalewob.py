import json
import re
import time
from pathlib import Path
from typing import Any, Literal

from ..api import APIClient
from ..icons import CHECK, MAGIC, ROCKET, THOUGHT, WARNING
from ..logger import get_logger
from .base import Action
from .doubao_scalewob import DoubaoScaleWoBAgent

ACTION_ALIASES = {
    "click": ["tap", "press", "touch"],
    "long_press": ["long tap", "long press", "hold"],
    "input_text": ["type", "enter_text", "write", "enter"],
    "scroll": ["swipe", "fling"],
    "keyboard_enter": ["enter"],
}

NORMALIZED_ACTION_MAP: dict[str, str] = {}
for standard_action, aliases in ACTION_ALIASES.items():
    NORMALIZED_ACTION_MAP[standard_action] = standard_action
    for alias in aliases:
        NORMALIZED_ACTION_MAP[alias.replace(" ", "_")] = standard_action
        NORMALIZED_ACTION_MAP[alias] = standard_action

CLAUDE_IMAGE_SIZE = (1280, 720)

GENERAL_E2E_PROMPT_TEMPLATE = """# Role: Android Phone Operator AI
You are an AI that controls an Android phone to complete user requests. Your responsibilities:
- Answer questions by retrieving information from the phone.
- Perform tasks by executing precise actions.

# Action Framework
Respond with EXACT JSON format for one of these actions:
| Action          | Description                              | JSON Format Example                                                         |
|-----------------|----------------------------------------- |-----------------------------------------------------------------------------|
| `click`         | Tap visible element (describe clearly)   | `{{"action_type": "click", "coordinate": [x, y]}}`   |
| `long_press`    | Long-press visible element (describe clearly) | `{{"action_type": "long_press", "coordinate": [x, y]}}`            |
| `drag`          | Drag from visible element to another visible element (describe both clearly) | `{{"action_type": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]}}`            |
| `input_text`    | Type into field (This action includes clicking the text field, typing, and pressing enter—no need to click the target field first.) | `{{"action_type":"input_text", "text":"Hello"}}|
| `scroll`        | Scroll direction (up/down/left/right)    | `{{"action_type":"scroll", "direction":"down"}}`                            |
| `status`        | Mark task as `complete` or `infeasible`, and provide optional response  | `{{"action_type":"status", "goal_status":"complete", "text": "answer to the user"}}`                      |
| `wait`          | Wait for screen to update                | `{{"action_type":"wait"}}`                                                  |
| `keyboard_enter`   | Press enter key         | `{{"action_type":"keyboard_enter"}}`               |

Note:
- The coordinate is the center of the element to be clicked/long-pressed/dragged.
- x, y are coordinates in the screen, the origin is the top-left corner of the screen.
- x, y are numbers, the range is normalized to [0, {scale_factor}].

# Execution Principles
1. Communication Rule:
   - ALWAYS use 'answer' action to reply to users - never assume on-screen text is sufficient
   - Please follow the user instruction strictly to answer the question, e.g., only return a single number, only return True/False, only return items separated by comma.
   - NEVER use 'answer' action to indicate waiting or loading - use 'wait' action instead
   - Note that `answer` will terminate the task immediately.

2. Efficiency First:
   - Choose simplest path to complete tasks
   - If action fails twice, try alternatives (e.g., long_press instead of click)

3. Smart Navigation:
   - Gather information when needed (e.g., open Calendar to check schedule)
   - For scrolling:
     * Scroll direction is INVERSE to swipe (scroll down to see lower content)
     * If scroll fails, try opposite direction

4. Text Operations:
   - You MUST first click the input box to activate it before typing the text.
   - For text manipulation:
     1. Long-press to select
     2. Use selection bar options (Copy/Paste/Select All)
     3. Delete by selecting then cutting

5. Ask User:
    - If you think you have no enough information to complete the task, you should use `ask_user` action to ask the user to get more information.


# Decision Process
1. Analyze goal, history, and current screen
2. Determine if task is already complete (use `status` if true)
3. If not, choose the most appropriate action to complete the task.
4. Output in exact format below, and ensure the Action is a valid JSON string:
5. The action output format is different for GUI actions and MCP tool actions. Note only one tool call is allowed in one action.

# Expected Output Format (`Thought: ` and `Action: ` are required):
Thought: [Analysis including reference to key steps/points when applicable]
Action: [Single JSON action]

# Output Format Example
## for GUI actions:
Thought: I need to ... to complete the task.
Action: {{"action_type": "type", "text": "What is weather like in San Francisco today?"}}

# User Goal
{goal}
"""


def normalize_action_type(action_type: str) -> str | None:
    if not action_type:
        return None
    processed_type = action_type.lower().strip().replace(" ", "_")
    return NORMALIZED_ACTION_MAP.get(processed_type, action_type)


def _parse_action_text(plan_output: str) -> tuple[str, str]:
    parts = plan_output.split("Action:")
    if len(parts) != 2:
        raise ValueError("Expected exactly one 'Action:' in model output")
    thought_part = parts[0].strip()
    thought = thought_part[8:].strip() if thought_part.startswith("Thought:") else thought_part
    action_text = parts[1].strip()
    return thought, action_text


def _parse_json_action(action_str: str) -> dict[str, Any]:
    cleaned = action_str.strip()
    markdown_match = re.search(r"```(?:json)?(.*?)```", cleaned, re.DOTALL)
    if markdown_match:
        cleaned = markdown_match.group(1).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        begin = cleaned.find("{")
        end = cleaned.rfind("}")
        if begin == -1 or end == -1 or begin >= end:
            raise
        data = json.loads(cleaned[begin : end + 1])
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object action, got: {type(data).__name__}")
    return data


class GeneralScaleWoBAgent(DoubaoScaleWoBAgent):
    """ScaleWoB agent using GeneralE2E prompt/context management."""

    def __init__(
        self,
        model_name: str,
        task_name: str,
        env_id: str,
        task_id: int,
        task_params: dict[str, Any] | None = None,
        language: Literal["CN", "EN"] = "EN",
        headless: bool = False,
        platform: Literal["mobile", "desktop"] = "mobile",
        enable_notepad: bool = False,
        enable_tracking: bool = False,
        tracking_output_dir: str | Path | None = None,
        max_steps: int | None = None,
        history_n_images: int = 3,
        scale_factor: int | tuple[int, int] = 1000,
    ) -> None:
        super().__init__(
            model_name=model_name,
            task_name=task_name,
            env_id=env_id,
            task_id=task_id,
            task_params=task_params,
            language=language,
            headless=headless,
            platform=platform,
            enable_notepad=enable_notepad,
            enable_tracking=enable_tracking,
            tracking_output_dir=tracking_output_dir,
            max_steps=max_steps,
        )
        self.name = "GENERAL"
        self.client = APIClient(prefix=self.name)
        self.logger = get_logger(__name__)
        self.history_n_images = history_n_images
        self.scale_factor = scale_factor
        if "claude" in self.model_name.lower():
            self.scale_factor = CLAUDE_IMAGE_SIZE
        if "k2.5" in self.model_name.lower():
            self.scale_factor = 1
        self.history_images: list[tuple[str, Any, Any]] = []
        self.history_responses: list[dict[str, str]] = []
        self.actions: list[dict[str, Any]] = []

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)
        self.history_images = []
        self.history_responses = []
        self.actions = []

    def _get_user_message(self, screenshot_b64: str, _tool_call: Any, _ask_user: Any) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                }
            ],
        }

    def _hide_history_images(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        num_images_used = 0
        for i in range(len(messages)):
            reverse_i = len(messages) - i - 1
            message = messages[reverse_i]
            content = message.get("content", [])
            if (
                message.get("role") == "user"
                and isinstance(content, list)
                and content
                and content[0].get("type") == "image_url"
            ):
                if num_images_used < self.history_n_images:
                    num_images_used += 1
                else:
                    message["content"] = [
                        {"type": "text", "text": "(Previous turn, screen not shown)"}
                    ]
        return messages

    def _build_messages(self) -> list[dict[str, Any]]:
        assert len(self.history_images) == len(self.history_responses) + 1
        scale_value: int | tuple[int, int] = self.scale_factor
        if isinstance(scale_value, tuple):
            scale_text = f"{scale_value[0]}"
        else:
            scale_text = f"{scale_value}"

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": GENERAL_E2E_PROMPT_TEMPLATE.format(
                    goal=self.task_name,
                    scale_factor=scale_text,
                ),
            },
            self._get_user_message(
                self.history_images[0][0], self.history_images[0][1], self.history_images[0][2]
            ),
        ]

        for idx, history_resp in enumerate(self.history_responses):
            history_img_data, tool_call_res, ask_user_response_res = self.history_images[idx + 1]
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": history_resp.get("content", "")}],
                }
            )
            messages.append(
                self._get_user_message(history_img_data, tool_call_res, ask_user_response_res)
            )
        return self._hide_history_images(messages)

    def _to_norm_1000(self, x: float, y: float) -> tuple[int, int]:
        scale_x, scale_y = (
            (self.scale_factor, self.scale_factor)
            if isinstance(self.scale_factor, int)
            else self.scale_factor
        )
        if scale_x == 1 and scale_y == 1:
            x = x * 1000
            y = y * 1000
            scale_x = 1000
            scale_y = 1000
        norm_x = int(float(x) * 1000 / float(scale_x))
        norm_y = int(float(y) * 1000 / float(scale_y))
        return norm_x, norm_y

    def _convert_to_scale_wob_action(self, thought: str, action_data: dict[str, Any]) -> Action:
        original_action_type = str(action_data.get("action_type", "")).strip()
        action_type = normalize_action_type(original_action_type)
        if not action_type:
            return Action(
                action_type="wait",
                params={"seconds": 1},
                thought=thought or "Action type missing, waiting briefly before retry.",
            )

        if action_type in {"click", "long_press", "double_tap"}:
            coordinate = action_data.get("coordinate", [500, 500])
            if isinstance(coordinate, list) and len(coordinate) == 2:
                x, y = self._to_norm_1000(float(coordinate[0]), float(coordinate[1]))
                mapped_type = "click" if action_type == "double_tap" else action_type
                return Action(action_type=mapped_type, params={"point": (x, y)}, thought=thought)
            return Action(action_type="wait", params={"seconds": 1}, thought=thought)

        if action_type == "drag":
            start_coordinate = action_data.get("start_coordinate", [500, 700])
            end_coordinate = action_data.get("end_coordinate", [500, 300])
            if (
                isinstance(start_coordinate, list)
                and len(start_coordinate) == 2
                and isinstance(end_coordinate, list)
                and len(end_coordinate) == 2
            ):
                start_x, start_y = self._to_norm_1000(
                    float(start_coordinate[0]), float(start_coordinate[1])
                )
                end_x, end_y = self._to_norm_1000(float(end_coordinate[0]), float(end_coordinate[1]))
                return Action(
                    action_type="drag",
                    params={
                        "start_point": (start_x, start_y),
                        "end_point": (end_x, end_y),
                    },
                    thought=thought,
                )
            return Action(action_type="wait", params={"seconds": 1}, thought=thought)

        if action_type == "input_text":
            return Action(
                action_type="type",
                params={"content": str(action_data.get("text", ""))},
                thought=thought,
            )
        if action_type == "scroll":
            direction = str(action_data.get("direction", "down")).lower()
            if direction not in {"up", "down", "left", "right"}:
                direction = "down"
            return Action(
                action_type="scroll",
                params={"point": (500, 500), "direction": direction},
                thought=thought,
            )
        if action_type == "keyboard_enter":
            return Action(action_type="press_enter", params={}, thought=thought)
        if action_type == "status":
            goal_status = str(action_data.get("goal_status", "infeasible")).lower()
            status_text = str(action_data.get("text", "")).strip()
            return Action(
                action_type="finished",
                params={
                    "content": (
                        status_text
                        if status_text
                        else ("task finished" if goal_status == "complete" else "task failed")
                    )
                },
                thought=thought,
            )
        if action_type == "answer":
            # Deprecated in prompt; keep compatibility with older model responses.
            return Action(
                action_type="finished",
                params={"content": str(action_data.get("text", "")).strip() or "task finished"},
                thought=thought,
            )
        if action_type == "wait":
            return Action(action_type="wait", params={"seconds": 1}, thought=thought)

        # Safe fallback for actions unsupported by current ScaleWoB executor.
        return Action(
            action_type="wait",
            params={"seconds": 1},
            thought=thought or f"Unsupported action '{action_type}' in ScaleWoB; waiting briefly.",
        )

    def _parse_action(self, response_text: str) -> Action:
        try:
            thought, action_text = _parse_action_text(response_text)
            if thought:
                self.logger.info(
                    f"\n[bold cyan]{THOUGHT} Thought[/]\n[italic dim]{thought}[/]\n",
                    extra={"markup": True},
                )
            action_data = _parse_json_action(action_text)
            return self._convert_to_scale_wob_action(thought, action_data)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                f"[bold yellow]{WARNING} Failed to parse General action[/]: {exc}",
                extra={"markup": True},
            )
            return Action(
                action_type="wait",
                params={"seconds": 1},
                thought="Parsing failed, waiting briefly before retry.",
            )

    def predict(self) -> Action:
        if not self.history_images:
            self.logger.info(
                f"[bold magenta]{ROCKET} Task Started[/]\n"
                f"[italic dim]{self.task_name}[/]\n",
                extra={"markup": True},
            )

        forced_action = self._check_step_limit()
        if forced_action:
            self.logger.warning(
                f"[bold yellow]{WARNING} Maximum steps reached ({self.max_steps})[/]\n"
                f"[dim]Forcing task completion[/]\n",
                extra={"markup": True},
            )
            duration, status = self.execute(forced_action)
            if self.tracker:
                screenshot = self.device.capture_screenshot("B64")
                self._record_action(forced_action, screenshot, duration)
                result = self._finalize_tracking(result=status)
                if result:
                    trace_path, json_path = result
                    self.logger.debug(f"Generated {trace_path} and {json_path}")
                self.task_status = False
            return forced_action

        self.logger.debug("Capturing screenshot...")
        screenshot = self.device.capture_screenshot("B64")
        screenshot_for_tracker = screenshot
        self.history_images.append((screenshot, None, None))

        messages = self._build_messages()

        self.logger.debug(
            f"[bold blue]{MAGIC} Requesting LLM prediction[/] ({self.model_name})",
            extra={"markup": True},
        )

        start_time = time.time()
        response = self.client.query(
            self.model_name,
            messages,
            logger=self.logger,
        )
        elapsed = time.time() - start_time

        self.logger.info(
            f"[dim]{CHECK} Response received in {elapsed:.2f}s[/]",
            extra={"markup": True},
        )
        self.logger.debug(f"Response length: {len(response.response)} chars")

        action = self._parse_action(response.response)
        duration, status = self.execute(action)

        if self.tracker:
            self._record_action(action, screenshot_for_tracker, duration)
            if action["action_type"] == "finished":
                self.task_status = status
                result = self._finalize_tracking(result=status)
                if result:
                    trace_path, json_path = result
                    self.logger.debug(f"Generated {trace_path} and {json_path}")

        self.history_responses.append({"role": "assistant", "content": response.response})
        self.actions.append(action) # type: ignore
        return action


__all__ = ["GeneralScaleWoBAgent"]
