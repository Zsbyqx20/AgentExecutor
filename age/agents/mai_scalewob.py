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

MAI_MOBILE_SYS_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
For each function call, return the thinking process in <thinking> </thinking> tags, and a json object with function name and arguments within <tool_call></tool_call> XML tags:
```
<thinking>
...
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": <args-json-object>}
</tool_call>
```

## Action Space

{"action": "click", "coordinate": [x, y]}
{"action": "long_press", "coordinate": [x, y]}
{"action": "type", "text": ""}
{"action": "swipe", "direction": "up or down or left or right", "coordinate": [x, y]} # "coordinate" is optional. Use the "coordinate" if you want to swipe a specific UI element.
{"action": "open", "text": "app_name"}
{"action": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]}
{"action": "system_button", "button": "button_name"} # Options: enter, clear
{"action": "wait"}
{"action": "terminate", "status": "success or fail"}
{"action": "answer", "text": "xxx"} # Use escape characters \\', \\", and \\n in text part to ensure we can parse the text in normal python string format.


## Note
- Write a small plan and finally summarize your next action (with its target element) in one sentence in <thinking></thinking> part.
- You must follow the Action Space strictly, and return the correct json object within <thinking> </thinking> and <tool_call></tool_call> XML tags.
""".strip()


class MAIScaleWoBAgent(DoubaoScaleWoBAgent):
    """ScaleWoB MAI agent with official MAI mobile prompt and context packaging."""

    SCALE_FACTOR = 999

    def __init__(
        self,
        model_name: str,
        task_name: str,
        env_id: str,
        task_id: int,
        task_params: dict[str, Any] | None = None,
        language: Literal["CN", "EN"] = "EN",
        headless: bool = False,
        platform: Literal["mobile", "desktop"] = "desktop",
        enable_notepad: bool = False,
        enable_tracking: bool = False,
        tracking_output_dir: str | Path | None = None,
        max_steps: int | None = None,
        history_n: int = 3,
        use_vertex_api: bool = False,
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
            use_vertex_api=use_vertex_api,
        )
        self.name = "MAI"
        self.client = APIClient(prefix=self.name, use_vertex_api=use_vertex_api)
        self.history_n = history_n
        self.history_steps: list[dict[str, str]] = []
        self.logger = get_logger(__name__)

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)
        self.history_steps = []

    def _build_prompt(self) -> str:
        prompt = MAI_MOBILE_SYS_PROMPT
        if self.task_params:
            prompt += (
                "\n\n## Answer Format\n"
                "### Hint: This task requires JSON output in the final answer. "
                "When using `answer`, make sure `text` is valid JSON matching this schema:\n"
                f"{self.task_params}"
            )
        return prompt

    def _build_messages(self, current_screenshot_b64: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self._build_prompt(),
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": self.task_name}],
            },
        ]

        if not self.history_steps:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{current_screenshot_b64}"
                            },
                        }
                    ],
                }
            )
            return messages

        start_image_idx = max(0, len(self.history_steps) - (self.history_n - 1))

        for idx, step in enumerate(self.history_steps):
            if idx >= start_image_idx:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{step['image_b64']}"
                                },
                            }
                        ],
                    }
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": step["output"]}],
                }
            )

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{current_screenshot_b64}"
                        },
                    }
                ],
            }
        )
        return messages

    def _extract_tagged_payload(self, response_text: str) -> tuple[str, dict[str, Any]]:
        text = response_text.strip()
        if "</think>" in text and "</thinking>" not in text:
            text = text.replace("</think>", "</thinking>")
            text = "<thinking>" + text

        pattern = r"<thinking>(.*?)</thinking>.*?<tool_call>(.*?)</tool_call>"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            raise ValueError("Missing <thinking>/<tool_call> block")

        thought = match.group(1).strip().strip('"')
        tool_call_raw = match.group(2).strip().strip('"')
        tool_call = json.loads(tool_call_raw)
        return thought, tool_call

    def _coerce_point(self, value: Any) -> tuple[int, int]:
        if not isinstance(value, list):
            raise TypeError(f"Invalid coordinate payload: {value}")
        if len(value) == 2:
            x, y = value
        elif len(value) == 4:
            x1, y1, x2, y2 = value
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
        else:
            raise ValueError(f"Invalid coordinate length: {len(value)}")

        return int(x), int(y)

    def _parse_action(self, response_text: str) -> Action:
        try:
            thought, tool_call = self._extract_tagged_payload(response_text)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                f"[bold yellow]{WARNING} Failed to parse MAI action[/]: {exc}",
                extra={"markup": True},
            )
            return Action(
                action_type="wait",
                params={"seconds": 1},
                thought="Parsing failed, waiting briefly before retry.",
            )

        if thought:
            self.logger.info(
                f"\n[bold cyan]{THOUGHT} Thought[/]\n[italic dim]{thought}[/]\n",
                extra={"markup": True},
            )

        args = tool_call.get("arguments", {})
        action_name = args.get("action", "")

        try:
            if action_name == "click":
                return Action(
                    action_type="click",
                    params={"point": self._coerce_point(args["coordinate"])},
                    thought=thought,
                )
            if action_name == "long_press":
                return Action(
                    action_type="long_press",
                    params={"point": self._coerce_point(args["coordinate"])},
                    thought=thought,
                )
            if action_name == "type":
                return Action(
                    action_type="type",
                    params={"content": str(args.get("text", ""))},
                    thought=thought,
                )
            if action_name == "swipe":
                point = self._coerce_point(args["coordinate"]) if "coordinate" in args else (500, 500)
                direction = "down" if str(args.get("direction", "down")) == "up" else "up"
                return Action(
                    action_type="scroll",
                    params={"point": point, "direction": direction},
                    thought=thought,
                )
            if action_name == "open":
                return Action(
                    action_type="wait",
                    params={"seconds": 1},
                    thought=(
                        thought
                        or "ScaleWoB has no app launcher semantics; waiting before next UI step."
                    ),
                )
            if action_name == "drag":
                return Action(
                    action_type="drag",
                    params={
                        "start_point": self._coerce_point(args["start_coordinate"]),
                        "end_point": self._coerce_point(args["end_coordinate"]),
                    },
                    thought=thought,
                )
            if action_name == "system_button":
                button = str(args.get("button", "")).lower()
                if button == "enter":
                    return Action(action_type="press_enter", params={}, thought=thought)
                return Action(
                    action_type="wait",
                    params={"seconds": 1},
                    thought=(
                        thought
                        or f"system_button={button} is not supported in ScaleWoB; waiting briefly."
                    ),
                )
            if action_name == "wait":
                return Action(action_type="wait", params={"seconds": 1}, thought=thought)
            if action_name == "terminate":
                return Action(
                    action_type="finished",
                    params={"content": str(args.get("status", "success"))},
                    thought=thought,
                )
            if action_name == "answer":
                return Action(
                    action_type="finished",
                    params={"content": str(args.get("text", ""))},
                    thought=thought,
                )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                f"[bold yellow]{WARNING} Invalid MAI action payload[/]: {exc}",
                extra={"markup": True},
            )
            return Action(
                action_type="wait",
                params={"seconds": 1},
                thought=thought or "Action payload invalid, waiting briefly before retry.",
            )

        self.logger.warning(
            f"[bold yellow]{WARNING} Unknown MAI action[/]: {action_name}",
            extra={"markup": True},
        )
        return Action(
            action_type="wait",
            params={"seconds": 1},
            thought=thought or "Unknown action, waiting briefly before retry.",
        )

    def predict(self) -> Action:
        if not self.history_steps:
            self.logger.info(
                f"[bold magenta]{ROCKET} Task Started[/]\n"
                f"[italic dim]{self.task_name}[/]\n",
                extra={"markup": True},
            )

        forced_action = self._check_step_limit()
        if forced_action:
            self.logger.warning(
                f"[bold yellow]{WARNING} Maximum steps reached ({self.max_steps})[/]\n"
                "[dim]Forcing task completion[/]\n",
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

        messages = self._build_messages(screenshot)

        self.logger.debug(
            f"[bold blue]{MAGIC} Requesting LLM prediction[/] ({self.model_name})",
            extra={"markup": True},
        )

        start_time = time.time()
        response = self.client.query(
            self.model_name,
            messages,
            logger=self.logger,
            thinking={"type": "disabled"},
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

        self.history_steps.append({"image_b64": screenshot, "output": response.response})
        return action


__all__ = ["MAIScaleWoBAgent"]
