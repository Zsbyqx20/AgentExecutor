import ast
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ..api import APIClient
from ..icons import CHECK, MAGIC, ROCKET, THOUGHT, WARNING
from ..logger import get_logger
from .base import Action
from .doubao_scalewob import DoubaoScaleWoBAgent

SYSTEM_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is 1000x1000.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: \\"volume_up\\", \\"volume_down\\", \\"power\\", \\"camera\\", \\"clear\\".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `answer`: Terminate the current task and output the answer.
* `interact`: Resolve the blocking window by interacting with the user.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "answer", "interact", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=key`, `action=type`, `action=open`, `action=answer`,and `action=interact`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one for Action.
- Do not output anything else outside those two parts.
- If finishing, use action=terminate in the tool call."""


class MobileAgent35ScaleWoBAgent(DoubaoScaleWoBAgent):
    """ScaleWoB agent following the MobileAgent-v3.5 prompting pattern."""

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
        history_n: int = 4,
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
        self.name = "MOBILE35"
        self.client = APIClient(prefix=self.name)
        self.history: list[dict[str, str]] = []
        self.history_n = history_n
        self.logger = get_logger(__name__)

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)
        self.history = []

    def _extract_previous_action_summary(self, text: str) -> str:
        if "Action:" in text and "<tool_call>" in text:
            return text.split("Action:", 1)[1].split("<tool_call>", 1)[0].strip()
        if "<tool_call>" in text:
            return text.split("<tool_call>", 1)[0].strip()
        return text.strip()

    def _build_messages(self, current_screenshot_b64: str) -> list[dict[str, Any]]:
        current_step = len(self.history)
        history_start_idx = max(0, current_step - self.history_n)

        previous_actions: list[str] = []
        for i in range(history_start_idx):
            action_summary = self._extract_previous_action_summary(self.history[i]["output"])
            previous_actions.append(f"Step {i + 1}: {action_summary}")
        previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

        today = datetime.today()  # noqa: DTZ002
        weekday_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        formatted_date = today.strftime("%Y-%m-%d") + " " + weekday_names[today.weekday()]
        instruction_prompt = (
            "Please generate the next move according to the UI screenshot, "
            "instruction and previous actions.\n\n"
            f"Instruction: Today's date is: {formatted_date}.{self.task_name}\n\n"
            f"Previous actions:\n{previous_actions_str}"
        )
        if self.task_params:
            instruction_prompt += (
                "\n\n## Answer Format\n"
                "### Hint: This task requires JSON output in the final answer. "
                "When using `action=answer`, make sure `text` is valid JSON "
                "matching this schema:\n"
                f"{self.task_params}"
            )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        history_len = min(self.history_n, len(self.history))
        if history_len > 0:
            for idx, item in enumerate(self.history[-self.history_n :]):
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{item['image_b64']}"},
                }
                if idx == 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": instruction_prompt},
                                image_content,
                            ],
                        }
                    )
                else:
                    messages.append({"role": "user", "content": [image_content]})
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": item["output"]}],
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
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{current_screenshot_b64}"
                            },
                        },
                    ],
                }
            )

        return messages

    def _extract_tool_call_payload(self, response_text: str) -> dict[str, Any]:
        tool_call_match = re.search(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response_text, re.DOTALL
        )
        payload = ""
        if tool_call_match:
            payload = tool_call_match.group(1).strip()
        else:
            json_match = re.search(
                r'(\{"name"\s*:\s*"mobile_use"\s*,\s*"arguments"\s*:\s*\{.*\}\s*\})',
                response_text,
                re.DOTALL,
            )
            if not json_match:
                raise ValueError("Missing <tool_call> payload")
            payload = json_match.group(1).strip()

        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return ast.literal_eval(payload)

    def _extract_thought(self, response_text: str) -> str:
        action_match = re.search(r"Action:\s*(.*?)(?=<tool_call>|\Z)", response_text, re.DOTALL)
        if action_match:
            return action_match.group(1).strip()
        return "Following the planned next UI action."

    def _parse_action(self, response_text: str) -> Action:
        thought = self._extract_thought(response_text)
        if thought:
            self.logger.info(
                f"\n[bold cyan]{THOUGHT} Thought[/]\n[italic dim]{thought}[/]\n",
                extra={"markup": True},
            )

        try:
            tool_call = self._extract_tool_call_payload(response_text)
        except Exception as exc:
            self.logger.warning(
                f"[bold yellow]{WARNING} Failed to parse tool call[/]: {exc}",
                extra={"markup": True},
            )
            return Action(
                action_type="wait",
                params={"seconds": 1},
                thought=thought or "Parsing failed, waiting briefly before retry.",
            )

        arguments = tool_call.get("arguments", {})
        action_type = arguments.get("action", "")

        if action_type == "click":
            return Action(
                action_type="click",
                params={"point": tuple(arguments.get("coordinate", [500, 500]))},
                thought=thought,
            )
        if action_type == "long_press":
            return Action(
                action_type="long_press",
                params={"point": tuple(arguments.get("coordinate", [500, 500]))},
                thought=thought,
            )
        if action_type == "type":
            return Action(
                action_type="type",
                params={"content": str(arguments.get("text", ""))},
                thought=thought,
            )
        if action_type == "key":
            key_name = str(arguments.get("text", "")).strip().lower()
            if key_name in {"enter"}:
                return Action(action_type="press_enter", params={}, thought=thought)
            if key_name in {"clear"}:
                return Action(action_type="clear_text", params={}, thought=thought)
            self.logger.warning(
                f"[bold yellow]{WARNING} Unsupported key '{key_name}'[/], fallback to wait.",
                extra={"markup": True},
            )
            return Action(action_type="wait", params={"seconds": 1}, thought=thought)
        if action_type in {"swipe", "scroll"}:
            if "coordinate2" in arguments:
                return Action(
                    action_type="drag",
                    params={
                        "start_point": tuple(arguments.get("coordinate", [500, 700])),
                        "end_point": tuple(arguments.get("coordinate2", [500, 300])),
                    },
                    thought=thought,
                )
            direction = str(arguments.get("direction", "down"))
            return Action(
                action_type="scroll",
                params={"point": tuple(arguments.get("coordinate", [500, 500])), "direction": direction},
                thought=thought,
            )
        if action_type == "system_button":
            button = str(arguments.get("button", "")).lower()
            if button == "enter":
                return Action(action_type="press_enter", params={}, thought=thought)
            # ScaleWoB action executor does not support home/back/menu directly.
            if button in {"home", "back", "menu"}:
                self.logger.warning(
                    f"[bold yellow]{WARNING} system_button '{button}' unsupported on ScaleWoB[/], fallback to wait.",
                    extra={"markup": True},
                )
                return Action(action_type="wait", params={"seconds": 1}, thought=thought)
            self.logger.warning(
                f"[bold yellow]{WARNING} Unsupported system_button '{button}'[/], fallback to wait.",
                extra={"markup": True},
            )
            return Action(action_type="wait", params={"seconds": 1}, thought=thought)
        if action_type in {"open", "interact"}:
            self.logger.warning(
                f"[bold yellow]{WARNING} Unsupported action '{action_type}' on ScaleWoB[/], fallback to wait.",
                extra={"markup": True},
            )
            return Action(action_type="wait", params={"seconds": 1}, thought=thought)
        if action_type == "wait":
            seconds = max(1, int(float(arguments.get("time", 1))))
            return Action(action_type="wait", params={"seconds": seconds}, thought=thought)
        if action_type == "answer":
            return Action(
                action_type="finished",
                params={"content": str(arguments.get("text", ""))},
                thought=thought,
            )
        if action_type == "terminate":
            return Action(
                action_type="finished",
                params={"content": str(arguments.get("status", "success"))},
                thought=thought,
            )

        self.logger.warning(
            f"[bold yellow]{WARNING} Unsupported action '{action_type}'[/], fallback to wait.",
            extra={"markup": True},
        )
        return Action(action_type="wait", params={"seconds": 1}, thought=thought)

    def predict(self) -> Action:
        if not self.history:
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
        self.logger.debug(f"Screenshot captured: {len(screenshot)} chars (base64)")
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
        )
        elapsed = time.time() - start_time
        self.logger.info(
            f"[dim]{CHECK} Response received in {elapsed:.2f}s[/]",
            extra={"markup": True},
        )

        action = self._parse_action(response.response)
        duration, status = self.execute(action)

        if self.tracker:
            self._record_action(action, screenshot, duration)
            if action["action_type"] == "finished":
                self.task_status = status
                result = self._finalize_tracking(result=status)
                if result:
                    trace_path, json_path = result
                    self.logger.debug(f"Generated {trace_path} and {json_path}")

        self.history.append({"output": response.response, "image_b64": screenshot})
        return action


__all__ = ["MobileAgent35ScaleWoBAgent"]
