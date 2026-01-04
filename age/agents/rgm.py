import json
import re
import time
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from age.devices import AndroidDevice

from ..api import APIClient
from ..icons import (
    CHECK,
    GEAR,
    MAGIC,
    ROCKET,
    ROTATE_LEFT,
    THOUGHT,
    TIMES,
    TROPHY,
    WARNING,
)
from ..logger import get_logger
from .base import Action, Agent


class RGMAction(TypedDict):
    action: str
    coordinate: NotRequired[list[int]]
    coordinate2: NotRequired[list[int]]
    text: NotRequired[str]
    time: NotRequired[float]
    status: NotRequired[str]


_SYS_PROMPT = """
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait for ssuccessive screenshots to see the results of your actions.\\n* The screen's resolution is 1000x1000.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `key`: Perform a key event on the mobile device.\\n    - This supports adb's `keyevent` syntax.\\n    - Examples: \\"volume_up\\", \\"volume_down\\", \\"home\\", \\"back\\", \\"camera\\", \\"brightness_up\\", \\"brightness_down\\".\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `open`: Open an app on the device.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "open", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=key`, `action=type`, and `action=open`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
"""

_USER_PROMPT = """
# Task
{instruction}

# Action History
{action_history}
"""


class RGMAgent(Agent):
    def __init__(
        self,
        model_name: str,
        task_name: str,
        language: Literal["CN", "EN"] = "CN",
        device_id: str | None = None,
        enable_tracking: bool = False,
        tracking_output_dir: str | Path | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.device = AndroidDevice(device_id)
        # Initialize base class with tracking and step limiting parameters
        super().__init__(
            enable_tracking=enable_tracking,
            tracking_output_dir=tracking_output_dir,
            task_name=task_name,
            model_name=model_name,
            language=language,
            max_steps=max_steps,
        )
        self.name = "RGM"
        self.model_name = model_name
        self.task_name = task_name
        self.window_size = self.device.size
        self.language = language
        self.client = APIClient(prefix=self.name)
        self.history = []
        self.action_history = []
        self.logger = get_logger(__name__)

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)
        self.task_name = kwargs.get("task_name", "")
        self.history = []
        self.action_history = []
        self.logger.info(
            f"[bold magenta]{ROTATE_LEFT} Agent reset[/]",
            extra={"markup": True},
        )

    def _normalize_coordinates(self, x: int, y: int) -> tuple[int, int]:
        width, height = self.window_size
        return int(x * width / 1000), int(y * height / 1000)

    def _format_action_history(self) -> str:
        """Format the action history for the prompt."""
        if not self.action_history:
            return "Empty. You are about to perform the first action."

        formatted = []
        for i, action in enumerate(self.action_history, start=1):
            formatted.append(f"Step {i}. {action}")
        return "\n".join(formatted)

    def _build_messages(self, screenshot_b64: str) -> list[dict]:
        """Build the messages for the LLM request."""
        # Build user prompt
        user_prompt = _USER_PROMPT.format(
            instruction=self.task_name,
            action_history=self._format_action_history(),
        )

        messages = [
            {
                "role": "system",
                "content": _SYS_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        return messages

    def _parse_action(self, response_text: str) -> Action:
        """Parse the LLM response to extract the action."""
        # Try to find JSON object anywhere in the response
        # Look for patterns like {"name": "mobile_use", "arguments": {...}}
        # Use a more robust pattern that can handle nested structures
        json_match = re.search(
            r'\{"name"\s*:\s*"mobile_use"\s*,\s*"arguments"\s*:\s*\{.*?\}\}',
            response_text,
            re.DOTALL,
        )
        if json_match:
            json_str = json_match.group(0)
            if '"action": "type"' in json_str:
                json_str = json_str.replace('")"}}', '"}}')
        else:
            raise ValueError(f"Cannot parse action from response: {response_text}")

        try:
            function_call = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {json_str}") from e

        if function_call.get("name") != "mobile_use":
            raise ValueError(f"Unknown function: {function_call.get('name')}")

        arguments = function_call.get("arguments", {})
        action_type = arguments.get("action") or ""

        # Extract thought from response if present
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\nAction:|\n\n|\Z)", response_text, re.DOTALL
        )
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            thought = f"Performing {action_type} action"

        # Extract action description from response if present
        action_desc_match = re.search(r"Action:\s*(.*)", response_text)
        if action_desc_match:
            action_description = action_desc_match.group(1).strip()
        else:
            action_description = None

        self.logger.info(
            f"\n[bold cyan]{THOUGHT} Thought[/]\n[italic dim]{thought}[/]\n",
            extra={"markup": True},
        )

        # Map RGM action types to standard Action types
        action_mapping = {
            "key": "key",
            "click": "click",
            "long_press": "long_press",
            "swipe": "drag",
            "type": "type",
            "open": "open_app",
            "wait": "wait",
            "terminate": "finished",
        }

        standard_action_type = action_mapping.get(action_type) or action_type

        # Build params dict
        params = {}
        if action_type == "click":
            params["point"] = tuple(arguments.get("coordinate", [0, 0]))
        elif action_type == "long_press":
            params["point"] = tuple(arguments.get("coordinate", [0, 0]))
            params["duration"] = arguments.get("time", 1.0)
        elif action_type == "swipe":
            params["start_point"] = tuple(arguments.get("coordinate", [0, 0]))
            params["end_point"] = tuple(arguments.get("coordinate2", [0, 0]))
        elif action_type == "type":
            params["content"] = arguments.get("text", "")
        elif action_type == "open":
            params["app"] = arguments.get("text", "")
        elif action_type == "key":
            params["key"] = arguments.get("text", "")
        elif action_type == "wait":
            params["seconds"] = int(arguments.get("time", 1))
        elif action_type == "terminate":
            params["content"] = arguments.get("status", "success")

        # Store action description separately (for history tracking)
        if action_description:
            params["description"] = action_description

        return Action(action_type=standard_action_type, params=params, thought=thought)

    def _execute(self, action: Action) -> float:
        """Execute action and return duration.

        Returns:
            Execution time in seconds
        """
        start_time = time.perf_counter()

        action_type = action["action_type"]
        params = action["params"]

        # Color coding for different action types
        action_colors = {
            "click": "green",
            "long_press": "green",
            "type": "blue",
            "drag": "purple",
            "open_app": "cyan",
            "key": "magenta",
            "wait": "dim",
            "finished": "bold green",
        }
        color = action_colors.get(action_type, "white")

        self.logger.info(
            f"[{color}]{GEAR} {action_type}[/]\n",
            extra={"markup": True},
        )
        self.logger.debug(f"Action parameters: {params}")

        if action_type == "click":
            x, y = self._normalize_coordinates(*params["point"])
            self.logger.debug(f"Clicking at: ({x}, {y})")
            self.device.press(x, y)
        elif action_type == "long_press":
            x, y = self._normalize_coordinates(*params["point"])
            duration = params.get("duration", 1.0)
            self.logger.debug(f"Long pressing at: ({x}, {y}) for {duration}s")
            self.device.long_press(x, y, duration=duration)
        elif action_type == "type":
            content = params["content"]
            display_content = content[:50] + "..." if len(content) > 50 else content
            self.logger.debug(f'Typing: "{display_content}"')
            self.device.type_text(content)
        elif action_type == "drag":
            start_point = self._normalize_coordinates(*params["start_point"])
            end_point = self._normalize_coordinates(*params["end_point"])
            self.logger.debug(f"Swiping: {start_point} → {end_point}")
            self.device.drag(
                start_point[0],
                start_point[1],
                end_point[0],
                end_point[1],
                duration=1.0,
            )
        elif action_type == "open_app":
            app = params["app"]
            self.logger.debug(f"Opening app: {app}")
            self.device.open_app(app)
        elif action_type == "key":
            key = params["key"]
            self.logger.debug(f"Pressing key: {key}")
            # Map key names to ADB key events
            key_mapping = {
                "home": "HOME",
                "back": "BACK",
                "volume_up": "VOLUME_UP",
                "volume_down": "VOLUME_DOWN",
                "camera": "CAMERA",
                "brightness_up": "BRIGHTNESS_UP",
                "brightness_down": "BRIGHTNESS_DOWN",
            }
            adb_key = key_mapping.get(key.lower(), key.upper())
            if adb_key:
                self.device._device.keyevent(adb_key)
            else:
                self.logger.warning(f"Unknown key: {key}")
        elif action_type == "wait":
            seconds = params["seconds"]
            self.logger.debug(f"Waiting for {seconds} second(s)...")
            time.sleep(seconds)
            self.logger.debug("Wait complete")
        elif action_type == "finished":
            status = params.get("content", "success")
            self.logger.info(
                f"\n[bold green]{TROPHY} Task {status}[/]\n",
                extra={"markup": True},
            )
        else:
            self.logger.error(
                f"[bold red]{TIMES} Unknown action type: {action_type}[/]",
                extra={"markup": True},
            )
            raise ValueError(f"Unknown action type: {action_type}")

        end_time = time.perf_counter()
        return end_time - start_time

    def predict(self) -> Action:
        # Check step limit (using base class method)
        forced_action = self._check_step_limit()
        if forced_action:
            self.logger.warning(
                f"[bold yellow]{WARNING} Maximum steps reached ({self.max_steps})[/]\n"
                f"[dim]Forcing task completion[/]\n",
                extra={"markup": True},
            )
            duration = self._execute(forced_action)
            if self.tracker:
                screenshot = self.device.capture_screenshot("B64")
                self._record_action(forced_action, screenshot, duration)
                result = self._finalize_tracking()
                if result:
                    trace_path, json_path = result
                    self.logger.debug(f"Generated {trace_path} and {json_path}")
            return forced_action

        if not self.history:
            self.logger.info(
                f"[bold magenta]{ROCKET} Task Started[/]\n"
                f"[italic dim]{self.task_name}[/]\n",
                extra={"markup": True},
            )

        self.logger.debug("Capturing screenshot...")
        screenshot = self.device.capture_screenshot("B64")
        self.logger.debug(f"Screenshot captured: {len(screenshot)} chars (base64)")

        # Store screenshot for tracking
        screenshot_for_tracker = screenshot

        # Build messages
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
        duration = self._execute(action)

        # Add to action history for display
        # Use the human-readable description if available, otherwise use action type and params
        if "description" in action["params"]:
            action_desc = action["params"]["description"]
        else:
            action_desc = f"{action['action_type']}: {action['params']}"
        self.action_history.append(action_desc)

        # Record action for tracking
        if self.tracker:
            self._record_action(action, screenshot_for_tracker, duration)

            # Finalize tracking on finished action
            if action["action_type"] == "finished":
                result = self._finalize_tracking()
                if result:
                    trace_path, json_path = result
                    self.logger.debug(f"Generated {trace_path} and {json_path}")

        # Add to history
        self.history.append({"role": "assistant", "content": response.response})

        return action
