import json
import re
import time
from pathlib import Path
from typing import Literal

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
from .base import Action, Agent, Language

# Base prompt components
_BASE_PROMPT_PREFIX = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
For each function call, return the thinking process in <thinking> </thinking> tags, and a json object with function name and arguments within <tool_call></tool_call> XML tags:
```
<thinking>
...
</thinking>
<tool_call>
{{"name": "mobile_use", "arguments": <args-json-object>}}
</tool_call>
```

## Action Space
"""

_BASE_ACTIONS = """{{"action": "click", "coordinate": [x, y]}}
{{"action": "long_press", "coordinate": [x, y]}}
{{"action": "type", "text": ""}}
{{"action": "swipe", "direction": "up or down or left or right", "coordinate": [x, y]}}
{{"action": "open", "text": "app_name"}}
{{"action": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]}}
{{"action": "system_button", "button": "button_name"}} # Options: back, home, menu, enter
{{"action": "wait"}}
{{"action": "terminate", "status": "success or fail"}}
{{"action": "answer", "text": "xxx"}} # Use escape characters \\', \\", and \\n in text part
{{"action": "ask_user", "text": "question"}} # Ask the user for input
"""

_FINISHED_ACTION = """
## Note
- Write a small plan and finally summarize your next action (with its target element) in one sentence in <thinking></thinking> part.
- You should use the `open` action to open the app as possible as you can, because it is the fast way to open the app.
- You must follow the Action Space strictly, and return the correct json object within <thinking> </thinking> and <tool_call></tool_call> XML tags.

## User Instruction
Current App: {current_app}
{instruction}
"""


class MAIAgent(Agent):
    """
    Mobile Automation Agent using XML-style prompt format.

    This agent uses the MAI prompt format with <thinking></thinking> and
     XML tags. It supports a configurable sliding window for history
    and includes ask_user action for real-time user input.

    Attributes:
        SCALE_FACTOR: Coordinate normalization factor (999 for MAI)
        history_n: Number of recent screenshots to keep in context
    """

    SCALE_FACTOR = 999

    def __init__(
        self,
        model_name: str,
        task_name: str,
        language: Literal["CN", "EN"] = "CN",
        device_id: str | None = None,
        history_n: int = 3,
        enable_tracking: bool = False,
        tracking_output_dir: str | Path | None = None,
        max_steps: int | None = None,
    ) -> None:
        """
        Initialize the MAIAgent.

        Args:
            model_name: Name of the LLM model to use
            task_name: Task description/instruction
            language: Language for responses ("CN" or "EN")
            device_id: Optional ADB device ID (auto-detect if None)
            history_n: Number of recent screenshots to keep in context (default: 3)
            enable_tracking: Enable action tracking with screenshots
            tracking_output_dir: Custom directory for tracking output
            max_steps: Maximum number of steps before forcing completion
        """
        self.device = AndroidDevice(device_id)
        # Initialize base class with tracking and step limiting parameters
        super().__init__(
            enable_tracking=enable_tracking,
            tracking_output_dir=tracking_output_dir,
            task_name=task_name,
            model_name=model_name,
            language=Language[language.upper()].value,
            max_steps=max_steps,
        )
        self.name = "MAI"
        self.model_name = model_name
        self.task_name = task_name
        self.window_size: tuple[int, int] = self.device.size
        self.language = Language[language.upper()]
        self.client = APIClient(prefix=self.name)
        self.history = []
        self.history_n = history_n
        self.logger = get_logger(__name__)
        self.user_input_buffer = None

    def reset(self, **kwargs) -> None:
        """Reset the agent state for a new task."""
        super().reset(**kwargs)  # Reset step counter
        self.task_name = kwargs.get("task_name", "")
        self.history = []
        self.user_input_buffer = None
        self.logger.info(
            f"[bold magenta]{ROTATE_LEFT} Agent reset[/]",
            extra={"markup": True},
        )

    def _normalize_coordinates(self, x: int, y: int) -> tuple[int, int]:
        """
        Normalize coordinates from SCALE_FACTOR to device pixels.

        MAI uses a 0-999 scale (SCALE_FACTOR), which needs to be converted
        to actual device pixel coordinates.

        Args:
            x: X coordinate in 0-SCALE_FACTOR range
            y: Y coordinate in 0-SCALE_FACTOR range

        Returns:
            Tuple of (x, y) in device pixel coordinates
        """
        width, height = self.window_size
        return int(x * width / self.SCALE_FACTOR), int(y * height / self.SCALE_FACTOR)

    def _build_prompt(self) -> str:
        """
        Build the system prompt with action space and instructions.

        Returns:
            Complete system prompt string
        """
        # Build the prompt
        prompt = _BASE_PROMPT_PREFIX + _BASE_ACTIONS + _FINISHED_ACTION

        # Get current app and add user instruction
        current_app = self.device.get_current_app()

        # Format language and instruction placeholders
        return prompt.format(
            language=self.language.value,
            current_app=current_app,
            instruction=self.task_name,
        )

    def _parse_action(self, response_text: str) -> Action:
        """
        Parse MAI XML-style response into Action.

        Handles the format:
        ```
        <thinking>
        ... reasoning ...
        </thinking>
        {"name": "mobile_use", "arguments": {...}}
        ```

        Args:
            response_text: Raw LLM response text

        Returns:
            Parsed Action dictionary

        Raises:
            ValueError: If response cannot be parsed
        """
        # Handle thinking model output format (uses  instead of </thinking>)
        if "" in response_text and "</thinking>" not in response_text:
            response_text = response_text.replace("", "</thinking>")
            response_text = "<thinking>" + response_text

        # Define regex pattern with non-greedy matching
        pattern = r"<thinking>(.*?)</thinking>.*?<tool_call>(.*?)</tool_call>"

        match = re.search(pattern, response_text, re.DOTALL)

        if not match:
            raise ValueError(f"Cannot parse response: {response_text}")

        thinking = match.group(1).strip().strip('"')
        tool_call_json = match.group(2).strip().strip('"')

        # Log thought if present
        if thinking:
            self.logger.info(
                f"\n[bold cyan]{THOUGHT} Thought[/]\n[italic dim]{thinking}[/]\n",
                extra={"markup": True},
            )

        # Parse tool_call as JSON
        try:
            tool_call = json.loads(tool_call_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tool_call: {e}")

        action_type = tool_call["arguments"]["action"]
        params = {}
        args = tool_call["arguments"]

        self.logger.debug(f"Parsing MAI action: {action_type}")

        # Map MAI actions to internal action types
        if action_type in {"click", "long_press"}:
            params["point"] = args["coordinate"]
        elif action_type == "type":
            params["content"] = args["text"]
        elif action_type == "swipe":
            params["point"] = args.get("coordinate", [500, 500])
            params["direction"] = args["direction"]
        elif action_type == "open":
            params["app"] = args["text"]
        elif action_type == "drag":
            params["start_point"] = args["start_coordinate"]
            params["end_point"] = args["end_coordinate"]
        elif action_type == "system_button":
            button = args["button"]
            if button == "home":
                action_type = "press_home"
            elif button == "back":
                action_type = "press_back"
            elif button == "menu":
                action_type = "press_menu"
            elif button == "enter":
                action_type = "type"
                params["content"] = "\n"
        elif action_type == "wait":
            # No additional params needed
            pass
        elif action_type == "terminate":
            action_type = "finished"
            params["content"] = args.get("status", "success")
        elif action_type == "answer":
            action_type = "finished"
            params["content"] = args["text"]
        elif action_type == "ask_user":
            params["text"] = args["text"]
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        return Action(action_type=action_type, params=params, thought=thinking)

    def _execute(self, action: Action) -> float:
        """
        Execute action and return duration.

        Args:
            action: Action dictionary with action_type and params

        Returns:
            Execution time in seconds

        Raises:
            ValueError: If action type is unknown
        """
        from age.devices.android import Direction

        start_time = time.perf_counter()

        action_type = action["action_type"]
        params = action["params"]

        # Color coding for different action types
        action_colors = {
            "click": "green",
            "long_press": "green",
            "type": "blue",
            "scroll": "yellow",
            "drag": "purple",
            "open_app": "cyan",
            "press_home": "magenta",
            "press_back": "magenta",
            "wait": "dim",
            "finished": "bold green",
            "ask_user": "bold cyan",
        }
        color = action_colors.get(action_type, "white")

        self.logger.info(
            f"[{color}]{GEAR} {action_type}[/]\n",
            extra={"markup": True},
        )
        self.logger.debug(f"Action parameters: {params}")

        if action_type == "click":
            x, y = self._normalize_coordinates(*params["point"])
            self.logger.debug(f"Normalized coordinates: ({x}, {y})")
            self.device.press(x, y)
        elif action_type == "long_press":
            x, y = self._normalize_coordinates(*params["point"])
            self.logger.debug(f"Normalized coordinates: ({x}, {y})")
            self.device.long_press(x, y, duration=1.0)
        elif action_type == "type":
            content = params["content"]
            display_content = content[:50] + "..." if len(content) > 50 else content
            self.logger.debug(f'Typing: "{display_content}"')
            self.device.type_text(params["content"])
        elif action_type == "swipe":
            x, y = self._normalize_coordinates(*params["point"])
            direction = params["direction"]
            distance = self.window_size[1] // 2
            self.logger.debug(f"Scroll direction: {direction}, distance: {distance}")
            if direction == "up":
                direction = "down"
            elif direction == "down":
                direction = "up"
            self.device.scroll(x, y, Direction[direction.upper()], distance)
        elif action_type == "open":
            self.logger.debug(f"Opening app: {params['app']}")
            self.device.open_app(params["app"])
        elif action_type == "drag":
            start_x, start_y = self._normalize_coordinates(*params["start_point"])
            end_x, end_y = self._normalize_coordinates(*params["end_point"])
            self.logger.debug(f"Drag: ({start_x}, {start_y}) → ({end_x}, {end_y})")
            self.device.drag(start_x, start_y, end_x, end_y, duration=1.0)
        elif action_type == "press_home":
            self.device.navigate_home()
        elif action_type == "press_back":
            self.device.navigate_back()
        elif action_type == "wait":
            # MAI doesn't specify wait duration, use default
            self.logger.debug("Waiting for 1 second...")
            time.sleep(1)
            self.logger.debug("Wait complete")
        elif action_type == "finished":
            self.logger.info(
                "\n[bold green]{} Task Completed[/]\n[italic dim]{}[/]\n".format(
                    TROPHY, params["content"]
                ),
                extra={"markup": True},
            )
        elif action_type == "ask_user":
            question = params["text"]
            self.logger.info(
                f"[bold cyan]❓ User Question:[/] [italic]{question}[/]",
                extra={"markup": True},
            )
            # Prompt for real user input
            user_input = input(f"\n{question}\nYour response: ")
            self.logger.debug(f"User input received: {user_input[:50]}...")
            # Store user input for next LLM call
            self.user_input_buffer = user_input
        else:
            self.logger.error(
                f"[bold red]{TIMES} Unknown action type: {action_type}[/]",
                extra={"markup": True},
            )
            raise ValueError(f"Unknown action type: {action_type}")

        end_time = time.perf_counter()
        return end_time - start_time

    def _compress_history(self):
        """
        Compress history keeping only the last history_n screenshots.

        Implements a sliding window over the screenshot history to manage
        token usage while maintaining recent context.

        Returns:
            Compressed history list with only recent screenshots
        """
        if not self.history:
            return []
        history = []
        cnt = 0
        screens_num = len(self.history) // 2
        screen_start_idx = max(0, screens_num - self.history_n)

        self.logger.debug(
            f"Compressing history: {len(self.history)} items, "
            f"keeping last {self.history_n} screenshots"
        )

        for h in self.history:
            if h["role"] == "user":
                hc = []
                for c in h["content"]:
                    if c["type"] == "image_url":
                        if cnt < screen_start_idx:
                            continue
                        else:
                            cnt += 1
                    hc.append(c)
                history.append({"role": "user", "content": hc})
            else:
                history.append(h)

        self.logger.debug(f"Compressed to {len(history)} items")
        return history

    def predict(self) -> Action:
        """
        Predict and execute the next action.

        This is the main prediction loop that:
        1. Checks step limits
        2. Captures screenshot
        3. Builds prompt/messages
        4. Queries LLM
        5. Parses response
        6. Executes action
        7. Records tracking if enabled

        Returns:
            Executed Action dictionary
        """
        if not self.history:
            prompt = self._build_prompt()
            self.logger.info(
                f"[bold magenta]{ROCKET} Task Started[/]\n"
                f"[italic dim]{self.task_name}[/]\n",
                extra={"markup": True},
            )
        else:
            prompt = "Please continue"

        # Check step limit (using base class method)
        forced_action = self._check_step_limit()
        if forced_action:
            self.logger.warning(
                f"[bold yellow]{WARNING} Maximum steps reached ({self.max_steps})[/]\n"
                f"[dim]Forcing task completion[/]\n",
                extra={"markup": True},
            )
            # Execute the forced finished action (for tracking and logging)
            duration = self._execute(forced_action)
            # Record action for tracking if enabled
            if self.tracker:
                screenshot = self.device.capture_screenshot("B64")
                self._record_action(forced_action, screenshot, duration)
                # Finalize tracking
                result = self._finalize_tracking()
                if result:
                    trace_path, json_path = result
                    self.logger.debug(f"Generated {trace_path} and {json_path}")
            return forced_action

        self.logger.debug("Capturing screenshot...")
        screenshot = self.device.capture_screenshot("B64")
        self.logger.debug(f"Screenshot captured: {len(screenshot)} chars (base64)")

        # Store screenshot for tracking (reused from LLM capture)
        screenshot_for_tracker = screenshot

        # Build messages
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot}"},
            },
        ]

        # Add user input buffer if present
        if self.user_input_buffer:
            content.append({
                "type": "text",
                "text": f"\n## User Response\n{self.user_input_buffer}",
            })
            self.user_input_buffer = None  # Clear after use

        messages = [{"role": "user", "content": content}]

        self.logger.debug(
            f"[bold blue]{MAGIC} Requesting LLM prediction[/] ({self.model_name})",
            extra={"markup": True},
        )

        start_time = time.time()
        response = self.client.query(
            self.model_name,
            self._compress_history() + messages,
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

        # Record action for tracking
        if self.tracker:
            self._record_action(action, screenshot_for_tracker, duration)

            # Finalize tracking on finished action
            if action["action_type"] == "finished":
                result = self._finalize_tracking()
                if result:
                    trace_path, json_path = result
                    self.logger.debug(f"Generated {trace_path} and {json_path}")

        self.history.extend(messages)
        self.history.append({"role": "assistant", "content": response.response})

        return action


__all__ = ["MAIAgent"]
