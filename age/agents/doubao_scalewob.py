import ast
import re
import time
from pathlib import Path
from typing import Literal, Any

from age.devices import ScaleWoBDevice

from ..api import APIClient
from ..icons import (
    CHECK,
    ERASER,
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
```
Thought: ...
Action: ...
```

## Action Space
"""

_BASE_ACTIONS = """click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') # If you want to submit your input, use "\\n" at the end of `content`.
clear_text() # Clear all text in the currently activated textbox
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
press_enter() # If you need to confirm the search
wait(seconds=1) # Wait for 1 second before next action (useful for loading screens)
"""

_NOTE_ACTIONS = """note_create(content='xxx') # Create a new note on notepad with the specified content
note_edit(index=1, content='xxx') # Edit the note at position 1 (1-based index)
note_remove(index=1) # Remove the note at position 1 (1-based index)
"""

_FINISHED_ACTION = """finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence (no more than 50 words) in `Thought` part.
- You are operating on a simulator which will not trigger actual payments. Please always finish the payment process unless the task disallows you to do so.
- Always login the app without explicit prohibition. Login with random email or phone number will be fine. The verification code and password can be random without explicit statements.
"""

_NOTEPAD_SECTION = """
## Notepad
### Hint: You can record important information here. It should be useful when you need to work on long-term tasks. Use related actions to manage notes yourself.
{notepad_section}
"""

_USER_INSTRUCTION = """
## User Instruction
{instruction}
"""


class DoubaoScaleWoBAgent(Agent):
    def __init__(
        self,
        model_name: str,
        task_name: str,
        env_id: str,  # Required for ScaleWoB
        task_id: int, 
        task_params: dict[str, Any] | None = None,
        language: Literal["CN", "EN"] = "CN",
        headless: bool = False,
        platform: Literal["mobile", "desktop"] = "desktop",
        enable_notepad: bool = False,
        enable_tracking: bool = False,
        tracking_output_dir: str | Path | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.device = ScaleWoBDevice(env_id, headless, platform)
        # Initialize base class with ScaleWoB device
        super().__init__(
            enable_tracking=enable_tracking,
            tracking_output_dir=tracking_output_dir,
            task_name=task_name,
            model_name=model_name,
            language=Language[language.upper()].value,
            max_steps=max_steps,
            env_id=f"Env: {env_id}. Task: {task_id}",
            description=task_name,
        )
        self.name = "Doubao"
        self.model_name = model_name
        self.task_name = task_name
        self.task_id = task_id
        self.task_params = task_params
        self.window_size = self.device.size
        self.language = Language[language.upper()]
        self.client = APIClient(prefix=self.name)
        self.history = []
        self.logger = get_logger(__name__)
        self.enable_notepad = enable_notepad
        self.notes: list[str] = []
        self.task_status: bool = False

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)  # Reset step counter
        self.task_name = kwargs.get("task_name", "")
        self.history = []
        self.notes = []
        self.logger.info(
            f"[bold magenta]{ROTATE_LEFT} Agent reset[/]",
            extra={"markup": True},
        )

    def _normalize_coordinates(self, x: int, y: int) -> tuple[int, int]:
        width, height = self.window_size
        return int(x * width / 1000), int(y * height / 1000)

    def _format_notepad(self) -> str:
        """Format the notepad section for the prompt."""
        if not self.enable_notepad:
            return ""
        if not self.notes:
            return "No notes yet."
        formatted_notes = []
        for i, note in enumerate(self.notes, start=1):
            formatted_notes.append(f"{i}. {note}")
        return "\n".join(formatted_notes)

    def _build_prompt(self) -> str:
        """Build the prompt dynamically based on notepad configuration."""
        # Build action space
        action_space = _BASE_ACTIONS
        if self.enable_notepad:
            action_space += _NOTE_ACTIONS
        action_space += _FINISHED_ACTION

        # Build the prompt
        prompt = _BASE_PROMPT_PREFIX + action_space

        # Add notepad section if enabled
        if self.enable_notepad:
            notepad_section = self._format_notepad()
            prompt += _NOTEPAD_SECTION.format(notepad_section=notepad_section)

        # Add user instruction (no current_app for ScaleWoB)
        prompt += _USER_INSTRUCTION.format(instruction=self.task_name)

        # Format language placeholder
        res = prompt.format(language=self.language.value)

        # Add json schema if needed
        if self.task_params:
            res += f"""
## Answer Format
### Hint: This task requires you to provide your answer with JSON content. The JSON schema required is as follows.
Please make sure you provide correct JSON format when calling `finished(content={...})`.
{self.task_params}
"""
        return res

    def _parse_action(self, response_text: str) -> Action:
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\nAction:|\n\n|\Z)", response_text, re.DOTALL
        )
        action_match = re.search(
            r"Action:\s*(.*?)(?=\n\n|\Z)", response_text, re.DOTALL
        )

        thought = thought_match.group(1).strip() if thought_match else ""
        action_text = action_match.group(1).strip() if action_match else ""

        # Log thought if present
        if thought:
            self.logger.info(
                "\n[bold cyan]{} Thought[/]\n[italic dim]{}[/]\n".format(
                    THOUGHT, thought
                ),
                extra={"markup": True},
            )

        self.logger.debug(
            f"Parsing action from response: {action_text[:100] if action_text else '(empty)'}..."
        )

        action_type = ""
        params = {}

        if action_text:
            if action_text.startswith("click("):
                action_type = "click"
                point_match = re.search(r"<point>(\d+)\s+(\d+)</point>", action_text)
                if point_match:
                    params["point"] = (
                        int(point_match.group(1)),
                        int(point_match.group(2)),
                    )
            elif action_text.startswith("long_press("):
                action_type = "long_press"
                point_match = re.search(r"<point>(\d+)\s+(\d+)</point>", action_text)
                if point_match:
                    params["point"] = (
                        int(point_match.group(1)),
                        int(point_match.group(2)),
                    )
            elif action_text.startswith("type("):
                action_type = "type"
                content_match = re.search(r"content='([^']*)'", action_text)
                if content_match:
                    params["content"] = content_match.group(1)
            elif action_text.startswith("scroll("):
                action_type = "scroll"
                point_match = re.search(r"<point>(\d+)\s+(\d+)</point>", action_text)
                direction_match = re.search(r"direction='([^']*)'", action_text)
                if point_match:
                    params["point"] = (
                        int(point_match.group(1)),
                        int(point_match.group(2)),
                    )
                if direction_match:
                    params["direction"] = direction_match.group(1)
            elif action_text.startswith("drag("):
                action_type = "drag"
                start_match = re.search(
                    r"start_point='<point>(\d+)\s+(\d+)</point>", action_text
                )
                end_match = re.search(
                    r"end_point='<point>(\d+)\s+(\d+)</point>", action_text
                )
                if start_match:
                    params["start_point"] = (
                        int(start_match.group(1)),
                        int(start_match.group(2)),
                    )
                if end_match:
                    params["end_point"] = (
                        int(end_match.group(1)),
                        int(end_match.group(2)),
                    )
            elif action_text == "press_enter()":
                action_type = "press_enter"
            elif action_text == "clear_text()":
                action_type = "clear_text"
            elif action_text.startswith("wait("):
                action_type = "wait"
                seconds_match = re.search(r"seconds=(\d+)", action_text)
                if seconds_match:
                    params["seconds"] = int(seconds_match.group(1))
            elif action_text.startswith("finished("):
                action_type = "finished"
                content_match = re.search(r"content='([^']*)'", action_text)
                if content_match:
                    params["content"] = content_match.group(1)
            elif action_text.startswith("note_create("):
                action_type = "note_create"
                content_match = re.search(r"content='([^']*)'", action_text)
                if content_match:
                    params["content"] = content_match.group(1)
            elif action_text.startswith("note_edit("):
                action_type = "note_edit"
                index_match = re.search(r"index=(\d+)", action_text)
                content_match = re.search(r"content='([^']*)'", action_text)
                if index_match:
                    params["index"] = int(index_match.group(1))
                if content_match:
                    params["content"] = content_match.group(1)
            elif action_text.startswith("note_remove("):
                action_type = "note_remove"
                index_match = re.search(r"index=(\d+)", action_text)
                if index_match:
                    params["index"] = int(index_match.group(1))
        else:
            raise ValueError(f"Cannot parse the action: {response_text}")

        return Action(action_type=action_type, params=params, thought=thought)

    def _execute(self, action: Action) -> tuple[float, bool]:
        """Execute action and return duration.

        Returns:
            Execution time in seconds
        """
        from ..devices.base import Direction

        start_time = time.perf_counter()

        action_type = action["action_type"]
        params = action["params"]

        # Color coding for different action types
        action_colors = {
            "click": "green",
            "long_press": "green",
            "type": "blue",
            "clear_text": "bold red",
            "scroll": "yellow",
            "drag": "purple",
            "press_enter": "magenta",
            "wait": "dim",
            "finished": "bold green",
            "note_create": "bold cyan",
            "note_edit": "bold yellow",
            "note_remove": "bold red",
        }
        color = action_colors.get(action_type, "white")
        status = True

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
        elif action_type == "scroll":
            x, y = self._normalize_coordinates(*params["point"])
            direction = params["direction"]
            width, height = self.window_size
            distance = height // 2
            self.logger.debug(f"Scroll direction: {direction}, distance: {distance}")
            self.device.scroll(x, y, Direction[direction.upper()], distance)
        elif action_type == "drag":
            start_x, start_y = self._normalize_coordinates(*params["start_point"])
            end_x, end_y = self._normalize_coordinates(*params["end_point"])
            self.logger.debug(f"Drag: ({start_x}, {start_y}) → ({end_x}, {end_y})")
            self.device.drag(start_x, start_y, end_x, end_y)  # type: ignore[arg-type]
        elif action_type == "press_enter":
            self.device.press_enter()
        elif action_type == "clear_text":
            self.logger.info(
                f"[bold red]{ERASER} Clearing text input[/]",
                extra={"markup": True},
            )
            self.device.clear_text()
        elif action_type == "wait":
            seconds = params["seconds"]
            self.logger.debug(f"Waiting for {seconds} second(s)...")
            time.sleep(seconds)
            self.logger.debug("Wait complete")
        elif action_type == "finished":
            if self.task_params:
                try:
                    result = self.device.eval(self.task_id, ast.literal_eval(params['content']))
                except Exception as e:
                    self.logger.error(f"Error when evaluation: {e}")
                    result = False
            else:
                result = self.device.eval(self.task_id, params=None)
            status = result
            self.logger.info(
                "\n[bold green]{} Task Completed. Success: {}[/]\n[italic dim]{}[/]\n".format(
                    TROPHY, status, params["content"]
                ),
                extra={"markup": True},
            )
        elif action_type == "note_create":
            content = params["content"]
            self.notes.append(content)
            display_content = content[:50] + "..." if len(content) > 50 else content
            self.logger.info(
                f'[bold cyan] Note created:[/] [italic]"{display_content}"[/] '
                f"[dim]({len(self.notes)} total notes)[/]",
                extra={"markup": True},
            )
        elif action_type == "note_edit":
            index = params["index"]
            content = params["content"]
            if index < 1 or index > len(self.notes):
                self.logger.error(
                    f"[bold red]{TIMES} Invalid note index: {index}[/] "
                    f"[dim](valid range: 1-{len(self.notes)})[/]",
                    extra={"markup": True},
                )
                raise ValueError(
                    f"Invalid note index: {index}. Valid range: 1-{len(self.notes)}"
                )
            old_content = self.notes[index - 1]
            self.notes[index - 1] = content
            display_old = (
                old_content[:30] + "..." if len(old_content) > 30 else old_content
            )
            display_new = content[:30] + "..." if len(content) > 30 else content
            self.logger.info(
                f"[bold yellow] Note {index} edited:[/] "
                f'[dim italic]"{display_old}"[/] [dim]→[/] [italic]"{display_new}"[/]',
                extra={"markup": True},
            )
        elif action_type == "note_remove":
            index = params["index"]
            if index < 1 or index > len(self.notes):
                self.logger.error(
                    f"[bold red]{TIMES} Invalid note index: {index}[/] "
                    f"[dim](valid range: 1-{len(self.notes)})[/]",
                    extra={"markup": True},
                )
                raise ValueError(
                    f"Invalid note index: {index}. Valid range: 1-{len(self.notes)}"
                )
            removed = self.notes.pop(index - 1)
            display_removed = removed[:50] + "..." if len(removed) > 50 else removed
            self.logger.info(
                f'[bold red] Note {index} removed:[/] [italic]"{display_removed}"[/] '
                f"[dim]({len(self.notes)} notes remaining)[/]",
                extra={"markup": True},
            )
        else:
            self.logger.error(
                f"[bold red]{TIMES} Unknown action type: {action}[/]",
                extra={"markup": True},
            )
            raise ValueError(f"Unknown action type: {action}")

        end_time = time.perf_counter()
        return end_time - start_time, status

    def execute(self, action: Action) -> tuple[float, bool]:
        try:
            duration, status = self._execute(action)
        except Exception as e:
            self.logger.error(f"Error in executing action {action}: {e}")
            duration = -1
            status = False
        return duration, status

    def _compress_history(self):
        if not self.history:
            return []
        history = []
        cnt = 0
        screens_num = len(self.history) // 2
        screen_start_idx = max(0, screens_num - 5)

        self.logger.debug(f"Compressing history: {len(self.history)} items")

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
            duration, status = self.execute(forced_action)
            # Record action for tracking if enabled
            if self.tracker:
                screenshot = self.device.capture_screenshot("B64")
                self._record_action(forced_action, screenshot, duration)
                # Finalize tracking
                result = self._finalize_tracking(result=status)
                if result:
                    trace_path, json_path = result
                    self.logger.debug(f"Generated {trace_path} and {json_path}")
                self.task_status = False
            return forced_action

        self.logger.debug("Capturing screenshot...")
        screenshot = self.device.capture_screenshot("B64")
        self.logger.debug(f"Screenshot captured: {len(screenshot)} chars (base64)")

        # Store screenshot for tracking (reused from LLM capture)
        screenshot_for_tracker = screenshot

        if self.history:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{screenshot}"},
                        },
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{screenshot}"},
                        },
                    ],
                }
            ]

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
        duration, status = self.execute(action)

        # Record action for tracking
        if self.tracker:
            self._record_action(action, screenshot_for_tracker, duration)

            # Finalize tracking on finished action
            if action["action_type"] == "finished":
                self.task_status = status
                result = self._finalize_tracking(result=status)
                if result:
                    trace_path, json_path = result
                    self.logger.debug(f"Generated {trace_path} and {json_path}")

        self.history.extend(messages)
        self.history.append({"role": "assistant", "content": response.response})

        return action
