"""Action tracking system for generating execution traces with annotated screenshots.

This module provides the ActionTracker class which records agent actions,
annotates screenshots with action visualizations, and generates markdown
execution trace documents.
"""

import base64
import hashlib
import io
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as PILImage

from .agents import Action
from .icons import FILE, FOLDER, TIMES, VIDEO


class MarkerColor(str, Enum):
    """Colors for action markers matching agent.py color scheme."""

    CLICK = "#00FF00"  # green
    TYPE = "#0000FF"  # blue
    SCROLL = "#FFFF00"  # yellow
    DRAG = "#800080"  # purple
    SYSTEM = "#00FFFF"  # cyan
    WAIT = "#808080"  # gray
    NOTEPAD = "#00FFFF"  # cyan
    FINISHED = "#00FF00"  # green


class ActionTracker:
    """Tracks agent actions and generates annotated execution traces.

    This class captures screenshots, annotates them with action visualizations,
    and generates markdown documents showing the complete execution trace.
    """

    def __init__(
        self,
        task_name: str,
        model_name: str,
        language: str,
        output_dir: Path | str,
        logger: logging.Logger,
        screen_size: tuple[int, int],
        touch_range: tuple[int, int] = (32767, 32767),
        **kwargs,
    ) -> None:
        """Initialize the action tracker.

        Args:
            task_name: The task description
            model_name: The LLM model being used
            language: The language setting (CN/EN)
            output_dir: Directory to store tracking output
            logger: Logger instance for output
            screen_size: Device screen size as (width, height) in pixels
            touch_range: Max touch coordinates as (x_max, y_max)
        """
        self.task_name = task_name
        self.model_name = model_name
        self.language = language
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.additional_metadata = kwargs

        # Create task directory with 8-digit hash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_code = self._generate_hash(task_name)
        self.task_dir = self.output_dir / f"task_{timestamp}_{hash_code}"
        self.task_dir.mkdir(parents=True, exist_ok=True)

        # Create screenshots subdirectory
        self.screenshots_dir = self.task_dir / "screenshots"
        self.screenshots_dir.mkdir(exist_ok=True)

        # Load font for annotations
        self._load_font()

        # Store recorded actions
        self.actions: list[dict] = []

        # Video recording state
        self.video_enabled: bool = False
        self.video_path: Path = self.task_dir / "recording.mp4"
        self.video_start_timestamp: datetime | None = None

        # Device info for JSON output
        self.screen_size = {"width": screen_size[0], "height": screen_size[1]}
        self.touch_range = {"x_max": touch_range[0], "y_max": touch_range[1]}
        self.session_start = datetime.now()

        self.logger.info(
            f"[cyan]{FOLDER} Tracking directory:[/] {self.task_dir}",
            extra={"markup": True},
        )

    def record_action(
        self,
        action: Action,
        screenshot_b64: str,
        timestamp: datetime,
        duration: float = 0.0,
    ) -> None:
        """Record an action with annotated screenshot.

        Args:
            action: The action dict with type, params, and thought
            screenshot_b64: Base64-encoded screenshot (PNG format)
            timestamp: When the action was recorded
            duration: Execution time in seconds
        """
        try:
            # Decode base64 screenshot to PIL Image
            image_data = base64.b64decode(screenshot_b64)
            screenshot = Image.open(io.BytesIO(image_data))

            # Annotate screenshot
            annotated = self._annotate_screenshot(screenshot, action)

            # Save annotated screenshot
            step_num = len(self.actions) + 1
            action_type = action["action_type"]
            screenshot_path = (
                self.screenshots_dir / f"step_{step_num:03d}_{action_type}.png"
            )
            annotated.save(screenshot_path)
            raw_screenshot_path = (
                self.screenshots_dir / f"step_{step_num:03d}_{action_type}_raw.png"
            )
            screenshot.save(raw_screenshot_path)

            # Store action metadata
            self.actions.append({
                "step": step_num,
                "timestamp": timestamp,
                "action": {**action, "duration": duration},
                "screenshot_path": screenshot_path.name,
            })

            self.logger.debug(
                f"[dim] Recorded action {step_num}: {action_type}[/]",
                extra={"markup": True},
            )

        except Exception as e:
            self.logger.error(
                f"[bold red]{TIMES} Tracking failed for action {action['action_type']}:[/] {e}",
                extra={"markup": True},
            )
            # Don't raise - allow agent to continue

    def finalize(self, **kwargs) -> tuple[Path, Path]:
        """Generate markdown trace and JSON log.

        Returns:
            Tuple of (trace_path, json_path) to generated files
        """
        try:
            # Generate markdown
            markdown = self._generate_markdown(**kwargs)
            trace_path = self.task_dir / "trace.md"
            trace_path.write_text(markdown, encoding="utf-8")

            # Generate JSON
            json_data = self._generate_actions_json()
            json_path = self.task_dir / "actions.json"
            json_path.write_text(json_data, encoding="utf-8")

            self.logger.info(
                f"[cyan]{FILE} Trace saved to:[/] {trace_path}\n"
                f"[cyan]{FILE} Actions JSON saved to:[/] {json_path}",
                extra={"markup": True},
            )

            return trace_path, json_path

        except Exception as e:
            self.logger.error(
                f"[bold red]{TIMES} Failed to generate trace/JSON:[/] {e}",
                extra={"markup": True},
            )
            # Return paths even if generation failed
            return self.task_dir / "trace.md", self.task_dir / "actions.json"

    def start_video_recording(self, device) -> bool:
        """Start video recording on the device.

        Args:
            device: Device instance to control recording

        Returns:
            True if recording started successfully, False otherwise
        """
        try:
            device.start_recording(str(self.video_path))
            self.video_enabled = True
            self.video_start_timestamp = datetime.now()
            self.logger.info(
                f"[cyan]{VIDEO} Video recording started:[/] {self.video_path}",
                extra={"markup": True},
            )
            return True
        except Exception as e:
            self.logger.error(
                f"[bold red]{TIMES} Failed to start video recording:[/] {e}",
                extra={"markup": True},
            )
            return False

    def stop_video_recording(self, device) -> bool:
        """Stop video recording.

        Args:
            device: Device instance to control recording

        Returns:
            True if recording stopped successfully, False otherwise
        """
        if not self.video_enabled:
            return False

        try:
            device.end_recording()
            self.logger.info(
                f"[cyan]{VIDEO} Video recording stopped[/]",
                extra={"markup": True},
            )
            self.video_enabled = False
            return True

        except Exception as e:
            self.logger.error(
                f"[bold red]{TIMES} Failed to stop video recording:[/] {e}",
                extra={"markup": True},
            )
            self.video_enabled = False
            return False

    def _generate_video_section(self) -> str:
        """Generate markdown section for video recording.

        Returns:
            Markdown string with video reference or empty string if no video
        """
        if not self.video_path.exists():
            return ""

        lines = ["## Video Recording\n"]

        if self.video_start_timestamp:
            lines.append(f"**Started**: {self.video_start_timestamp.isoformat()}\n")

        lines.append("### Execution Video\n")
        lines.append("[Download video](recording.mp4)\n")
        lines.append('<video controls width="100%">\n')
        lines.append('  <source src="recording.mp4" type="video/mp4">\n')
        lines.append("  Your browser does not support the video tag.\n")
        lines.append("</video>\n")
        lines.append("---\n")

        return "\n".join(lines)

    def _generate_hash(self, name: str) -> str:
        """Generate 8-digit hash from task name.

        Args:
            name: Task name to hash

        Returns:
            8-digit hexadecimal hash code
        """
        hash_bytes = hashlib.md5(name.encode("utf-8")).digest()
        return hash_bytes[:4].hex()

    def _load_font(self) -> None:
        """Load font for screenshot annotations.

        Attempts to load Noto Sans SC font with fallback to default.
        """
        try:
            # Try to load Noto Sans SC Bold (common on Linux systems)
            self.font = ImageFont.truetype(
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 28
            )
            self.font_small = ImageFont.truetype(
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 24
            )
            self.font_large = ImageFont.truetype(
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 36
            )
        except OSError:
            # Fallback to default font with specified sizes
            self.font = ImageFont.load_default(size=28)
            self.font_small = ImageFont.load_default(size=24)
            self.font_large = ImageFont.load_default(size=36)

    def _annotate_screenshot(self, screenshot: PILImage, action: Action) -> PILImage:
        """Create annotated copy of screenshot with action visualization.

        Args:
            screenshot: Original PIL Image
            action: Action dict with type and params

        Returns:
            Annotated screenshot as PIL Image
        """
        # Create copy to avoid modifying original
        annotated = screenshot.copy()
        draw = ImageDraw.Draw(annotated)

        action_type = action["action_type"]
        params = action["params"]

        width, height = screenshot.size

        # Draw appropriate marker based on action type
        if action_type in ("click", "long_press"):
            x, y = self._norm_to_pixels(params.get("point", (500, 500)), width, height)
            self._draw_point_marker(draw, x, y, action_type)

        elif action_type == "scroll":
            x, y = self._norm_to_pixels(params.get("point", (500, 500)), width, height)
            direction = params.get("direction", "down")
            self._draw_scroll_marker(draw, x, y, direction)

        elif action_type == "drag":
            start = self._norm_to_pixels(
                params.get("start_point", (200, 500)), width, height
            )
            end = self._norm_to_pixels(
                params.get("end_point", (800, 500)), width, height
            )
            self._draw_drag_marker(draw, start[0], start[1], end[0], end[1])

        elif action_type == "type":
            content = params.get("content", "")
            self._draw_text_marker(draw, content, width, height)

        elif action_type in ("open_app", "press_home", "press_back"):
            label = action_type.replace("_", " ").title()
            self._draw_system_overlay(draw, label, width, height)

        elif action_type == "wait":
            seconds = params.get("seconds", 1)
            self._draw_wait_marker(draw, seconds, width, height)

        elif action_type.startswith("note_"):
            content = params.get("content", "")
            self._draw_notepad_marker(draw, action_type, content, width, height)

        elif action_type == "finished":
            content = params.get("content", "Task completed")
            self._draw_finished_overlay(draw, content, width, height)

        return annotated

    def _norm_to_pixels(
        self, point: tuple[int, int], width: int, height: int
    ) -> tuple[int, int]:
        """Convert normalized coordinates (0-1000) to device pixels.

        Args:
            point: Normalized (x, y) coordinates
            width: Screen width in pixels
            height: Screen height in pixels

        Returns:
            Pixel (x, y) coordinates
        """
        x, y = point
        return int(x * width / 1000), int(y * height / 1000)

    def _draw_point_marker(
        self, draw: ImageDraw.ImageDraw, x: int, y: int, label: str
    ) -> None:
        """Draw a point marker for click/long_press actions.

        Args:
            draw: ImageDraw object
            x, y: Center coordinates
            label: Label text (click or long_press)
        """
        color = MarkerColor.CLICK

        # Outer circle (semi-transparent effect)
        draw.ellipse([x - 25, y - 25, x + 25, y + 25], outline=color, width=3)

        # Center dot
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)

        # Label
        draw.text((x + 30, y - 10), label, fill=color, font=self.font)

    def _draw_scroll_marker(
        self, draw: ImageDraw.ImageDraw, x: int, y: int, direction: str
    ) -> None:
        """Draw a scroll marker with directional arrow.

        Args:
            draw: ImageDraw object
            x, y: Scroll start coordinates
            direction: Scroll direction (up, down, left, right)
        """
        color = MarkerColor.SCROLL

        # Draw point marker
        self._draw_point_marker(draw, x, y, "scroll")

        # Draw directional arrow
        arrow_map = {
            "up": (0, -50),
            "down": (0, 50),
            "left": (-50, 0),
            "right": (50, 0),
        }
        dx, dy = arrow_map.get(direction, (0, 50))
        draw.line([x, y, x + dx, y + dy], fill=color, width=4)

        # Draw arrowhead
        if direction == "up":
            points = [(x, y - 60), (x - 10, y - 45), (x + 10, y - 45)]
        elif direction == "down":
            points = [(x, y + 60), (x - 10, y + 45), (x + 10, y + 45)]
        elif direction == "left":
            points = [(x - 60, y), (x - 45, y - 10), (x - 45, y + 10)]
        else:  # right
            points = [(x + 60, y), (x + 45, y - 10), (x + 45, y + 10)]

        draw.polygon(points, fill=color)

        # Direction label
        draw.text((x + 35, y + 15), direction, fill=color, font=self.font)

    def _draw_drag_marker(
        self,
        draw: ImageDraw.ImageDraw,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
    ) -> None:
        """Draw a drag marker with start/end points and connecting line.

        Args:
            draw: ImageDraw object
            start_x, start_y: Start coordinates
            end_x, end_y: End coordinates
        """
        # Start point (green circle)
        draw.ellipse(
            [start_x - 20, start_y - 20, start_x + 20, start_y + 20],
            outline=MarkerColor.CLICK,
            width=3,
        )

        # End point (red circle)
        draw.ellipse(
            [end_x - 20, end_y - 20, end_x + 20, end_y + 20],
            outline="#FF0000",
            width=3,
        )

        # Connecting line
        draw.line([start_x, start_y, end_x, end_y], fill=MarkerColor.DRAG, width=3)

        # Labels
        draw.text(
            (start_x + 25, start_y - 10),
            "start",
            fill=MarkerColor.CLICK,
            font=self.font,
        )
        draw.text((end_x + 25, end_y - 10), "end", fill="#FF0000", font=self.font)

    def _draw_text_marker(
        self, draw: ImageDraw.ImageDraw, content: str, width: int, height: int
    ) -> None:
        """Draw a text input marker.

        Args:
            draw: ImageDraw object
            content: Text that was typed
            width: Screen width
            height: Screen height
        """
        color = MarkerColor.TYPE

        # Draw keyboard area indicator at bottom
        keyboard_y = int(height * 0.8)
        draw.rectangle(
            [20, keyboard_y, width - 20, height - 20], outline=color, width=2
        )

        # Show text preview (truncate to 50 chars)
        preview = content[:50] + "..." if len(content) > 50 else content
        draw.text(
            (30, keyboard_y - 30), f"Typed: {preview}", fill=color, font=self.font_small
        )

        # Label
        draw.text((30, keyboard_y + 10), "text input", fill=color, font=self.font_small)

    def _draw_system_overlay(
        self, draw: ImageDraw.ImageDraw, label: str, width: int, height: int
    ) -> None:
        """Draw a semi-transparent overlay for system actions.

        Args:
            draw: ImageDraw object
            label: Action label
            width: Screen width
            height: Screen height
        """
        # Note: PIL doesn't support transparent overlays easily without additional processing
        # We'll just draw text for simplicity

        # Draw label at center
        text = label.upper()
        # Get text size (approximate)
        text_width = len(text) * 10
        text_x = (width - text_width) // 2
        text_y = height // 2

        draw.text((text_x, text_y), text, fill="white", font=self.font_large)

    def _draw_wait_marker(
        self, draw: ImageDraw.ImageDraw, seconds: int, width: int, height: int
    ) -> None:
        """Draw a wait indicator.

        Args:
            draw: ImageDraw object
            seconds: Number of seconds waited
            width: Screen width
            height: Screen height (unused, for API consistency)
        """
        color = MarkerColor.WAIT

        # Draw in top-right corner
        text = f"wait {seconds}s"
        draw.text((width - 150, 20), text, fill=color, font=self.font)

    def _draw_notepad_marker(
        self,
        draw: ImageDraw.ImageDraw,
        action_type: str,
        content: str,
        width: int,
        height: int,
    ) -> None:
        """Draw a notepad action indicator.

        Args:
            draw: ImageDraw object
            action_type: Type of note action
            content: Note content
            width: Screen width
            height: Screen height (unused, for API consistency)
        """
        color = MarkerColor.NOTEPAD

        # Draw in top-right corner
        action_label = action_type.replace("note_", "").title()
        text = action_label
        draw.text((width - 150, 50), text, fill=color, font=self.font)

        # Show preview if content exists
        if content:
            preview = content[:30] + "..." if len(content) > 30 else content
            draw.text((width - 150, 75), preview, fill=color, font=self.font_small)

    def _draw_finished_overlay(
        self, draw: ImageDraw.ImageDraw, content: str, width: int, height: int
    ) -> None:
        """Draw a completion overlay.

        Args:
            draw: ImageDraw object
            content: Completion message
            width: Screen width
            height: Screen height
        """
        color = MarkerColor.FINISHED

        # Draw message at center
        draw.text(
            (width // 2 - 100, height // 2),
            "TASK COMPLETED",
            fill=color,
            font=self.font_large,
        )

        if content:
            preview = content[:50] + "..." if len(content) > 50 else content
            draw.text(
                (width // 2 - 150, height // 2 + 50),
                preview,
                fill=color,
                font=self.font,
            )

    def _calc_scroll_end(self, start: tuple[int, int], direction: str) -> dict:
        """Calculate scroll end position based on direction.

        Args:
            start: Normalized (x, y) start coordinates
            direction: Scroll direction (up, down, left, right)

        Returns:
            {"x": int, "y": int} end position
        """
        distance = 500  # Approximate scroll distance in normalized coordinates
        vectors = {
            "up": (0, -distance),
            "down": (0, distance),
            "left": (-distance, 0),
            "right": (distance, 0),
        }
        dx, dy = vectors.get(direction, (0, distance))
        return {"x": start[0] + dx, "y": start[1] + dy}

    def _calc_distance(self, start: tuple[int, int], end: tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points.

        Args:
            start: (x, y) start coordinates
            end: (x, y) end coordinates

        Returns:
            Distance as float
        """
        return round(((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5, 1)

    def _calc_direction(self, start: tuple[int, int], end: tuple[int, int]) -> str:
        """Calculate primary direction from start to end.

        Args:
            start: (x, y) start coordinates
            end: (x, y) end coordinates

        Returns:
            Direction string: "up", "down", "left", or "right"
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"

    def _generate_markdown(self, **kwargs) -> str:
        """Generate markdown execution trace.

        Returns:
            Complete markdown document as string
        """
        self.additional_metadata.update(kwargs)
        lines = [
            f"# Task Execution Trace: {self.task_name}",
            "",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Model**: {self.model_name}",
            f"**Language**: {self.language}",
            f"**Total Steps**: {len(self.actions)}",
            *[f"**{k}**: {v}" for k, v in self.additional_metadata.items()],
            "",
            "---",
            "",
        ]

        # Add video section
        video_section = self._generate_video_section()
        if video_section:
            lines.append(video_section)

        for action_data in self.actions:
            step = action_data["step"]
            timestamp = action_data["timestamp"]
            action = action_data["action"]
            screenshot_path = action_data["screenshot_path"]

            action_type = action["action_type"]
            thought = action.get("thought", "")
            params = action["params"]

            # Step header
            lines.append(f"## Step {step}: {action_type.replace('_', ' ').title()}")
            lines.append("")

            # Metadata
            lines.append(f"**Time**: {timestamp.isoformat()}")
            lines.append(f"**Action Type**: `{action_type}`")

            if thought:
                # Escape markdown special characters in thought
                escaped_thought = self._escape_markdown(thought)
                lines.append(f"**Thought**: {escaped_thought}")

            lines.append("")

            # Parameters
            lines.append("**Parameters**:")
            lines.append("```python")
            lines.append(f"{params}")
            lines.append("```")
            lines.append("")

            # Screenshot
            lines.append("**Screenshot**:")
            lines.append(f"![Step {step}](screenshots/{screenshot_path})")
            lines.append("")
            lines.append("---")
            lines.append("")

        # Footer
        if any(a["action"]["action_type"] == "finished" for a in self.actions):
            lines.append("")
            lines.append("*Task completed successfully*")

        return "\n".join(lines)

    def _escape_markdown(self, text: str) -> str:
        """Escape special markdown characters.

        Args:
            text: Raw text

        Returns:
            Text with markdown special characters escaped
        """
        # Escape basic markdown characters
        escape_chars = r"\\`*_{}[]<>()#+-.!|"
        for char in escape_chars:
            text = text.replace(char, f"\\{char}")
        return text

    def _convert_action_to_json(self, action_data: dict) -> dict | None:
        """Convert internal action format to JSON action format.

        Args:
            action_data: Internal action dict with step, timestamp, action, screenshot_path

        Returns:
            JSON-formatted action dict or None for unknown action types
        """
        action = action_data["action"]
        action_type = action["action_type"]
        params = action["params"]
        duration = action.get("duration", 0.0)

        timestamp = action_data["timestamp"]
        relative_time = (timestamp - self.session_start).total_seconds()
        unix_ts = timestamp.timestamp()

        # Click actions
        if action_type in ("click", "long_press"):
            point = params.get("point", (500, 500))
            return {
                "type": "click",
                "timestamp": unix_ts,
                "relative_time": round(relative_time, 3),
                "position": {"x": point[0], "y": point[1]},
                "duration": round(duration, 3),
            }

        # Scroll action
        if action_type == "scroll":
            point = params.get("point", (500, 500))
            direction = params.get("direction", "down")
            return {
                "type": "swipe",
                "timestamp": unix_ts,
                "relative_time": round(relative_time, 3),
                "start_position": {"x": point[0], "y": point[1]},
                "end_position": self._calc_scroll_end(point, direction),
                "distance": 500,  # Approximate scroll distance in normalized coords
                "direction": direction,
                "duration": round(duration, 3),
            }

        # Drag action
        if action_type == "drag":
            start = params.get("start_point", (200, 500))
            end = params.get("end_point", (800, 500))
            return {
                "type": "swipe",
                "timestamp": unix_ts,
                "relative_time": round(relative_time, 3),
                "start_position": {"x": start[0], "y": start[1]},
                "end_position": {"x": end[0], "y": end[1]},
                "distance": self._calc_distance(start, end),
                "direction": self._calc_direction(start, end),
                "duration": round(duration, 3),
            }

        # Key press actions
        if action_type == "press_home":
            return {
                "type": "key_press",
                "timestamp": unix_ts,
                "relative_time": round(relative_time, 3),
                "key": "KEY_HOME",
                "duration": round(duration, 3),
            }

        if action_type == "press_back":
            return {
                "type": "key_press",
                "timestamp": unix_ts,
                "relative_time": round(relative_time, 3),
                "key": "KEY_BACK",
                "duration": round(duration, 3),
            }

        # Type action
        if action_type == "type":
            return {
                "type": "type",
                "timestamp": unix_ts,
                "relative_time": round(relative_time, 3),
                "content": params.get("content", ""),
                "duration": round(duration, 3),
            }

        # Open app action
        if action_type == "open_app":
            return {
                "type": "open_app",
                "timestamp": unix_ts,
                "relative_time": round(relative_time, 3),
                "app_name": params.get("app", ""),
                "duration": round(duration, 3),
            }

        # Wait action
        if action_type == "wait":
            return {
                "type": "wait",
                "timestamp": unix_ts,
                "relative_time": round(relative_time, 3),
                "seconds": params.get("seconds", 1),
                "duration": round(duration, 3),
            }

        # Notepad actions
        if action_type == "note_create":
            return {
                "type": "note_create",
                "timestamp": unix_ts,
                "relative_time": round(relative_time, 3),
                "content": params.get("content", ""),
                "duration": round(duration, 3),
            }

        if action_type == "note_edit":
            return {
                "type": "note_edit",
                "timestamp": unix_ts,
                "relative_time": round(relative_time, 3),
                "index": params.get("index", 1),
                "content": params.get("content", ""),
                "duration": round(duration, 3),
            }

        if action_type == "note_remove":
            return {
                "type": "note_remove",
                "timestamp": unix_ts,
                "relative_time": round(relative_time, 3),
                "index": params.get("index", 1),
                "duration": round(duration, 3),
            }

        # Finished action
        if action_type == "finished":
            return {
                "type": "finished",
                "timestamp": unix_ts,
                "relative_time": round(relative_time, 3),
                "content": params.get("content", ""),
                "duration": round(duration, 3),
            }

        # Unknown action type - shouldn't happen
        return None

    def _generate_actions_json(self) -> str:
        """Generate machine-readable JSON action log.

        Returns:
            Complete JSON document as string
        """
        import json

        session_end = datetime.now()

        # Convert actions to JSON format
        json_actions = []
        for action_data in self.actions:
            json_action = self._convert_action_to_json(action_data)
            if json_action:
                json_actions.append(json_action)

        # Build complete JSON structure
        output = {
            "device_info": {
                "screen_size": self.screen_size,
                "touch_range": self.touch_range,
            },
            "session_start": self.session_start.isoformat(),
            "session_end": session_end.isoformat(),
            "video_recording_start": (
                self.video_start_timestamp.isoformat()
                if self.video_start_timestamp
                else None
            ),
            "actions": json_actions,
        }

        return json.dumps(output, indent=2, ensure_ascii=False)
