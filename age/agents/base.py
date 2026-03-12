from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TypedDict


class Action(TypedDict):
    action_type: str
    params: dict
    thought: str


class Language(str, Enum):
    CN = "Chinese"
    EN = "English"


class Agent(ABC):
    def __init__(
        self,
        enable_tracking: bool = False,
        tracking_output_dir: str | Path | None = None,
        # Tracking metadata (subclasses should provide these)
        task_name: str = "",
        model_name: str = "",
        language: str = "Chinese",
        # Step limiting to prevent infinite loops
        max_steps: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.tracker = None
        self._task_name = task_name
        self._model_name = model_name
        self._language = language
        self.max_steps = max_steps
        self.current_step = 0

        if enable_tracking:
            self._init_tracker(tracking_output_dir, **kwargs)

    def _init_tracker(self, output_dir: str | Path | None = None, **kwargs) -> None:
        """Initialize the action tracking system.

        This method sets up the ActionTracker for recording agent actions
        with annotated screenshots and video recording.
        """
        from ..logger import get_logger
        from ..tracker import ActionTracker

        logger = get_logger(__name__)
        self.tracker = ActionTracker(
            task_name=self._task_name,
            model_name=self._model_name,
            language=self._language,
            output_dir=output_dir or Path("tracks"),
            logger=logger,
            screen_size=self.device.size,  # type: ignore
            touch_range=(32767, 32767),
            **kwargs,
        )
        self.tracker.start_video_recording(self.device)  # type: ignore

    def _record_action(
        self, action: Action, screenshot_b64: str, duration: float
    ) -> None:
        """Record an action with its associated screenshot and execution time.

        Args:
            action: The action that was executed
            screenshot_b64: Base64-encoded screenshot of the device state
            duration: Execution time in seconds
        """
        if self.tracker:
            self.tracker.record_action(
                action=action,
                screenshot_b64=screenshot_b64,
                timestamp=datetime.now(UTC),
                duration=duration,
            )

    def _finalize_tracking(self, **kwargs) -> tuple[Path, Path] | None:
        """Finalize tracking and generate the execution trace.

        Stops video recording and generates markdown trace and JSON log files.

        Returns:
            Tuple of (trace_path, json_path) if tracking was enabled,
            None otherwise.
        """
        if self.tracker:
            self.tracker.stop_video_recording(self.device)  # type: ignore
            trace_path, json_path = self.tracker.finalize(**kwargs)
            return trace_path, json_path
        return None

    def _check_step_limit(self) -> Action | None:
        """Check if the step limit has been reached.

        This method should be called at the beginning of each predict() call
        to prevent infinite loops.

        Returns:
            A forced 'finished' action if the step limit is exceeded,
            None otherwise.

        Example:
            def predict(self) -> Action:
                # Check step limit first
                forced_action = self._check_step_limit()
                if forced_action:
                    # Execute, track, and return the forced action
                    duration = self._execute(forced_action)
                    if self.tracker:
                        screenshot = self.device.capture_screenshot("B64")
                        self._record_action(forced_action, screenshot, duration)
                        result = self._finalize_tracking()
                        if result:
                            trace_path, json_path = result
                            self.logger.debug(f"Generated {trace_path} and {json_path}")
                    return forced_action

                # Continue with normal prediction...
        """
        if self.max_steps is None:
            return None

        self.current_step += 1
        if self.current_step > self.max_steps:
            return Action(
                action_type="finished",
                params={"content": "Maximum steps reached"},
                thought="Reached maximum step limit - forcing completion",
            )
        return None

    @abstractmethod
    def predict(self) -> Action:
        pass

    def reset(self, **kwargs) -> None:
        """Reset the agent state.

        This method resets the step counter. Subclasses should override
        this method to reset their own state and call super().reset(**kwargs)
        to ensure the step counter is also reset.
        """
        self.current_step = 0

    def __enter__(self):
        """Context manager entry - returns the Agent instance."""
        # Let the device manage its own lifecycle.
        if hasattr(self, "device"):
            self.device.__enter__()  # type: ignore[attr-defined]
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit - finalizes tracking and restores device IME.

        This method is guaranteed to be called even when exceptions occur,
        including KeyboardInterrupt (Ctrl+C).

        Args:
            _exc_type: Exception type if an exception was raised (unused)
            _exc_val: Exception value if an exception was raised (unused)
            _exc_tb: Exception traceback if an exception was raised (unused)

        Returns:
            False to propagate any exceptions (don't suppress them)
        """
        try:
            # Finalize tracking if enabled
            self._finalize_tracking()
        finally:
            if hasattr(self, "device"):
                self.device.__exit__(_exc_type, _exc_val, _exc_tb)  # type: ignore[attr-defined]
        return False  # Don't suppress exceptions


__all__ = ["Action", "Agent", "Language"]
