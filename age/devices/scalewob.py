import time
from typing import Literal, overload

from PIL.Image import Image
from scalewob.automation import ScaleWoBAutomation

from age.icons import KEYBOARD, TIMES
from age.logger import get_logger

from .base import Device, Direction


class ScaleWoBDevice(Device):
    def __init__(
        self,
        env_id: str,
        headless: bool = False,
        platform: Literal["mobile", "desktop"] = "desktop",
    ):
        self.logger = get_logger(__name__)
        self._env_id = env_id
        self._headless = headless
        self._platform = platform
        self._auto = ScaleWoBAutomation(env_id, headless, platform=platform)
        self.logger.info(
            f"[cyan]ScaleWoB Device initialized:[/] env_id={env_id}, platform={platform}, headless={headless}",
            extra={"markup": True},
        )

    def __enter__(self):
        try:
            self._auto.start()
            self.logger.info(
                f"{KEYBOARD} ScaleWoB automation started",
                extra={"markup": True},
            )
            return self
        except Exception as e:
            self.logger.error(
                f"[bold red]{TIMES} Failed to start ScaleWoB automation:[/] {e}",
                extra={"markup": True},
            )
            raise

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        try:
            self._auto.close()
            self.logger.info("ScaleWoB automation closed")
        except Exception as e:
            self.logger.error(
                f"[bold red]{TIMES} Error closing ScaleWoB automation:[/] {e}",
                extra={"markup": True},
            )
        return False

    @overload
    def capture_screenshot(self, format: Literal["PIL"]) -> Image: ...

    @overload
    def capture_screenshot(self, format: Literal["B64"]) -> str: ...

    def capture_screenshot(self, format: Literal["PIL", "B64"]) -> Image | str:
        if format == "PIL":
            result = self._auto.take_screenshot(format="pil")
            self.logger.debug(f"Screenshot captured (PIL): {result.size}")
            return result
        elif format == "B64":
            result = self._auto.take_screenshot(format="base64")
            self.logger.debug(f"Screenshot captured (B64): {len(result)} chars")
            return result
        raise ValueError(
            f"Supported formats: PIL & B64. The provided format {format} is not valid."
        )

    @property
    def size(self) -> tuple[int, int]:
        if self._platform == "mobile":
            return 390 * 3, 844 * 3
        return 1280 * 3, 800 * 3

    def _normalize_coordinates(self, x: int, y: int) -> tuple[int, int]:
        return int(x * self.size[0] / 1000), int(y * self.size[1] / 1000)

    def press(self, x: int, y: int):
        self.logger.debug(f"press({x}, {y})")
        try:
            x_norm, y_norm = self._normalize_coordinates(x, y)
            self.logger.debug(f"Normalized to: ({x_norm}, {y_norm})")
            self._auto.click(x_norm, y_norm)
            time.sleep(1)
        except Exception as e:
            self.logger.error(
                f"[bold red]{TIMES} press({x}, {y}) failed:[/] {e}",
                extra={"markup": True},
            )
            raise

    def long_press(self, x: int, y: int, duration: float):
        self.logger.debug(f"long_press({x}, {y}, {duration}s)")
        try:
            x_norm, y_norm = self._normalize_coordinates(x, y)
            self.logger.debug(f"Normalized to: ({x_norm}, {y_norm})")
            self._auto.long_press(x_norm, y_norm, int(duration * 1000))
            time.sleep(1)
        except Exception as e:
            self.logger.error(
                f"[bold red]{TIMES} long_press({x}, {y}, {duration}s) failed:[/] {e}",
                extra={"markup": True},
            )
            raise

    def type_text(self, text: str):
        display_text = text[:50] + "..." if len(text) > 50 else text
        self.logger.debug(f'type_text("{display_text}")')
        try:
            self._auto.type(text, append=True)
            time.sleep(1)
        except Exception as e:
            self.logger.error(
                f"[bold red]{TIMES} type_text failed:[/] {e}",
                extra={"markup": True},
            )
            raise

    def clear_text(self):
        self.logger.debug("clear_text()")
        try:
            self._auto.type("")
            time.sleep(1)
        except Exception as e:
            self.logger.error(
                f"[bold red]{TIMES} clear_text failed:[/] {e}",
                extra={"markup": True},
            )
            raise

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int):
        self.logger.debug(f"drag({start_x}, {start_y}) → ({end_x}, {end_y})")
        try:
            start_x_norm, start_y_norm = self._normalize_coordinates(start_x, start_y)
            end_x_norm, end_y_norm = self._normalize_coordinates(end_x, end_y)
            self.logger.debug(
                f"Normalized: ({start_x_norm}, {start_y_norm}) → ({end_x_norm}, {end_y_norm})"
            )
            self._auto.drag(start_x_norm, start_y_norm, end_x_norm, end_y_norm)
            time.sleep(1)
        except Exception as e:
            self.logger.error(
                f"[bold red]{TIMES} drag({start_x}, {start_y}) → ({end_x}, {end_y}) failed:[/] {e}",
                extra={"markup": True},
            )
            raise

    def scroll(self, start_x: int, start_y: int, direction: Direction, distance: int):
        self.logger.debug(f"scroll({start_x}, {start_y}, {direction}, {distance})")
        try:
            start_x_norm, start_y_norm = self._normalize_coordinates(start_x, start_y)
            self.logger.debug(f"Normalized start: ({start_x_norm}, {start_y_norm})")
            self._auto.scroll(start_x_norm, start_y_norm, str(direction), distance)
            time.sleep(1)
        except Exception as e:
            self.logger.error(
                f"[bold red]{TIMES} scroll({start_x}, {start_y}, {direction}, {distance}) failed:[/] {e}",
                extra={"markup": True},
            )
            raise

    def navigate_back(self):
        self.logger.debug("navigate_back()")
        try:
            self._auto.back()
            time.sleep(1)
        except Exception as e:
            self.logger.error(
                f"[bold red]{TIMES} navigate_back failed:[/] {e}",
                extra={"markup": True},
            )
            raise

    def start_recording(self, video_path: str | None = None) -> bool:
        """ScaleWoB does not support video recording.

        This method exists for compatibility with ActionTracker but always returns False.

        Args:
            video_path: Ignored (kept for API compatibility)

        Returns:
            False (video recording not supported)
        """
        self.logger.debug("Video recording not supported on ScaleWoB device")
        return False

    def end_recording(self) -> None:
        """ScaleWoB does not support video recording.

        This method exists for compatibility with ActionTracker but does nothing.
        """
        pass

    @property
    def recording_error(self) -> str | None:
        """ScaleWoB does not support video recording.

        Returns:
            Error message indicating video recording is not supported
        """
        return "Video recording not supported on ScaleWoB device"
