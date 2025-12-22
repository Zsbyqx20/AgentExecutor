import base64
import time
from enum import Enum
from io import BytesIO
from typing import Literal, overload

from adbutils import adb
from PIL.Image import Image


class Direction(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class Device:
    def __init__(self, device_id: str | None = None) -> None:
        available_devices = {str(s.serial): s for s in adb.device_list()}
        assert available_devices, "Currently no adb devices available!"
        if isinstance(device_id, str):
            if device_id in available_devices:
                self._device_id = device_id
            else:
                raise ValueError(
                    f"The provided device id {device_id} is not valid. Current available ids: {list(available_devices.keys())}"
                )
            self._device = available_devices[self._device_id]
        else:
            self._device_id, self._device = min(sorted(available_devices.items()))
        self._set_ime()
        self._is_recording = False

    def _set_ime(self):
        target_ime = "com.android.adbkeyboard/.AdbIME"
        ime = self._device.shell(["settings", "get", "secure", "default_input_method"])
        if ime != target_ime:
            try:
                self._device.shell(["ime", "enable", target_ime])
                self._device.shell(["ime", "set", target_ime])
            except Exception as e:
                raise e

    @overload
    def capture_screenshot(self, format: Literal["PIL"]) -> Image: ...

    @overload
    def capture_screenshot(self, format: Literal["B64"]) -> str: ...

    def capture_screenshot(self, format: Literal["PIL", "B64"]) -> Image | str:
        screen = self._device.screenshot()
        if format == "PIL":
            return screen
        buffered = BytesIO()
        screen.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    @property
    def size(self):
        return self._device.window_size()

    def press(self, x: int, y: int) -> None:
        self._device.click(x, y)
        time.sleep(0.5)

    def long_press(self, x: int, y: int, duration: float):
        self._device.swipe(x, y, x, y, duration)
        time.sleep(0.5)

    def type_text(self, text: str):
        b64_text = str(base64.b64encode(text.encode("utf-8")))[1:]
        self._device.shell([
            "am",
            "broadcast",
            "-a",
            "ADB_INPUT_B64",
            "--es",
            "msg",
            b64_text,
        ])
        time.sleep(0.5)

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float):
        self._device.drag(start_x, start_y, end_x, end_y, duration)
        time.sleep(0.5)

    def open_app(self, package_name: str):
        self._device.shell([
            "monkey",
            "-p",
            package_name,
            "-c",
            "android.intent.category.LAUNCHER",
            "1",
        ])
        time.sleep(0.5)

    def scroll(
        self,
        start_x: int,
        start_y: int,
        direction: Direction,
        distance: int,
    ):
        coord_map = {
            "up": (0, distance),
            "down": (0, -distance),
            "left": (-distance, 0),
            "right": (distance, 0),
        }
        delta_x, delta_y = coord_map[direction]
        end_x = delta_x + start_x
        end_y = delta_y + start_y
        self._device.swipe(start_x, start_y, end_x, end_y)

    def navigate_home(self):
        self._device.keyevent("HOME")
        time.sleep(0.5)

    def navigate_back(self):
        self._device.keyevent("BACK")
        time.sleep(0.5)

    def start_recording(self, video_path: str):
        self._device.start_recording(video_path)
        self._is_recording = True

    def end_recording(self):
        if self._is_recording:
            self._device.stop_recording()
