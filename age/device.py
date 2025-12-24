import base64
import re
import time
from difflib import get_close_matches
from enum import Enum
from io import BytesIO
from typing import Literal, overload

from adbutils import adb
from PIL.Image import Image

from .icons import KEYBOARD, PHONE, TIMES, VIDEO, WARNING
from .logger import get_logger


class Direction(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class Device:
    def __init__(self, device_id: str | None = None) -> None:
        self.logger = get_logger(__name__)

        available_devices = {str(s.serial): s for s in adb.device_list()}
        assert available_devices, "Currently no adb devices available!"

        if isinstance(device_id, str):
            if device_id in available_devices:
                self._device_id = device_id
                self.logger.info(
                    f"[cyan]{PHONE} Using device:[/] [dim]{device_id}[/]",
                    extra={"markup": True},
                )
            else:
                self.logger.error(
                    f"[bold red]{TIMES} Invalid device ID: {device_id}[/]\n"
                    f"[dim]Available: {list(available_devices.keys())}[/]",
                    extra={"markup": True},
                )
                raise ValueError(
                    f"The provided device id {device_id} is not valid. Current available ids: {list(available_devices.keys())}"
                )
            self._device = available_devices[self._device_id]
        else:
            self._device_id, self._device = min(sorted(available_devices.items()))
            self.logger.info(
                f"[cyan]{PHONE} Auto-selected device:[/] [dim]{self._device_id}[/]",
                extra={"markup": True},
            )

        self._set_ime()
        self._is_recording = False

        self.app_name_mapping = self._device.shell([
            "content",
            "query",
            "--uri",
            "content://com.mobilellm.appscan.provider/apps?launchable_only=1",
        ])
        self._app_mapping, self._app_mapping_lower, self._all_apps = (
            self._parse_app_mapping(self.app_name_mapping)
        )
        self.logger.info(
            f"[cyan]{PHONE} Loaded {len(self._all_apps)} launchable apps[/]",
            extra={"markup": True},
        )

    def _parse_app_mapping(self, raw_output: str) -> tuple[dict, dict, list]:
        """Parse content provider output into app mappings.

        Args:
            raw_output: Raw output from content query command

        Returns:
            Tuple of (name_to_package, name_to_package_lower, all_apps)
        """
        name_to_package = {}
        name_to_package_lower = {}
        all_apps = []

        # Split into rows
        rows = raw_output.split("Row:")

        for row in rows:
            if not row.strip():
                continue

            # Extract display_name and package_name
            display_match = re.search(r"display_name=([^\s]+),", row)
            package_match = re.search(r"package_name=([^\s]+),", row)

            if display_match and package_match:
                display_name = display_match.group(1)
                package_name = package_match.group(1)

                name_to_package[display_name] = package_name
                name_to_package_lower[display_name.lower()] = package_name
                all_apps.append({"name": display_name, "package": package_name})

        self.logger.debug(f"Parsed {len(all_apps)} apps from content provider")
        return name_to_package, name_to_package_lower, all_apps

    def _set_ime(self):
        target_ime = "com.android.adbkeyboard/.AdbIME"
        ime = self._device.shell(["settings", "get", "secure", "default_input_method"])
        self.logger.debug(f"Current IME: {ime}")

        if ime != target_ime:
            try:
                self._device.shell(["ime", "enable", target_ime])
                self._device.shell(["ime", "set", target_ime])
                self.logger.info(
                    f"{KEYBOARD} ADB keyboard enabled", extra={"markup": True}
                )
            except Exception as e:
                self.logger.error(
                    f"{TIMES} Failed to set IME: {e}", extra={"markup": True}
                )
                raise

    @overload
    def capture_screenshot(self, format: Literal["PIL"]) -> Image: ...

    @overload
    def capture_screenshot(self, format: Literal["B64"]) -> str: ...

    def capture_screenshot(self, format: Literal["PIL", "B64"]) -> Image | str:
        screen = self._device.screenshot()
        if format == "PIL":
            self.logger.debug(f"Screenshot captured (PIL): {screen.size}")
            return screen
        buffered = BytesIO()
        screen.save(buffered, format="PNG")
        result = base64.b64encode(buffered.getvalue()).decode()
        self.logger.debug(f"Screenshot captured (B64): {len(result)} chars")
        return result

    @property
    def size(self):
        return self._device.window_size()

    def press(self, x: int, y: int) -> None:
        self.logger.debug(f"press({x}, {y})")
        self._device.click(x, y)
        time.sleep(0.5)

    def long_press(self, x: int, y: int, duration: float):
        self.logger.debug(f"long_press({x}, {y}, {duration}s)")
        self._device.swipe(x, y, x, y, duration)
        time.sleep(0.5)

    def type_text(self, text: str):
        display_text = text[:50] + "..." if len(text) > 50 else text
        self.logger.debug(f'type_text("{display_text}")')
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
        self.logger.debug(
            f"drag({start_x}, {start_y}) → ({end_x}, {end_y}), {duration}s"
        )
        self._device.drag(start_x, start_y, end_x, end_y, duration)
        time.sleep(0.5)

    def list_apps(self) -> list[dict]:
        """Return list of all available launchable apps.

        Returns:
            List of dicts with 'name' and 'package' keys
        """
        return self._all_apps

    def _find_app_package(self, app_identifier: str) -> str | None:
        """Find package name for app identifier.

        Args:
            app_identifier: Display name or package name

        Returns:
            Package name if found, None otherwise
        """
        # Check if it looks like a package name (contains dots)
        if "." in app_identifier:
            # Validate it's a known package
            for app in self._all_apps:
                if app["package"] == app_identifier:
                    return app_identifier
            return None

        # Try exact display name match
        if app_identifier in self._app_mapping:
            return self._app_mapping[app_identifier]

        # Try case-insensitive match
        lower_key = app_identifier.lower()
        if lower_key in self._app_mapping_lower:
            # Find the actual display name for logging
            for app in self._all_apps:
                if app["name"].lower() == lower_key:
                    self.logger.debug(
                        f"Using case-insensitive match: '{app_identifier}' → '{app['name']}'"
                    )
                    return self._app_mapping_lower[lower_key]

        return None

    def _find_similar_apps(self, app_name: str, max_results: int = 3) -> list[str]:
        """Find apps with similar names for error suggestions.

        Args:
            app_name: The app name to find similar matches for
            max_results: Maximum number of results to return

        Returns:
            List of similar app names
        """
        all_names = [app["name"] for app in self._all_apps]
        similar = get_close_matches(app_name, all_names, n=max_results, cutoff=0.6)
        return similar

    def open_app(self, app_identifier: str):
        """Open an app by display name or package name.

        Args:
            app_identifier: App display name (e.g., "Chrome") or package name
                           (e.g., "com.android.chrome")

        Raises:
            ValueError: If app not found
        """
        self.logger.debug(f'open_app("{app_identifier}")')

        # Find package name
        package_name = self._find_app_package(app_identifier)

        if package_name is None:
            # App not found - try fuzzy matching
            similar = self._find_similar_apps(app_identifier)

            # If only one fuzzy match, use it automatically
            if len(similar) == 1:
                matched_app = similar[0]
                package_name = self._app_mapping[matched_app]
                self.logger.warning(
                    f'[yellow]{WARNING} App "{app_identifier}" not found, using closest match: "{matched_app}"[/]',
                    extra={"markup": True},
                )
            else:
                # Multiple matches or no matches - show error
                error_msg = f'[bold red]{TIMES} App "{app_identifier}" not found[/]\n'

                if similar:
                    error_msg += "[dim]Did you mean one of these?[/]\n"
                    for app in similar:
                        error_msg += f"  • {app}\n"

                error_msg += f"[dim]Available apps: use device.list_apps() to see all {len(self._all_apps)} apps[/]"

                self.logger.error(error_msg, extra={"markup": True})
                raise ValueError(f'App "{app_identifier}" not found')

        # Launch the app
        self._device.shell([
            "monkey",
            "-p",
            package_name,
            "-c",
            "android.intent.category.LAUNCHER",
            "1",
        ])

        self.logger.debug(f"Launched {package_name}")
        time.sleep(0.5)

    def scroll(
        self,
        start_x: int,
        start_y: int,
        direction: Direction,
        distance: int,
    ):
        self.logger.debug(f"scroll({start_x}, {start_y}, {direction}, {distance})")
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
        self.logger.debug("navigate_home()")
        self._device.keyevent("HOME")
        time.sleep(0.5)

    def navigate_back(self):
        self.logger.debug("navigate_back()")
        self._device.keyevent("BACK")
        time.sleep(0.5)

    def start_recording(self, video_path: str):
        self.logger.info(
            f"[dim]{VIDEO} Starting recording:[/] {video_path}",
            extra={"markup": True},
        )
        self._device.start_recording(video_path)
        self._is_recording = True

    def end_recording(self):
        if self._is_recording:
            self.logger.info(f"{VIDEO} Stopping recording", extra={"markup": True})
            self._device.stop_recording()
