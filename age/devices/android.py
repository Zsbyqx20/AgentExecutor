import base64
import re
import time
from difflib import get_close_matches
from io import BytesIO
from typing import Literal, overload

from adbutils import adb
from PIL.Image import Image

from age.devices.base import Device, Direction
from age.icons import KEYBOARD, PHONE, TIMES, VIDEO, WARNING
from age.logger import get_logger
from age.video_recorder import VideoRecorder


class AndroidDevice(Device):
    def __init__(self, device_id: str | None = None) -> None:
        super().__init__()
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

        self._original_ime: str | None = None  # Track original IME for restoration

        self._set_ime()
        self._is_recording = False
        self._video_recorder: VideoRecorder | None = None

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

    def _parse_app_mapping(self, raw_output: str) -> tuple[dict, dict, dict]:
        """Parse content provider output into app mappings.

        Args:
            raw_output: Raw output from content query command

        Returns:
            Tuple of (name_to_package, name_to_package_lower, package_to_name)
        """
        name_to_package = {}
        name_to_package_lower = {}
        package_to_name = {}

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
                package_to_name[package_name] = display_name

        self.logger.debug(f"Parsed {len(package_to_name)} apps from content provider")
        return name_to_package, name_to_package_lower, package_to_name

    def _set_ime(self):
        target_ime = "com.android.adbkeyboard/.AdbIME"
        ime = self._device.shell(["settings", "get", "secure", "default_input_method"])
        self.logger.debug(f"Current IME: {ime}")

        # Save original IME before switching
        if ime and ime.strip():
            self._original_ime = ime.strip()
            self.logger.debug(f"Saved original IME: {self._original_ime}")

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

    def restore_ime(self) -> None:
        """Restore the original IME that was active before ADB keyboard was set.

        This method is idempotent - it can be called multiple times safely.
        If restoration fails, an error is logged but no exception is raised.
        """
        # Check if we have an original IME to restore
        if not self._original_ime:
            self.logger.debug(
                f"[dim]{WARNING} No original IME to restore[/]",
                extra={"markup": True},
            )
            return

        # Check current IME to avoid unnecessary changes
        current_ime = self._device.shell([
            "settings",
            "get",
            "secure",
            "default_input_method",
        ])

        if current_ime == self._original_ime:
            self.logger.debug(f"Original IME already active: {self._original_ime}")
            return

        # Attempt to restore the original IME
        try:
            self._device.shell(["ime", "set", self._original_ime])
            self.logger.info(
                f"{KEYBOARD} Restored original IME: {self._original_ime}",
                extra={"markup": True},
            )
            # Mark as restored by clearing the saved value
            self._original_ime = None
        except Exception as e:
            # Log but don't raise - this is cleanup code
            self.logger.error(
                f"[bold red]{TIMES} Failed to restore IME:[/] {e}",
                extra={"markup": True},
            )

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
    def size(self) -> tuple[int, int]:
        return self._device.window_size()

    def press(self, x: int, y: int) -> None:
        self.logger.debug(f"press({x}, {y})")
        self._device.click(x, y)
        time.sleep(1)

    def long_press(self, x: int, y: int, duration: float):
        self.logger.debug(f"long_press({x}, {y}, {duration}s)")
        self._device.swipe(x, y, x, y, duration)
        time.sleep(1)

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
        time.sleep(1)

    def clear_text(self):
        """Clear all text in currently activated textbox."""
        self.logger.debug("clear_text()")
        self._device.shell(["am", "broadcast", "-a", "ADB_CLEAR_TEXT"])
        time.sleep(1)

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float):
        self.logger.debug(
            f"drag({start_x}, {start_y}) → ({end_x}, {end_y}), {duration}s"
        )
        self._device.drag(start_x, start_y, end_x, end_y, duration)
        time.sleep(1)

    def list_apps(self) -> dict[str, str]:
        """Return mapping of all available launchable apps.

        Returns:
            Dict mapping package names to display names
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
            if app_identifier in self._all_apps:
                return app_identifier
            return None

        # Try exact display name match
        if app_identifier in self._app_mapping:
            return self._app_mapping[app_identifier]

        # Try case-insensitive match
        lower_key = app_identifier.lower()
        if lower_key in self._app_mapping_lower:
            # Find the actual display name for logging
            display_name = self._all_apps.get(self._app_mapping_lower[lower_key])
            if display_name:
                self.logger.debug(
                    f"Using case-insensitive match: '{app_identifier}' → '{display_name}'"
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
        all_names = list(self._all_apps.values())
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
        time.sleep(1)

    def get_current_app(self) -> str:
        """Get the current app package name (or display name if available).

        Returns:
            Package/Display name of the current app
        """
        package = self._device.app_current().package
        return self._all_apps.get(package, package)

    def kill_current_app(self):
        """Kill the current app using am force-stop."""
        package = self._device.app_current().package
        self.logger.debug(f'kill_current_app("{package}")')
        self._device.shell(["am", "force-stop", package])
        time.sleep(1)

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
        time.sleep(1)

    def navigate_back(self):
        self.logger.debug("navigate_back()")
        self._device.keyevent("BACK")
        time.sleep(1)

    def start_recording(self, video_path: str | None = None) -> bool:
        """Start video recording using scrcpy.

        Args:
            video_path: Optional path for output video file. If not provided,
                        uses a default path in current directory.

        Returns:
            True if recording started successfully, False otherwise.
        """
        output_path = video_path or "recording.mp4"

        self.logger.info(
            f"[dim]{VIDEO} Starting recording:[/] {output_path}",
            extra={"markup": True},
        )

        self._video_recorder = VideoRecorder(
            output_path=output_path,
            device_serial=self._device_id,
        )

        if self._video_recorder.start_recording():
            self._is_recording = True
            return True
        else:
            if self._video_recorder.error_message:
                self.logger.warning(
                    f"[dim]{WARNING} Recording unavailable:[/] {self._video_recorder.error_message}",
                    extra={"markup": True},
                )
            return False

    def end_recording(self) -> None:
        """Stop video recording using scrcpy."""
        if self._is_recording and self._video_recorder:
            self.logger.info(f"{VIDEO} Stopping recording", extra={"markup": True})
            self._video_recorder.stop_recording()
            self._is_recording = False

    @property
    def recording_error(self) -> str | None:
        """Return the last recording error message, if any."""
        return self._video_recorder.error_message if self._video_recorder else None

    def __del__(self) -> None:
        """Destructor to restore original IME when Device object is destroyed."""
        try:
            if hasattr(self, "_original_ime") and self._original_ime:
                self.restore_ime()
        except Exception as e:
            # Destructor should never raise - log if possible, otherwise fail silently
            if hasattr(self, "logger"):
                self.logger.error(
                    f"[bold red]{TIMES} Error in Device destructor:[/] {e}",
                    extra={"markup": True},
                )

    def __enter__(self):
        """Context manager entry - returns the Device instance."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit - restores the original IME.

        This method is guaranteed to be called even when exceptions occur,
        including KeyboardInterrupt (Ctrl+C).

        Args:
            _exc_type: Exception type if an exception was raised (unused)
            _exc_val: Exception value if an exception was raised (unused)
            _exc_tb: Exception traceback if an exception was raised (unused)

        Returns:
            False to propagate any exceptions (don't suppress them)
        """
        self.restore_ime()
        return False  # Don't suppress exceptions
