"""Video recording using scrcpy for screen capture."""

import subprocess
import time
from pathlib import Path
from typing import Optional


class VideoRecorder:
    """Manages scrcpy subprocess for video recording."""

    def __init__(self, output_path: str, device_serial: Optional[str] = None):
        """
        Initialize video recorder.

        Args:
            output_path: Path to output video file
            device_serial: Optional device serial for multiple devices
        """
        self.output_path = output_path
        self.device_serial = device_serial
        self.process: Optional[subprocess.Popen] = None
        self.is_recording = False
        self.error_message: Optional[str] = None
        self.recording_start_time: Optional[float] = None

    def is_available(self) -> bool:
        """
        Check if scrcpy is available on the system.

        Returns:
            True if scrcpy is installed, False otherwise
        """
        try:
            result = subprocess.run(
                ["which", "scrcpy"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def start_recording(self) -> bool:
        """
        Start scrcpy recording subprocess.

        Returns:
            True if recording started successfully, False on error
        """
        # Check if scrcpy is available
        if not self.is_available():
            self.error_message = (
                "scrcpy not found. Please install scrcpy to enable video recording."
            )
            return False

        # Ensure output directory exists
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build scrcpy command
        scrcpy_cmd = [
            "scrcpy",
            "--record",
            str(output_path),
            "--no-playback",
            "--no-window",
            "--no-audio",
            "--no-control",
        ]

        # Add device serial if specified
        if self.device_serial:
            scrcpy_cmd.extend(["-s", self.device_serial])

        try:
            # Start scrcpy subprocess
            self.process = subprocess.Popen(
                scrcpy_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Capture the actual recording start time
            self.recording_start_time = time.time()
            self.is_recording = True
            return True

        except FileNotFoundError:
            self.error_message = "scrcpy command not found in PATH"
            return False
        except Exception as e:
            self.error_message = f"Failed to start scrcpy: {str(e)}"
            return False

    def stop_recording(self) -> None:
        """Stop scrcpy recording with graceful shutdown."""
        if not self.process:
            return

        self.is_recording = False

        try:
            # Graceful shutdown: terminate → wait → kill
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

            # Capture any error output
            if self.process.stderr:
                stderr_output = self.process.stderr.read()
                if stderr_output and self.process.returncode != 0:
                    self.error_message = f"scrcpy error: {stderr_output.strip()}"

        except Exception as e:
            self.error_message = f"Error stopping scrcpy: {str(e)}"
        finally:
            self.process = None
