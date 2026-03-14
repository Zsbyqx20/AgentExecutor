import os
import requests
import time
from pathlib import Path
from typing import Any, Literal
import json

from ..logger import get_logger
from ..icons import ROCKET, MAGIC, CHECK, WARNING
from .base import Action, Agent, Language
from ..devices import ScaleWoBDevice

class GPT54ScaleWoBAgent(Agent):
    """Agent implementation for GPT-5.4 using the experimental Responses API."""
    def __init__(
        self,
        model_name: str,
        task_name: str,
        env_id: str,
        task_id: int,
        task_params: dict[str, Any] | None = None,
        language: Literal["CN", "EN"] = "EN",
        headless: bool = False,
        platform: Literal["mobile", "desktop"] = "mobile",
        enable_notepad: bool = False,
        enable_tracking: bool = False,
        tracking_output_dir: str | Path | None = None,
        max_steps: int | None = None,
        use_vertex_api: bool = False,
    ) -> None:
        self.device = ScaleWoBDevice(env_id, headless, platform)
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
        self.name = "gpt54"
        self.model_name = model_name
        self.task_name = task_name
        self.task_id = task_id
        self.task_params = task_params
        self.window_size = self.device.size
        self.language = Language[language.upper()]
        
        self.logger = get_logger(__name__)
        
        self.previous_response_id = None
        self.next_input = None
        self.api_key = os.environ.get("GENERAL_API_KEY", "")
        if not self.api_key:
            self.logger.warning("OPENAI_API_KEY is not set. Responses API requests will fail.")
        self.base_url = os.environ.get("GPT54_BASE_URL", "https://api.openai.com/v1/responses")
        self.task_status = False

    def _build_instructions(self) -> str:
        return (
            "You are controlling an Android-based mobile application through the built-in computer tool.\n"
            f"Task: {self.task_name}\n"
            "Use only the provided task description as the source of truth.\n"
            "Only stop requesting computer actions once the task is complete.\n"
            "Reply briefly once the task is complete."
        )

    def predict(self) -> Action:
        # Check step limit
        forced_action = self._check_step_limit()
        if forced_action:
            self.logger.warning(
                f"[bold yellow]{WARNING} Maximum steps reached ({self.max_steps})[/]\n"
                f"[dim]Forcing task completion[/]\n",
                extra={"markup": True},
            )
            # Finish tracking
            if self.tracker:
                screenshot = self.device.capture_screenshot("B64")
                self._record_action(forced_action, screenshot, 0.0)
                self._finalize_tracking(result=False)
            return forced_action

        self.logger.debug("Capturing screenshot...")
        screenshot = self.device.capture_screenshot("B64")
        screenshot_data_url = f"data:image/png;base64,{screenshot}"

        if self.next_input is None:
            # First turn
            self.next_input = [
                {
                    "content": [
                        {
                            "text": self.task_name,
                            "type": "input_text"
                        },
                        {
                            "detail": "original",
                            "image_url": screenshot_data_url,
                            "type": "input_image"
                        }
                    ],
                    "role": "user"
                }
            ]

        payload = {
            "instructions": self._build_instructions(),
            "input": self.next_input,
            "model": self.model_name,
            "parallel_tool_calls": False,
            "reasoning": {"effort": "low"},
            "tools": [{"type": "computer"}],
            "truncation": "auto"
        }
        if self.previous_response_id:
            payload["previous_response_id"] = self.previous_response_id

        self.logger.debug(f"[bold blue]{MAGIC} Requesting Responses API[/] ({self.model_name})", extra={"markup": True})
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Responses API request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                error_msg += f"\n{e.response.text}"
                self.logger.error(f"Response: {e.response.text}")
            
            action = {"action_type": "finished", "params": {"error": error_msg}, "thought": "API Error"}
            if self.tracker:
                self._record_action(action, screenshot, time.time() - start_time)
                self._finalize_tracking(result=False)
            return action

        self.previous_response_id = data.get("id")
        outputs = data.get("output", [])
        
        has_tool_calls = any(item.get("type") in ["computer_call", "function_call"] for item in outputs)
        
        if not has_tool_calls:
            # Task finished
            final_message = ""
            for item in outputs:
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            final_message += c.get("text", "") + "\n"
            
            action = {"action_type": "finished", "params": {"message": final_message.strip()}, "thought": "Task complete"}
            duration = time.time() - start_time
            self.task_status = True
            
            self.logger.info(f"[dim]{CHECK} Task marked as finished[/]", extra={"markup": True})
            if self.tracker:
                self._record_action(action, screenshot, duration)
                self._finalize_tracking(result=True)
            return action

        # Execute tool calls
        tool_outputs = []
        last_action = {"action_type": "wait", "params": {}, "thought": "Executing computer tools"}
        
        for item in outputs:
            if item.get("type") == "computer_call":
                actions = item.get("actions", [])
                for act in actions:
                    duration = self._execute_computer_action(act)
                    mapped_action = self._map_computer_action_to_action(act)
                    last_action = mapped_action
                    
                    if self.tracker:
                        current_screenshot = self.device.capture_screenshot("B64")
                        self._record_action(mapped_action, current_screenshot, duration)
                
                # Get screenshot after executing all actions in this batch
                after_screenshot = self.device.capture_screenshot("B64")
                tool_outputs.append({
                    "type": "computer_call_output",
                    "call_id": item.get("call_id"),
                    "output": {
                        "type": "computer_screenshot",
                        "image_url": f"data:image/png;base64,{after_screenshot}"
                    }
                })
            elif item.get("type") == "function_call":
                # Fallback for unexpected function calls, just echo back to satisfy API
                self.logger.warning(f"Unexpected function call: {item.get('name')}")
                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": item.get("call_id"),
                    "output": [{"type": "input_text", "text": "Function calls are not supported."}]
                })

        self.next_input = tool_outputs
        
        # Return the last action executed so the outer loop doesn't break
        return last_action

    def _execute_computer_action(self, action: dict) -> float:
        from age.devices.base import Direction
        start_time = time.perf_counter()
        
        act_type = action.get("type")
        x = action.get("x")
        y = action.get("y")
        
        try:
            if act_type == "click":
                self.device.press(int(x), int(y))
            elif act_type == "double_click":
                self.device.press(int(x), int(y))
                time.sleep(0.1)
                self.device.press(int(x), int(y))
            elif act_type == "type":
                text = action.get("text", "")
                self.device.type_text(text)
            elif act_type == "keypress":
                keys = action.get("keys", [])
                if not keys and action.get("key"):
                    keys = [action.get("key")]
                
                for key in keys:
                    k = str(key).lower()
                    if k in ["enter", "return"]:
                        self.device.press_enter()
                    elif k in ["backspace", "delete", "escape", "esc"]:
                        self.device.navigate_back()
            elif act_type == "scroll":
                delta_x = action.get("delta_x", action.get("deltaX", 0))
                delta_y = action.get("delta_y", action.get("deltaY", action.get("scroll_y", 0)))
                
                sx = int(x) if x is not None else self.window_size[0] // 2
                sy = int(y) if y is not None else self.window_size[1] // 2
                
                distance = int(abs(delta_y) if delta_y else abs(delta_x))
                if distance == 0:
                    distance = 300
                    
                if delta_y > 0:
                    self.device.scroll(sx, sy, Direction.DOWN, distance)
                elif delta_y < 0:
                    self.device.scroll(sx, sy, Direction.UP, distance)
                elif delta_x > 0:
                    self.device.scroll(sx, sy, Direction.RIGHT, distance)
                elif delta_x < 0:
                    self.device.scroll(sx, sy, Direction.LEFT, distance)
            elif act_type == "drag":
                path = action.get("path", [])
                if len(path) >= 2:
                    start_pt = path[0]
                    end_pt = path[-1]
                    self.device.drag(
                        int(start_pt.get("x", 0)), int(start_pt.get("y", 0)),
                        int(end_pt.get("x", 0)), int(end_pt.get("y", 0))
                    )
            elif act_type == "wait":
                ms = action.get("ms", action.get("duration_ms", 1000))
                time.sleep(max(0, ms) / 1000.0)
            elif act_type == "screenshot":
                pass
            else:
                self.logger.warning(f"Unsupported computer action: {act_type}")
        except Exception as e:
            self.logger.error(f"Error executing computer action {act_type}: {e}")
            
        time.sleep(0.5)
        return time.perf_counter() - start_time

    def _map_computer_action_to_action(self, action: dict) -> Action:
        act_type = action.get("type", "unknown")
        params = dict(action)
        return {"action_type": act_type, "params": params, "thought": f"Executing {act_type}"}

