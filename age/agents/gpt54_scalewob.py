import os
import time
from pathlib import Path
from typing import Any, Literal
import json
from openai import OpenAI

from ..logger import get_logger
from ..icons import ROCKET, MAGIC, CHECK, WARNING
from .base import Action, Language
from .doubao_scalewob import DoubaoScaleWoBAgent

class GPT54ScaleWoBAgent(DoubaoScaleWoBAgent):
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
        super().__init__(
            model_name=model_name,
            task_name=task_name,
            env_id=env_id,
            task_id=task_id,
            task_params=task_params,
            language=language,
            headless=headless,
            platform=platform,
            enable_notepad=enable_notepad,
            enable_tracking=enable_tracking,
            tracking_output_dir=tracking_output_dir,
            max_steps=max_steps,
            use_vertex_api=use_vertex_api,
        )
        self.name = "gpt54"
        self.logger = get_logger(__name__)
        
        self.previous_response_id = None
        self.next_input = None
        
        self.api_key = os.environ.get("GENERAL_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        self.base_url = os.environ.get("GPT54_BASE_URL")
        
        if not self.api_key:
            self.logger.warning("API Key is not set. Responses API requests will likely fail.")
            
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        self.openai_client = OpenAI(**client_kwargs)
        self.task_status = False
        from openai import AzureOpenAI
        self.openai_client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint="https://yjzl-openai-yuanchun5.openai.azure.com",
            api_version="2025-04-01-preview"
        )
        self._setup_tools()

    def _normalize_coordinates(self, x: int, y: int) -> tuple[int, int]:
        """Override to skip normalization, since GPT outputs absolute coordinates."""
        return int(x), int(y)

    def _setup_tools(self):
        self.tools = [
            {
                "type": "function",
                "name": "click",
                "description": "Click or tap on the screen at a specific coordinate.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "The x-coordinate"},
                        "y": {"type": "integer", "description": "The y-coordinate"}
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False
                }
            },
            {
                "type": "function",
                "name": "long_press",
                "description": "Long press on the screen at a specific coordinate.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "The x-coordinate"},
                        "y": {"type": "integer", "description": "The y-coordinate"}
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False
                }
            },
            {
                "type": "function",
                "name": "type",
                "description": "Type text into the currently focused text field.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The text to type."}
                    },
                    "required": ["content"],
                    "additionalProperties": False
                }
            },
            {
                "type": "function",
                "name": "clear_text",
                "description": "Clear all text in the currently activated textbox.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "type": "function",
                "name": "scroll",
                "description": "Scroll the screen in a specified direction.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "Starting x coordinate for scroll"},
                        "y": {"type": "integer", "description": "Starting y coordinate for scroll"},
                        "direction": {
                            "type": "string", 
                            "enum": ["up", "down", "left", "right"],
                            "description": "The direction to scroll"
                        }
                    },
                    "required": ["x", "y", "direction"],
                    "additionalProperties": False
                }
            },
            {
                "type": "function",
                "name": "drag",
                "description": "Drag from one point to another.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_x": {"type": "integer"},
                        "start_y": {"type": "integer"},
                        "end_x": {"type": "integer"},
                        "end_y": {"type": "integer"}
                    },
                    "required": ["start_x", "start_y", "end_x", "end_y"],
                    "additionalProperties": False
                }
            },
            {
                "type": "function",
                "name": "navigate_back",
                "description": "Go back to the previous screen or close the keyboard.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "type": "function",
                "name": "press_enter",
                "description": "Press the Enter key.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "type": "function",
                "name": "wait",
                "description": "Wait for a few seconds.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seconds": {"type": "number", "description": "Seconds to wait."}
                    },
                    "required": ["seconds"],
                    "additionalProperties": False
                }
            },
            {
                "type": "function",
                "name": "finished",
                "description": "Submit the task when it is completed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "A completion message or extracted information."}
                    },
                    "required": ["content"],
                    "additionalProperties": False
                }
            }
        ]

    def _build_instructions(self) -> str:
        return (
            "You are an AI agent controlling an Android-based mobile application to complete a specific task.\n"
            f"Task: {self.task_name}\n"
            "Use the provided tools to interact with the device. If you want visual context, ask for a screenshot (but you will receive a new screenshot automatically after every action).\n"
            "Only stop and call 'finished' once the task is completed.\n"
        )

    def _map_function_call_to_action(self, function_call: dict) -> Action:
        name = function_call.get("name", "")
        try:
            args = json.loads(function_call.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
            
        action_type = name
        params = {}
        
        if name in ["click", "long_press", "scroll"]:
            x = args.get("x", self.window_size[0] // 2)
            y = args.get("y", self.window_size[1] // 2)
            params["point"] = [x, y]
            if name == "scroll":
                params["direction"] = args.get("direction", "down")
        elif name == "drag":
            params["start_point"] = [args.get("start_x", 0), args.get("start_y", 0)]
            params["end_point"] = [args.get("end_x", 0), args.get("end_y", 0)]
        elif name == "type":
            params["content"] = args.get("content", "")
        elif name == "wait":
            params["seconds"] = args.get("seconds", 1.0)
        elif name == "finished":
            params["content"] = args.get("content", "")
            
        # fallback for parameter-less
        if name == "navigate_back":
            # the framework handles this if we call device directly, 
            # wait, doubao doesn't have "navigate_back" in its _execute method directly?
            # actually let's map navigate_back to something or execute it here.
            # Base action space has it in prompt, but maybe _execute handles it as press_back?
            # Let's map it to something if it's not supported.
            pass
            
        return {"action_type": action_type, "params": params, "thought": f"Calling {name}"}

    def predict(self) -> Action:
        # Check step limit
        forced_action = self._check_step_limit()
        if forced_action:
            self.logger.warning(
                f"[bold yellow]{WARNING} Maximum steps reached ({self.max_steps})[/]\n"
                f"[dim]Forcing task completion[/]\n",
                extra={"markup": True},
            )
            duration, status = self.execute(forced_action)
            if self.tracker:
                screenshot = self.device.capture_screenshot("B64")
                self._record_action(forced_action, screenshot, duration)
                result = self._finalize_tracking(result=status)
                if result:
                    self.logger.debug(f"Generated {result[0]} and {result[1]}")
                self.task_status = False
            return forced_action

        self.logger.debug("Capturing screenshot...")
        screenshot = self.device.capture_screenshot("B64")
        screenshot_data_url = f"data:image/png;base64,{screenshot}"
        screenshot_for_tracker = screenshot

        if self.next_input is None:
            # First turn
            self.next_input = [
                {
                    "content": [
                        {"type": "input_text", "text": f"Task: {self.task_name}"},
                        {"detail": "original", "image_url": screenshot_data_url, "type": "input_image"}
                    ],
                    "role": "user"
                }
            ]
        else:
            # Add the new screenshot to the output of the previous function call
            for item in self.next_input:
                if item["type"] == "function_call_output":
                    item["output"].append({
                        "type": "input_image",
                        "image_url": screenshot_data_url,
                        "detail": "original"
                    })

        kwargs = {
            "model": self.model_name,
            "instructions": self._build_instructions(),
            "input": self.next_input,
            "parallel_tool_calls": False,
            "reasoning": {"effort": "low"},
            "tools": self.tools,
        }

        if self.previous_response_id:
            kwargs["previous_response_id"] = self.previous_response_id

        self.logger.info(f"[bold blue]{MAGIC} Requesting Responses API[/] ({self.model_name})", extra={"markup": True})
        start_time = time.time()

        try:
            response = self.openai_client.responses.create(**kwargs)
            data = response.model_dump()
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Responses API request failed: {e}")
            action = {"action_type": "finished", "params": {"content": error_msg}, "thought": "API Error"}
            if self.tracker:
                self._record_action(action, screenshot, time.time() - start_time)
                self._finalize_tracking(result=False)
            return action

        self.previous_response_id = data.get("id")
        outputs = data.get("output", [])

        # Find the first function call
        function_call = next((item for item in outputs if item.get("type") == "function_call"), None)

        if not function_call:
            # Task finished or model responded with text only
            final_message = ""
            for item in outputs:
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            final_message += c.get("text", "") + "\n"

            action = {"action_type": "finished", "params": {"content": final_message.strip()}, "thought": "Task complete"}
            duration, status = self.execute(action)
            self.task_status = status
            if self.tracker:
                self._record_action(action, screenshot, duration)
                self._finalize_tracking(result=status)
            return action

        # Map function call to action
        action = self._map_function_call_to_action(function_call)

        # Setup next_input so we can append screenshot in the next predict() call
        self.next_input = [
            {
                "type": "function_call_output",
                "call_id": function_call.get("call_id"),
                "output": [
                    {"type": "input_text", "text": "Action executed. See the updated screenshot."}
                ]
            }
        ]

        # In DoubaoScaleWoBAgent, _execute expects `navigate_back` logic or we do it here.
        # Actually `DoubaoAgent._execute` doesn't have navigate_back.
        # So let's handle navigate_back here explicitly if it's called.
        if action["action_type"] == "navigate_back":
            self.device.navigate_back()
            duration = 1.0
            status = True
        else:
            duration, status = self.execute(action)

        if self.tracker:
            self._record_action(action, screenshot_for_tracker, duration)
            if action["action_type"] == "finished":
                self.task_status = status
                result = self._finalize_tracking(result=status)
                if result:
                    self.logger.debug(f"Generated {result[0]} and {result[1]}")

        return action
