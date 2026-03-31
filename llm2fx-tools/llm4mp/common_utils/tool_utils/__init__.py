from llm4mp.common_utils.fx_utils.ndsp_utils import pedalboard_fx_function_dict
import json
import re
import numpy as np
def parse_tool_calls(text):
    """Parse tool calls from model output text using regex for efficient parsing"""
    tool_calls = []
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.finditer(pattern, text, re.DOTALL)
    for match in matches:
        tool_text = match.group(1).strip()
        try:
            tool_dict = json.loads(tool_text)
            tool_calls.append(tool_dict)
        except json.JSONDecodeError:
            print(f"Failed to parse tool call: {tool_text}")
    return tool_calls
            
def apply_tool_calls(tool_calls: list, audio_data: np.ndarray):
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]
        if tool_name in pedalboard_fx_function_dict:
            current_processor = pedalboard_fx_function_dict[tool_name]
            audio_data = current_processor(x=audio_data, **tool_args)
    return audio_data