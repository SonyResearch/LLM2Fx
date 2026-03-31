import re
import json
import ast

def parsing_multiple_tool_response(text: str):
    """
    Parse the text to a list of tool calls.
    Args:
        text: The text to parse.
    Returns:
        list_of_tool_calls: A list of tool calls.
    """
    list_of_tool_calls = []
    # Match <tool_call> ... </tool_call> blocks, capturing the inside
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    for match in pattern.finditer(text):
        block = match.group(1).strip()
        # Try to parse as JSON first
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                list_of_tool_calls.append(obj)
                continue
        except Exception:
            pass
        # Try to parse as Python dict (single quotes, etc)
        try:
            obj = ast.literal_eval(block)
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                list_of_tool_calls.append(obj)
                continue
        except Exception:
            pass
        # Try to extract the first {...} block and parse that
        start_idx = block.find("{")
        end_idx = block.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = block[start_idx:end_idx + 1]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                    list_of_tool_calls.append(obj)
                    continue
            except Exception:
                pass
            try:
                obj = ast.literal_eval(candidate)
                if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                    list_of_tool_calls.append(obj)
                    continue
            except Exception:
                pass
    return list_of_tool_calls

# # Test string for verification
# test_str = "<tool_call>\n{'name': 'stereo_widener', 'arguments': {'width': 1.4}}\n</tool_call><tool_call>\n{'name': 'panner', 'arguments': {'pan': 0.0}}\n</tool_call><tool_call>\n{'name': 'gain', 'arguments': {'gain_db': 3.0}}\n</tool_call><tool_call>\n{'name': 'equalizer', 'arguments': {'low_gain_db': -1.0, 'low_cutoff_freq': 70.0, 'low_q_factor': 2.0, 'mid_gain_db': -1.0, 'mid_cutoff_freq': 850.0, 'mid_q_factor': 2.0, 'high_gain_db': -2.0, 'high_cutoff_freq': 8000.0, 'high_q_factor': 1.0}}\n</tool_call><tool_call>\n{'name': 'reverb', 'arguments': {'room_size': 0.35, 'damping': 0.6, 'width': 0.55, 'mix_ratio': 0.3}}\n</tool_call><tool_call>\n{'name': 'distortion', 'arguments': {'drive_db': 2.0}}\n</tool_call>\nHere are the parameters for the requested audio effects, maintaining the specified order: \n1. Stereo Widener: Set to a width of 1.4 to broaden the stereo image.\n2. Panner: Centered at 0.0 for a balanced stereo placement.\n3. Gain: Increased by 3.0 dB for a subtle volume boost.\n4. Equalizer: Applied with a low cut at 70 Hz (-1.0 dB, Q=2.0), a mid cut at 850 Hz (-1.0 dB, Q=2.0), and a high cut at 8000 Hz (-2.0 dB, Q=1.0) to shape the tonal balance.\n5. Reverb: Set with a room size of 0.35, damping of 0.6, width of 0.55, and a mix ratio of 0.3 for spatial depth.\n6. Distortion: Applied with a drive of 2.0 dB for harmonic enhancement."
# if __name__ == "__main__":
#     print(parsing_multiple_tool_response(test_str))