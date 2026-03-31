# from .dasp_utils import dasp_fx_list, dasp_fx_module_dict, dasp_fx_param_keys
from .ndsp_utils import ndsp_fx_tools, pedalboard_fx_list, pedalboard_fx_function_dict, pedalboard_random_fx_module_dict, pedalboard_fx_param_keys, pedalboard_fx_param_ranges
from .utils import normalize_param, denormalize_param

def load_fx_config(fx_config_type="ndsp"):
    if fx_config_type == "ndsp":
        return {
            "fx_list": pedalboard_fx_list,
            "fx_function_dict": pedalboard_fx_function_dict,
            "fx_random_module_dict": pedalboard_random_fx_module_dict,
            "fx_param_keys": pedalboard_fx_param_keys,
            "fx_param_ranges": pedalboard_fx_param_ranges,
            "fx_tools": ndsp_fx_tools,
        }
    elif fx_config_type == "dasp":
        raise ValueError(f"Invalid fx config type: {fx_config_type}")
    elif fx_config_type == "vst":
        raise ValueError(f"Invalid fx config type: {fx_config_type}")
    else:
        raise ValueError(f"Invalid fx config type: {fx_config_type}")