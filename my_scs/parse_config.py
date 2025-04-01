# Assuming check_params is available in the same directory or package
try:
    from .check_params import check_params
except ImportError:
    # Fallback if running script directly or check_params is elsewhere
    from check_params import check_params
import warnings

def parse_config(cfg, CFG):
    """
    Parses and processes configuration options using defaults.

    Mimics the MATLAB parse_config function, assuming dictionary inputs.

    Args:
        cfg (dict or None): The configuration dictionary provided by the user.
                            If None or empty, it's treated as an empty dict.
        CFG (dict): A dictionary containing the default configuration values.

    Returns:
        tuple: A tuple containing:
            - dict: The final configuration dictionary with defaults applied.
            - list: Keys from CFG that were explicitly provided in cfg.
            - list: Keys for which default values from CFG were used.

    Raises:
        TypeError: If cfg is provided but is not a dictionary.
    """
    # Ensure cfg is a dict, initialize if None
    if cfg is None:
        cfg = {}
    elif not isinstance(cfg, dict):
        # The MATLAB version could parse a string, but we don't have str2cfg.
        # In Python, we expect a dictionary for configuration.
        raise TypeError(f"Input 'cfg' must be a dictionary or None, got {type(cfg)}")

    # Ensure CFG is a dict
    if not isinstance(CFG, dict):
         raise TypeError(f"Input 'CFG' (defaults) must be a dictionary, got {type(CFG)}")

    # Apply defaults using check_params
    # check_params returns: result_s, overridden, defaulted
    final_cfg, overridden, defaulted = check_params(cfg, CFG, required=None) # No required fields checked here

    return final_cfg, overridden, defaulted

# Example Usage (optional, for testing)
# if __name__ == '__main__':
#     defaults = {'param_a': 10, 'param_b': 'hello', 'param_c': True}

#     # --- Test Case 1: User provides some params ---
#     user_config1 = {'param_a': 5, 'param_d': 100}
#     print("--- Test Case 1 ---")
#     final1, over1, default1 = parse_config(user_config1, defaults)
#     print("User Config:", user_config1)
#     print("Defaults:", defaults)
#     print("Final Config:", final1)
#     print("Overridden:", over1)
#     print("Defaulted:", default1)
#     # Expected: final1 = {'param_a': 5, 'param_d': 100, 'param_b': 'hello', 'param_c': True}
#     # Expected: over1 = ['param_a'] (or potentially ['param_a', 'param_d'] depending on check_params logic)
#     # Expected: default1 = ['param_b', 'param_c']

#     # --- Test Case 2: User provides None ---
#     user_config2 = None
#     print("\n--- Test Case 2 ---")
#     final2, over2, default2 = parse_config(user_config2, defaults)
#     print("User Config:", user_config2)
#     print("Defaults:", defaults)
#     print("Final Config:", final2)
#     print("Overridden:", over2)
#     print("Defaulted:", default2)
#     # Expected: final2 = {'param_a': 10, 'param_b': 'hello', 'param_c': True}
#     # Expected: over2 = []
#     # Expected: default2 = ['param_a', 'param_b', 'param_c']

#     # --- Test Case 3: User overrides all ---
#     user_config3 = {'param_a': 1, 'param_b': 'world', 'param_c': False}
#     print("\n--- Test Case 3 ---")
#     final3, over3, default3 = parse_config(user_config3, defaults)
#     print("User Config:", user_config3)
#     print("Defaults:", defaults)
#     print("Final Config:", final3)
#     print("Overridden:", over3)
#     print("Defaulted:", default3)
#     # Expected: final3 = {'param_a': 1, 'param_b': 'world', 'param_c': False}
#     # Expected: over3 = ['param_a', 'param_b', 'param_c']
#     # Expected: default3 = []

#     # --- Test Case 4: Invalid input type ---
#     user_config4 = "param_a=5"
#     print("\n--- Test Case 4 ---")
#     try:
#         parse_config(user_config4, defaults)
#     except TypeError as e:
#         print("Caught expected error:", e)
