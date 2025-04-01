def check_params(s, defaults=None, required=None):
    """
    Verifies a parameter dictionary, sets defaults, and checks required keys.

    Mimics the behavior of the MATLAB check_params function.

    Args:
        s (dict): The parameters dictionary to check. If None, treated as empty.
        defaults (dict, optional): Default values for optional parameters.
                                   Defaults to None or an empty dict.
        required (list, optional): List of required parameter keys (strings).
                                   Defaults to None or an empty list.

    Returns:
        tuple: A tuple containing:
            - dict: The parameters dictionary with defaults applied.
            - list: Keys from 'defaults' that were explicitly provided in 's'
                    (thus overriding the default).
            - list: Keys for which default values from 'defaults' were used
                    because they were missing in 's'.

    Raises:
        ValueError: If a required key is missing from 's'.
    """
    if s is None:
        s = {}
    if defaults is None:
        defaults = {}
    if required is None:
        required = []

    result_s = s.copy()  # Start with the user-provided parameters
    overridden = []
    defaulted = []

    # 1. Check required fields
    for req_key in required:
        if req_key not in s:
            # In Python, it's more idiomatic to raise an error for required params
            raise ValueError(f"Required parameter '{req_key}' is missing.")

    # 2. Iterate through defaults to apply them or mark as overridden
    if defaults:
        for key, default_value in defaults.items():
            if key not in s:
                # Apply default if key is missing in s
                result_s[key] = default_value
                defaulted.append(key)
            else:
                # Mark as overridden if key is present in s (even if value is same as default)
                overridden.append(key)

    # Note: Keys present in 's' but not in 'defaults' are kept in 'result_s'
    # but are not included in the 'overridden' list, matching the MATLAB script's
    # apparent logic which focuses on the interaction with the 'defaults' dict.

    return result_s, overridden, defaulted

# Example Usage (optional, for testing)
# if __name__ == '__main__':
#     user_params = {'a': 1, 'c': 30}
#     default_params = {'a': 10, 'b': 20, 'c': 300, 'd': 40}
#     required_params = ['a']
#
#     try:
#         final_params, overridden_keys, defaulted_keys = check_params(
#             user_params, default_params, required_params
#         )
#         print("Final Params:", final_params)
#         print("Overridden Keys:", overridden_keys)
#         print("Defaulted Keys:", defaulted_keys)
#
#         # Expected Output:
#         # Final Params: {'a': 1, 'c': 30, 'b': 20, 'd': 40}
#         # Overridden Keys: ['a', 'c']
#         # Defaulted Keys: ['b', 'd']
#
#     except ValueError as e:
#         print("Error:", e)
#
#     # Test missing required
#     user_params_missing = {'c': 30}
#     try:
#         check_params(user_params_missing, default_params, required_params)
#     except ValueError as e:
#         print("\nTesting missing required:")
#         print("Error:", e)
#         # Expected Output:
#         # Error: Required parameter 'a' is missing.
