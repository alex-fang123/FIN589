import warnings
# Assuming characteristics_names_map provides a function returning a dict
try:
    from .characteristics_names_map import characteristics_names_map
except ImportError:
    # Fallback if running directly
    try:
        from characteristics_names_map import characteristics_names_map
    except ImportError:
        raise ImportError("Could not import characteristics_names_map function.")

def _read_desc(desc_map, key):
    """
    Helper function to safely read description from the map.
    Returns the key itself and issues a warning if the key is not found.
    """
    if key in desc_map:
        return desc_map[key]
    else:
        warnings.warn(f"No description available for key [{key}] in map.")
        return key # Return the key itself if not found

def anomdescr(anom):
    """
    Translates anomaly codes into descriptive names using a predefined map.

    Args:
        anom (list): A list of anomaly code strings.

    Returns:
        list: A list of formatted anomaly descriptions.
    """
    if not isinstance(anom, list) or not anom:
        raise ValueError("Input 'anom' must be a non-empty list.")

    descriptions = []
    try:
        desc_map = characteristics_names_map() # Get the description map
        if not isinstance(desc_map, dict):
             raise TypeError("characteristics_names_map() must return a dictionary.")
    except Exception as e:
        raise RuntimeError(f"Failed to load description map from characteristics_names_map: {e}") from e

    for a in anom:
        n = a # Default to original if no specific rule matches
        s_desc = a # Default description part is the original string

        if len(a) >= 3:
            prefix = a[:3]
            s = a[3:] # Potential key for the map

            if prefix == 'rme':
                n = 'Market'
            elif prefix == 're_':
                n = _read_desc(desc_map, s)
            elif prefix == 'r2_':
                s_desc = _read_desc(desc_map, s)
                n = f"{s_desc}$^2$"
            elif prefix == 'r3_':
                s_desc = _read_desc(desc_map, s)
                n = f"{s_desc}$^3$"
            elif prefix == 'rX_':
                xsep = '__'
                idx = s.find(xsep)
                if idx == -1:
                    xsep = '_' # OLD convention
                    idx = s.find(xsep)

                if idx != -1:
                    part1_key = s[:idx]
                    part2_key = s[idx+len(xsep):]
                    part1_desc = _read_desc(desc_map, part1_key)
                    part2_desc = _read_desc(desc_map, part2_key)
                    n = f"{part1_desc}$\\times${part2_desc}"
                else:
                    # If separator not found, try reading desc for the whole 's' part
                    n = _read_desc(desc_map, s) # Or default back to 'a'? Let's use s.
            else: # Corresponds to 'otherwise' in MATLAB outer switch
                if len(a) >= 2 and a[:2] == 'r_':
                     s_key = a[2:] # Key is part after 'r_'
                else:
                     s_key = a # Key is the whole string
                n = _read_desc(desc_map, s_key)
        else:
             # If string is too short for prefix logic, try reading desc for whole string
             n = _read_desc(desc_map, a)


        # Replace '_' with '\_' for LaTeX compatibility (applied to the final string 'n')
        n = n.replace('_', r'\_') # Use raw string for backslash

        descriptions.append(n)

    return descriptions

# Example Usage (optional, requires characteristics_names_map.py)
# if __name__ == '__main__':
#     # Assuming characteristics_names_map returns something like:
#     # {'size': 'Size', 'value': 'Value', 'mom': 'Momentum', 'beta': 'Beta',
#     #  'prof': 'Profitability', 'inv': 'Investment', 'I_DUM_1': 'Industry 1'}
#
#     test_anomalies = [
#         'rme', 're_size', 'r2_value', 'r3_mom',
#         'rX_size__value', 'rX_prof_inv', 'r_beta', 'other_code',
#         're_I_DUM_1', 'rX_bm_I_DUM_2' # Assume bm and I_DUM_2 are not in map
#      ]
#     try:
#         formatted_names = anomdescr(test_anomalies)
#         print("Original -> Formatted Description")
#         for original, formatted in zip(test_anomalies, formatted_names):
#             print(f"{original} -> {formatted}")
#     except Exception as e:
#         print(f"Error during example: {e}")
#
#     # Expected output (assuming map above and bm/I_DUM_2 missing):
#     # Original -> Formatted Description
#     # rme -> Market
#     # re_size -> Size
#     # r2_value -> Value$^2$
#     # r3_mom -> Momentum$^3$
#     # rX_size__value -> Size$\times$Value
#     # rX_prof_inv -> Profitability$\times$Investment
#     # r_beta -> Beta
#     # other_code -> other\_code  (Warning issued)
#     # re_I_DUM_1 -> Industry 1
#     # rX_bm_I_DUM_2 -> bm$\times$I\_DUM\_2 (Warnings issued for bm and I_DUM_2)
