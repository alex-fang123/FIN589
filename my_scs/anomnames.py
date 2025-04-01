import re

def anomnames(anom):
    """
    Translates anomaly codes into descriptive names, often formatted for LaTeX.

    Args:
        anom (list): A list of anomaly code strings.

    Returns:
        list: A list of formatted, lowercase anomaly names.
    """
    names = []
    for a in anom:
        n = a # Default to original if no specific rule matches
        if len(a) >= 3:
            prefix = a[:3]
            s = a[3:]

            if prefix == 'rme':
                n = 'market'
            elif prefix == 're_':
                n = s
            elif prefix == 'r2_':
                n = f"{s}$^2$"
            elif prefix == 'r3_':
                n = f"{s}$^3$"
            elif prefix == 'rX_':
                xsep = '__'
                idx = s.find(xsep)
                if idx == -1:
                    xsep = '_' # OLD convention
                    idx = s.find(xsep)

                if idx != -1:
                    # Ensure indices are valid before slicing
                    part1 = s[:idx]
                    part2 = s[idx+len(xsep):]
                    n = f"{part1}$\\times${part2}"
                # else: keep n = a (original string) if separator not found
            # This 'else' corresponds to the 'otherwise' in MATLAB's outer switch
            else:
                # This 'if' corresponds to the nested 'switch prefix(1:2)'
                if len(a) >= 2 and a[:2] == 'r_':
                     # Corresponds to a(3:end) in MATLAB (index 2 in Python)
                     n = a[2:]
                # else: keep n = a (original string)

        # Replace 'I_DUM' with 'ind'
        # This replacement happens regardless of the prefix matching
        n = n.replace('I_DUM', 'ind')

        # Replace '_' with '\_' for LaTeX compatibility
        # This replacement also happens regardless of the prefix matching
        n = n.replace('_', r'\_') # Use raw string for backslash

        names.append(n.lower())

    return names

# Example Usage (optional, for testing)
# if __name__ == '__main__':
#     test_anomalies = [
#         'rme', 're_size', 'r2_value', 'r3_mom',
#         'rX_size__value', 'rX_prof_inv', 'r_beta', 'other_code',
#         're_I_DUM_1', 'rX_bm_I_DUM_2'
#      ]
#     formatted_names = anomnames(test_anomalies)
#     print("Original -> Formatted")
#     for original, formatted in zip(test_anomalies, formatted_names):
#         print(f"{original} -> {formatted}")
#
#     # Expected output:
#     # Original -> Formatted
#     # rme -> market
#     # re_size -> size
#     # r2_value -> value$^2$
#     # r3_mom -> mom$^3$
#     # rX_size__value -> size$\times$value
#     # rX_prof_inv -> prof$\times$inv
#     # r_beta -> beta
#     # other_code -> other\_code
#     # re_I_DUM_1 -> ind\_1
#     # rX_bm_I_DUM_2 -> bm$\times$ind\_2
