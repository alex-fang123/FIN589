import numpy as np
import warnings

def regcov(r):
    """
    Calculates a regularized covariance matrix.

    First computes the sample covariance matrix, then applies shrinkage towards
    a scaled identity matrix based on the trace of the sample covariance.
    The shrinkage intensity 'a' depends on the number of samples (T) and
    variables (n).

    Args:
        r (np.ndarray): Input data matrix (T x n), where T is samples and n is variables.

    Returns:
        np.ndarray: The regularized covariance matrix (n x n). Returns NaN matrix
                    if T < 2.
    """
    r = np.asarray(r)
    if r.ndim != 2:
        raise ValueError(f"Input 'r' must be a 2D array (T x n), got shape {r.shape}")

    T, n = r.shape

    if T < 2:
        warnings.warn(f"Need at least 2 samples (T={T}) to compute covariance. Returning NaN matrix.")
        return np.full((n, n), np.nan)
    if n == 0:
        return np.empty((0, 0)) # Return empty matrix if no variables

    # Calculate sample covariance matrix
    # rowvar=False means columns are variables. ddof=1 for sample covariance (divides by T-1).
    # MATLAB's cov default might divide by T or T-1 depending on args, assuming T-1 here.
    sample_cov = np.cov(r, rowvar=False, ddof=1)

    # Handle case where cov might return scalar if n=1
    if n == 1:
        sample_cov = np.array([[sample_cov.item()]]) # Ensure it's 2D

    # Calculate shrinkage intensity 'a'
    # Avoid division by zero if n + T = 0 (though unlikely with T>=2)
    if n + T == 0:
        a = 0.0 # Or handle as error? Let's default to 0 shrinkage.
    else:
        a = n / (n + T)

    # Calculate trace of the sample covariance
    trace_X = np.trace(sample_cov)

    # Calculate the scaled identity matrix target
    # Avoid division by zero if n=0 (already handled) or trace is zero
    if n > 0 and abs(trace_X) > np.finfo(float).eps:
         scaled_identity = (trace_X / n) * np.identity(n)
    elif n > 0:
         # If trace is zero (e.g., all data is zero), target is zero matrix
         scaled_identity = np.zeros((n, n))
    else: # n == 0 case
         scaled_identity = np.empty((0, 0))


    # Apply regularization: a * Target + (1 - a) * SampleCov
    regularized_cov = a * scaled_identity + (1 - a) * sample_cov

    return regularized_cov

# Example Usage (optional, for testing)
# if __name__ == '__main__':
#     # Dummy data
#     T = 50
#     n = 10
#     np.random.seed(0)
#     dummy_r = np.random.randn(T, n) * 2 + 5

#     # --- Test Case 1: Standard calculation ---
#     print("--- Test Case 1: Standard ---")
#     reg_cov_matrix = regcov(dummy_r)
#     print("Shape of regularized covariance:", reg_cov_matrix.shape)
#     # print("Regularized Covariance Matrix:\n", np.round(reg_cov_matrix, 3))
#     sample_cov_matrix = np.cov(dummy_r, rowvar=False, ddof=1)
#     print("Shape of sample covariance:", sample_cov_matrix.shape)
#     # print("Sample Covariance Matrix:\n", np.round(sample_cov_matrix, 3))
#     # Check if diagonal elements are generally larger than off-diagonal after regularization
#     print("Diagonal dominance check (diag > mean abs off-diag):",
#           np.all(np.diag(reg_cov_matrix) > np.mean(np.abs(reg_cov_matrix - np.diag(np.diag(reg_cov_matrix))))))


#     # --- Test Case 2: Low T ---
#     print("\n--- Test Case 2: Low T (T=10) ---")
#     dummy_r_low_T = dummy_r[:10, :]
#     reg_cov_low_T = regcov(dummy_r_low_T)
#     print("Shape:", reg_cov_low_T.shape)

#     # --- Test Case 3: Very Low T (T=1) ---
#     print("\n--- Test Case 3: Very Low T (T=1) ---")
#     dummy_r_T1 = dummy_r[:1, :]
#     reg_cov_T1 = regcov(dummy_r_T1) # Should return NaNs and warning
#     print("Shape:", reg_cov_T1.shape)
#     print("Is NaN matrix?", np.all(np.isnan(reg_cov_T1)))

#     # --- Test Case 4: n=1 ---
#     print("\n--- Test Case 4: n=1 ---")
#     dummy_r_n1 = dummy_r[:, :1]
#     reg_cov_n1 = regcov(dummy_r_n1)
#     print("Shape:", reg_cov_n1.shape)
#     print("Value:", np.round(reg_cov_n1, 4))
#     print("Sample Var:", np.round(np.var(dummy_r_n1, ddof=1), 4)) # Should be equal as a=1/(1+T) is small
