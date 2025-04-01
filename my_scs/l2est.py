import numpy as np
import warnings

def l2est(X, y, params, compute_errors=False):
    """
    Performs L2 shrinkage estimation (Ridge Regression) of the form:
    b = (X + l*I)^(-1) * y
    Optionally computes standard errors.

    Args:
        X (np.ndarray): Input matrix (N x N), typically a covariance or Gram matrix.
        y (np.ndarray): Input vector (N x 1 or N,), typically mean returns or X.T @ y.
        params (dict): Dictionary containing parameters:
            - 'L2pen' (float): L2 penalty weight (lambda or l).
            - 'T' (int): Number of time periods (required only if compute_errors=True).
        compute_errors (bool, optional): If True, compute standard errors. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Estimated coefficient vector b (N,).
            - dict: The input params dictionary (passed through).
            - np.ndarray: Standard errors of b (N,), or NaNs if compute_errors is False.

    Raises:
        ValueError: If required parameters ('L2pen', or 'T' when compute_errors=True)
                    are missing or if matrix dimensions are incompatible.
        np.linalg.LinAlgError: If matrix inversion or solving fails.
    """
    # Ensure inputs are numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)

    # Validate shapes
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError(f"Input X must be a square matrix (N x N), got shape {X.shape}")
    N = X.shape[0]

    if y.shape[0] != N:
        raise ValueError(f"Input y must have length N ({N}), got shape {y.shape}")
    # Ensure y is a column vector for calculations
    y = y.reshape(-1, 1)

    # Get L2 penalty
    if 'L2pen' not in params:
        raise ValueError("Parameter 'L2pen' is required in params dictionary.")
    l = params['L2pen']
    if not isinstance(l, (int, float)) or l < 0:
         warnings.warn(f"L2pen should be a non-negative number, got {l}. Proceeding anyway.")
         # raise ValueError("L2pen must be a non-negative number.") # Or raise error

    # Form regularized matrix X + l*I
    identity_matrix = np.identity(N)
    X_reg = X + l * identity_matrix

    b = np.full((N, 1), np.nan) # Initialize b
    se = np.full((N, 1), np.nan) # Initialize se

    try:
        if compute_errors:
            if 'T' not in params or not isinstance(params['T'], (int, float)) or params['T'] <= 0:
                raise ValueError("Parameter 'T' (positive number of time periods) is required in params when compute_errors=True.")
            T = params['T']

            # Calculate inverse and then coefficients
            Xinv = np.linalg.inv(X_reg)
            b = Xinv @ y

            # Calculate standard errors
            diag_inv = np.diag(Xinv).copy() # Ensure we have a copy
            # Handle potential negative values due to numerical precision before sqrt
            diag_inv[diag_inv < 0] = 0.0
            se = np.sqrt((1 / T) * diag_inv).reshape(-1, 1)

        else:
            # Solve linear system (more stable than inversion)
            b = np.linalg.solve(X_reg, y)
            # se remains NaNs

    except np.linalg.LinAlgError as e:
        warnings.warn(f"Linear algebra error during L2 estimation: {e}. Returning NaNs.")
        # b and se remain NaNs if error occurs

    # Return flattened arrays and params
    return b.flatten(), params, se.flatten()

# Example Usage (optional, for testing)
# if __name__ == '__main__':
#     # Dummy data
#     N = 5
#     T_sim = 100
#     np.random.seed(1)
#     # Simulate data where X is like covariance and y is like mean returns
#     sim_ret = np.random.randn(T_sim, N) * 0.05
#     dummy_X = np.cov(sim_ret, rowvar=False)
#     dummy_y = np.mean(sim_ret, axis=0) + np.random.randn(N) * 0.001 # Add some signal

#     l2_penalty = 0.1
#     params_dict = {'L2pen': l2_penalty, 'T': T_sim}

#     # --- Test Case 1: Compute without errors ---
#     print("--- Test Case 1: No Standard Errors ---")
#     b1, p1, se1 = l2est(dummy_X, dummy_y, params_dict, compute_errors=False)
#     print("Coefficients (b):", np.round(b1, 4))
#     print("Standard Errors (se):", se1) # Should be NaNs

#     # --- Test Case 2: Compute with errors ---
#     print("\n--- Test Case 2: With Standard Errors ---")
#     b2, p2, se2 = l2est(dummy_X, dummy_y, params_dict, compute_errors=True)
#     print("Coefficients (b):", np.round(b2, 4))
#     print("Standard Errors (se):", np.round(se2, 4))

#     # Verify b1 and b2 are close (different computation methods)
#     print("\nAre b1 and b2 close?", np.allclose(b1, b2))

#     # --- Test Case 3: Missing T when errors requested ---
#     print("\n--- Test Case 3: Missing T ---")
#     params_no_T = {'L2pen': l2_penalty}
#     try:
#         l2est(dummy_X, dummy_y, params_no_T, compute_errors=True)
#     except ValueError as e:
#         print("Caught expected error:", e)

#     # --- Test Case 4: Singular Matrix (add large penalty) ---
#     # (Making X singular is tricky, let's test near-singular with small penalty)
#     print("\n--- Test Case 4: Near Singular (small penalty) ---")
#     dummy_X_singular = dummy_X.copy()
#     dummy_X_singular[:, -1] = dummy_X_singular[:, -2] # Make last two columns dependent
#     params_singular = {'L2pen': 1e-10, 'T': T_sim} # Very small penalty
#     # Without penalty, solve/inv would likely fail or give huge numbers
#     # With small penalty, it should work but might be ill-conditioned
#     try:
#         b_sing, _, se_sing = l2est(dummy_X_singular, dummy_y, params_singular, compute_errors=True)
#         print("Coefficients (near singular X):", np.round(b_sing, 2))
#         print("Standard Errors (near singular X):", np.round(se_sing, 2))
#     except np.linalg.LinAlgError as e:
#         print("Caught expected LinAlgError for near-singular matrix:", e)
