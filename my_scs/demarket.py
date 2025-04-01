import numpy as np

def demarket(r, mkt, b=None):
    """
    Calculates market betas and market-adjusted excess returns.

    Args:
        r (np.ndarray): Asset returns matrix (T x N).
        mkt (np.ndarray): Market return vector (T x 1 or T,).
        b (np.ndarray, optional): Pre-calculated market betas (1 x N or N,).
                                  If None, betas are calculated via OLS. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Market-adjusted excess returns (rme, T x N).
            - np.ndarray: Market betas used or calculated (b, 1 x N).

    Raises:
        ValueError: If input dimensions are incompatible.
    """
    # Ensure inputs are numpy arrays
    r = np.asarray(r)
    mkt = np.asarray(mkt)

    # Ensure r is 2D (T x N)
    if r.ndim != 2:
        raise ValueError(f"Input 'r' must be a 2D array (T x N), got shape {r.shape}")
    T, N = r.shape

    # Ensure mkt is a column vector (T x 1)
    if mkt.ndim == 1:
        mkt = mkt.reshape(-1, 1)
    if mkt.shape[0] != T or mkt.shape[1] != 1:
        raise ValueError(f"Input 'mkt' must have shape ({T}, 1) or ({T},), got {mkt.shape}")

    if b is None:
        # Calculate market beta via OLS: r_i = alpha_i + b_i * mkt + eps_i
        # rhs = [ones(T, 1), mkt]
        rhs = np.hstack((np.ones((T, 1)), mkt))
        try:
            # Solve rhs @ coeffs = r for coeffs using least squares
            coeffs, residuals, rank, s = np.linalg.lstsq(rhs, r, rcond=None)
            # coeffs will have shape (2, N). Betas are the second row.
            b_calc = coeffs[1, :].reshape(1, N) # Ensure shape (1, N)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"OLS regression failed: {e}") from e
        b_used = b_calc
    else:
        # Use provided betas
        b = np.asarray(b)
        # Ensure b is a row vector (1 x N)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        if b.shape != (1, N):
            raise ValueError(f"Provided 'b' must have shape (1, {N}) or ({N},), got {b.shape}")
        b_used = b

    # Calculate market-adjusted returns: rme = r - mkt * b
    # mkt (T, 1) @ b_used (1, N) -> (T, N)
    market_component = mkt @ b_used
    rme = r - market_component

    return rme, b_used

# Example Usage (optional, for testing)
# if __name__ == '__main__':
#     # Dummy data
#     T = 100
#     N = 5
#     np.random.seed(0)
#     mkt_ret = np.random.randn(T, 1) * 0.02 + 0.001
#     # Generate asset returns with known betas and some noise
#     true_betas = np.array([[0.8, 1.0, 1.2, 0.5, 1.5]]) # Shape (1, N)
#     noise = np.random.randn(T, N) * 0.01
#     asset_ret = 0.0005 + mkt_ret @ true_betas + noise # Add small alpha

#     # --- Test Case 1: Calculate betas ---
#     print("--- Calculating Betas ---")
#     rme_calc, b_calc = demarket(asset_ret, mkt_ret)
#     print("Calculated Betas (b):", np.round(b_calc, 3))
#     print("True Betas:", np.round(true_betas, 3))
#     print("Shape of rme:", rme_calc.shape)
#     # Check if mean of rme is close to the true alpha (0.0005)
#     print("Mean of rme (should be near alpha):", np.round(np.mean(rme_calc, axis=0), 4))


#     # --- Test Case 2: Provide betas ---
#     print("\n--- Providing Betas ---")
#     provided_betas = np.array([0.81, 1.01, 1.19, 0.51, 1.49]) # Slightly different
#     rme_prov, b_prov = demarket(asset_ret, mkt_ret, b=provided_betas)
#     print("Provided Betas used (b):", np.round(b_prov, 3))
#     print("Shape of rme:", rme_prov.shape)
#     print("Mean of rme (using provided b):", np.round(np.mean(rme_prov, axis=0), 4))

#     # --- Test with 1D market input ---
#     print("\n--- Test with 1D market input ---")
#     mkt_ret_1d = mkt_ret.flatten()
#     rme_1d, b_1d = demarket(asset_ret, mkt_ret_1d)
#     print("Calculated Betas (1D mkt):", np.round(b_1d, 3))
#     print("Are betas close to original calc?", np.allclose(b_calc, b_1d))
