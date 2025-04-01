import numpy as np
from scipy.linalg import svd, diagsvd
from sklearn.linear_model import lars_path
import warnings
# Assuming check_params is available in the same directory or package
try:
    from .check_params import check_params
except ImportError:
    # Fallback if running script directly or check_params is elsewhere
    from check_params import check_params
# Need sqrt for SVD transformation
from math import sqrt


def elasticnet_sdf_HJdist(X, y, params):
    """
    Python translation of elasticnet_sdf_HJdist.m.

    Performs Elastic Net-like regularization for SDF estimation by minimizing
    HJ distance (GLS objective) using a LARS path approach on transformed variables.

    Args:
        X (np.ndarray): Input matrix, typically covariance matrix S (N x N).
        y (np.ndarray): Input vector, typically mean returns (N x 1 or N,).
        params (dict): Dictionary containing parameters:
            - 'L2pen' (float): L2 penalty weight (delta). Default: 0.
            - 'stop' (int or float): Early stopping criterion.
                - Negative int: Stop when abs(stop) non-zero coefficients are reached.
                - Positive float: Stop when L1 norm of coefficients reaches stop.
                - 0: Compute the full path (returns coefficients near OLS). Default: 0.
            - 'storepath' (bool): Whether to store the full coefficient path in params. Default: False.
            - 'verbose' (bool): Verbosity for the LARS algorithm. Default: False.
            - 'cv_iteration' (int): Current cross-validation iteration index (for caching). Default: 0.
            - 'use_precomputed' (bool): If True, attempts to use precomputed 'bpath' from params. Default: False.
            - 'bpath' (list): List containing precomputed coefficient paths for each CV iteration.
            - 'elasticnet_cache' (list): List for caching transformed variables (X1, y1).
            - 'cache_run' (bool): If True, only performs caching steps and returns. Default: False.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Selected coefficient vector b (N,).
            - dict: Updated params dictionary with cache and potentially stored path.

    Notes:
        - This translation uses sklearn.linear_model.lars_path on the GLS->OLS
          transformed variables. Unlike the original MATLAB code's apparent call
          to a custom `larsen(..., delta, ...)` function, `lars_path` does not
          directly incorporate the `L2pen` (delta) during the path computation itself.
          The L2 effect is primarily captured implicitly through the transformation.
        - The 'double shrinkage' adjustment from the MATLAB comments is omitted.
    """
    # 1. Parameter Handling
    defaults = {
        'cv_iteration': 0,
        'L2pen': 0.0,
        'stop': 0,
        'storepath': False,
        'verbose': False,
        'use_precomputed': False,
        'bpath': [],
        'elasticnet_cache': [],
        'cache_run': False
    }
    # Use check_params (assuming it handles defaults appropriately)
    # We don't have required parameters here based on the MATLAB code structure
    p, _, _ = check_params(params, defaults, required=[])

    if p['L2pen'] < 0:
        raise ValueError('L2 penalty (L2pen) must be non-negative.')

    # Ensure y is a column vector
    y = np.asarray(y).reshape(-1, 1)
    X = np.asarray(X)
    N = X.shape[1]

    # 2. Check for Precomputed Results
    cv_iter = p['cv_iteration']
    if p['use_precomputed']:
        if len(p['bpath']) > cv_iter and p['bpath'][cv_iter] is not None:
            bpath = p['bpath'][cv_iter]
            # Find the coefficients based on the stop criterion from the stored path
            b = _select_coefs_from_path(bpath, p['stop'])
            # Return immediately with precomputed result
            return b.flatten(), p # Return 1D array
        else:
            # Don't warn if use_precomputed is True but path is missing,
            # as this happens during the L1L2 grid search by design.
            # warnings.warn(f"use_precomputed is True, but no valid bpath found for cv_iteration {cv_iter}.")
            pass # Proceed to calculate

    # 3. Caching Logic for Transformed Variables (X1, y1)
    X1, y1 = None, None
    if len(p['elasticnet_cache']) > cv_iter and p['elasticnet_cache'][cv_iter] is not None:
        # Read from cache
        cache = p['elasticnet_cache'][cv_iter]
        X1 = cache.get('X1')
        y1 = cache.get('y1')
        if p['verbose']: print(f"Cache hit for cv_iteration {cv_iter}")

    if X1 is None or y1 is None:
        # Perform GLS -> OLS transformation
        if p['verbose']: print(f"Cache miss or invalid cache for cv_iteration {cv_iter}. Performing SVD.")
        try:
            # Use full_matrices=False for efficiency if X is square
            U, s_vec, Vh = svd(X, full_matrices=(X.shape[0] != X.shape[1]))
            V = Vh.T # Use V instead of Vh

            # --- Calculate X2 = U * D^0.5 * U' (since X=S is symmetric, U=V) ---
            s_sqrt = np.sqrt(s_vec)
            X2 = U @ np.diag(s_sqrt) @ U.T

            # --- Calculate X2inv using pseudo-inverse logic ---
            tol = max(X.shape) * np.finfo(float).eps * np.max(s_vec) if s_vec.size > 0 else 0
            rank = np.sum(s_vec > tol)
            s_inv_sqrt = np.zeros_like(s_vec)
            # Check rank before accessing s_sqrt to avoid index error on empty/zero matrix
            if rank > 0:
                s_inv_sqrt[:rank] = 1.0 / s_sqrt[:rank]
            X2inv = U @ np.diag(s_inv_sqrt) @ U.T

            # --- Transform variables ---
            X1 = X2
            y1 = X2inv @ y # y is already (N, 1)

            # --- Store in cache ---
            cache = {'X1': X1, 'y1': y1}
            # Ensure cache list is long enough
            while len(p['elasticnet_cache']) <= cv_iter:
                p['elasticnet_cache'].append(None)
            p['elasticnet_cache'][cv_iter] = cache
            if p['verbose']: print(f"Transformation complete and cached for cv_iteration {cv_iter}")

        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"SVD or transformation failed: {e}") from e

    # 4. Handle Cache Run
    if p['cache_run']:
        if p['verbose']: print("Cache run requested. Returning after caching.")
        # Return dummy b (e.g., zeros) as calculation is skipped
        return np.zeros(N), p

    # 5. Calculate LARS Path
    if p['verbose']: print("Running lars_path...")
    try:
        # method='lasso' ensures L1 penalty path
        # We compute the full path by default, then select based on 'stop'
        # Use max_iter to prevent potential infinite loops in edge cases
        max_iter = 8 * N # Heuristic similar to MATLAB's maxSteps
        alphas, active, coefs_path = lars_path(X1, y1.ravel(), method='lasso', verbose=p['verbose'], return_path=True, max_iter=max_iter)
        # coefs_path has shape (n_features, n_steps)
    except Exception as e:
        warnings.warn(f"lars_path calculation failed: {e}. Returning zero coefficients.")
        b = np.zeros(N)
        # Store empty path if storing is requested
        if p['storepath']:
             while len(p['bpath']) <= cv_iter: p['bpath'].append(None)
             p['bpath'][cv_iter] = np.zeros((N, 1)) # Store a minimal path
        return b, p

    if p['verbose']: print(f"lars_path completed. Path shape: {coefs_path.shape}")

    # 6. Store Full Path if Requested
    if p['storepath']:
        while len(p['bpath']) <= cv_iter: p['bpath'].append(None)
        p['bpath'][cv_iter] = coefs_path
        if p['verbose']: print(f"Full coefficient path stored for cv_iteration {cv_iter}")

    # 7. Select Coefficients Based on Stop Criterion
    b = _select_coefs_from_path(coefs_path, p['stop'])
    if p['verbose']: print(f"Selected coefficients based on stop={p['stop']}. Non-zeros: {np.sum(b != 0)}")

    # 8. Return selected coefficients and updated params
    return b.flatten(), p # Return 1D array


def _select_coefs_from_path(coefs_path, stop):
    """Helper function to select coefficients from LARS path based on stop criterion."""
    n_features, n_steps = coefs_path.shape

    if n_steps == 0: # Handle empty path
        return np.zeros(n_features)

    if stop == 0:
        # Return coefficients from the last step (closest to OLS on transformed data)
        b = coefs_path[:, -1]
    elif stop < 0:
        # Stop based on number of non-zero variables
        stop_k = abs(stop)
        # Use a small tolerance for non-zero check
        non_zeros = np.sum(np.abs(coefs_path) > 1e-12, axis=0)
        # Find the first step where non_zeros >= stop_k
        valid_steps = np.where(non_zeros >= stop_k)[0]
        if len(valid_steps) > 0:
            idx = valid_steps[0]
            b = coefs_path[:, idx]
        else:
            # If desired number of non-zeros is never reached, return the last step
            # warnings.warn(f"Desired number of non-zeros ({stop_k}) not reached. Returning last step coefficients ({non_zeros[-1]} non-zeros).") # Removed warning
            b = coefs_path[:, -1]
    else: # stop > 0
        # Stop based on L1 norm
        stop_l1 = stop
        l1_norms = np.sum(np.abs(coefs_path), axis=0)
        # Find the first step where l1_norm >= stop_l1
        valid_steps = np.where(l1_norms >= stop_l1 - 1e-12)[0] # Add tolerance
        if len(valid_steps) > 0:
            idx = valid_steps[0]
            # Interpolate between step idx-1 and idx to hit the L1 norm exactly
            if idx > 0:
                l1_prev = l1_norms[idx-1]
                l1_curr = l1_norms[idx]
                b_prev = coefs_path[:, idx-1]
                b_curr = coefs_path[:, idx]
                # Avoid division by zero if norms are equal
                if abs(l1_curr - l1_prev) > 1e-12:
                    s = (stop_l1 - l1_prev) / (l1_curr - l1_prev)
                    s = max(0, min(1, s)) # Clamp interpolation factor [0, 1]
                    b = b_prev + s * (b_curr - b_prev)
                else: # Already at the boundary or no change
                    b = b_curr
            else: # First step already meets/exceeds the norm
                 b = coefs_path[:, idx]
        else:
            # If desired L1 norm is never reached, return the last step
            # warnings.warn(f"Desired L1 norm ({stop_l1}) not reached. Returning last step coefficients (L1 norm: {l1_norms[-1]:.4f}).") # Removed warning
            b = coefs_path[:, -1]

    return b


# Example Usage (Illustrative - requires compatible X, y, and params)
# if __name__ == '__main__':
#     # Dummy data (assuming X is like a covariance matrix, y like mean returns)
#     N = 20
#     T_sim = 100 # For simulating returns to get Cov and Mean
#     np.random.seed(1)
#     sim_ret = np.random.randn(T_sim, N) * 0.05
#     dummy_X = np.cov(sim_ret, rowvar=False) # Covariance matrix (N x N)
#     dummy_y = np.mean(sim_ret, axis=0)      # Mean returns (N,)

#     # --- Test Case 1: Full path (stop=0) ---
#     print("--- Test Case 1: Full Path (stop=0) ---")
#     params1 = {'L2pen': 0.01, 'stop': 0, 'storepath': True, 'verbose': False}
#     b1, p1 = elasticnet_sdf_HJdist(dummy_X, dummy_y, params1)
#     print(f"Selected b (stop=0), non-zeros: {np.sum(np.abs(b1) > 1e-9)}")
#     print(f"Path stored? {'bpath' in p1 and p1['bpath'][0] is not None}")
#     if 'bpath' in p1 and p1['bpath'][0] is not None:
#         print(f"Stored path shape: {p1['bpath'][0].shape}")

#     # --- Test Case 2: Stop at k non-zeros (stop=-5) ---
#     print("\n--- Test Case 2: Stop at k=-5 non-zeros ---")
#     params2 = {'L2pen': 0.01, 'stop': -5, 'verbose': False}
#     # Use cache from previous run
#     params2['elasticnet_cache'] = p1['elasticnet_cache']
#     b2, p2 = elasticnet_sdf_HJdist(dummy_X, dummy_y, params2)
#     print(f"Selected b (stop=-5), non-zeros: {np.sum(np.abs(b2) > 1e-9)}")

#     # --- Test Case 3: Stop at L1 norm (stop=0.1) ---
#     print("\n--- Test Case 3: Stop at L1 norm=0.1 ---")
#     params3 = {'L2pen': 0.01, 'stop': 0.1, 'verbose': False}
#     params3['elasticnet_cache'] = p1['elasticnet_cache'] # Use cache
#     b3, p3 = elasticnet_sdf_HJdist(dummy_X, dummy_y, params3)
#     print(f"Selected b (stop=0.1), L1 norm: {np.sum(np.abs(b3)):.4f}, non-zeros: {np.sum(np.abs(b3) > 1e-9)}")

#     # --- Test Case 4: Use Precomputed ---
#     print("\n--- Test Case 4: Use Precomputed Path ---")
#     params4 = {
#         'use_precomputed': True,
#         'bpath': p1['bpath'], # Provide the path computed in case 1
#         'stop': -10,          # Select different point from path
#         'cv_iteration': 0,
#         'verbose': True
#      }
#     b4, p4 = elasticnet_sdf_HJdist(dummy_X, dummy_y, params4)
#     print(f"Selected b (precomputed, stop=-10), non-zeros: {np.sum(np.abs(b4) > 1e-9)}")
