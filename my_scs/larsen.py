import numpy as np
from scipy.linalg import solve_triangular, pinv, solve
import warnings
# Assuming cholinsert and choldelete are available
try:
    from .cholinsert import cholinsert
    from .choldelete import choldelete
except ImportError:
    # Fallback if running directly
    try:
        from cholinsert import cholinsert
        from choldelete import choldelete
    except ImportError:
        raise ImportError("Could not import cholinsert and choldelete functions.")


def larsen(X, y, delta, stop=0, Gram=None, storepath=False, verbose=False):
    """
    Python translation of the LARS-EN algorithm from larsen.m.

    Args:
        X (np.ndarray): Predictor matrix (n x p).
        y (np.ndarray): Response vector (n x 1 or n,).
        delta (float): L2 penalty weight (Elastic Net parameter). Must be non-negative.
        stop (int or float, optional): Early stopping criterion. Defaults to 0.
            - Negative int: Stop when abs(stop) non-zero variables are reached.
            - Positive float: Stop when L1 norm of coefficients reaches stop/(1+delta).
            - 0: Compute the full path.
        Gram (np.ndarray, optional): Precomputed Gram matrix X'X (p x p). Defaults to None.
        storepath (bool, optional): Whether to store the entire coefficient path. Defaults to False.
        verbose (bool, optional): Print progress information. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Coefficient vector(s). If storepath, matrix (p x steps+1), else vector (p,).
                          Includes the initial zero step if storepath is True.
            - int: Number of steps taken (excluding initial zero step).

    Raises:
        ValueError: If inputs are invalid.
        np.linalg.LinAlgError: If linear algebra operations fail.
    """
    # --- Input Validation and Setup ---
    X = np.asarray(X)
    y = np.asarray(y).ravel() # Ensure y is 1D

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError(f"X has {n} rows, but y has {y.shape[0]} elements.")
    if delta < 0:
        raise ValueError("L2 penalty delta must be non-negative.")

    # Determine maximum number of active variables and steps
    if delta < np.finfo(float).eps:
        maxVariables = min(n, p) # LASSO case
    else:
        maxVariables = p # Elastic Net case
    # Max steps slightly increased for safety margin
    maxSteps = 8 * maxVariables + p

    # Initialize coefficient vector/path
    if storepath:
        # Pre-allocate reasonable size, can grow if needed. Start with step 0 (all zeros)
        b_path = np.zeros((p, min(maxSteps + 1, 2 * p + 1)))
        b_current = np.zeros(p) # Keep track of current b for calculations
    else:
        b_path = None # Not storing the path
        b_current = np.zeros(p)
        b_prev = np.zeros(p) # Need previous step for L1 stop interpolation

    mu = np.zeros(n) # Current prediction/position
    I = list(range(p)) # Inactive set indices (0-based)
    A = [] # Active set indices (0-based)
    R = None # Cholesky factor R'R = X_A'X_A + delta*I

    useGram = Gram is not None
    if useGram:
        if Gram.shape != (p, p):
            raise ValueError(f"Gram matrix must have shape ({p}, {p}), got {Gram.shape}")
    else:
        Gram = None # Ensure Gram is None if not provided

    # Correction for stopping criterion (naive Elastic Net) as in MATLAB code
    stop_adj = stop
    if delta > 0 and stop > 0:
         stop_adj = stop / (1.0 + delta)

    lassoCond = False # Flag indicating a variable needs to be dropped next
    stopCond = False # Flag indicating early stopping condition met
    step = 0 # Step count (starts at 0, first step calculates first non-zero coef)
    initial_cmax = 1.0 # For relative correlation check

    if verbose:
        print('Step\tAdded\tDropped\t\tActive set size')

    # --- LARS Main Loop ---
    # Stop when active set full, stop condition met, or max steps reached
    while len(A) < maxVariables and not stopCond and step < maxSteps:
        r = y - mu # Current residual
        c = X.T @ r # Correlations c_j = x_j' * r

        # Find variable in I with max correlation
        if not I: # If inactive set is empty
             if verbose: print("All variables active. Moving towards final solution.")
             cmax = 0
             cidx = -1
        else:
             abs_c_I = np.abs(c[I])
             cmax_idx_in_I = np.argmax(abs_c_I)
             cmax = abs_c_I[cmax_idx_in_I]
             cidx = I[cmax_idx_in_I]

        # Store max initial correlation for relative stopping check
        if step == 0:
            initial_cmax = cmax if cmax > 1e-12 else 1.0

        # Add variable to active set (if not dropping in this iteration)
        if not lassoCond:
            # Stop if all correlations are negligible relative to max initial corr
            # Check only after first step
            if step > 0 and cmax < 1e-12 * initial_cmax:
                 if verbose: print(f"All inactive correlations close to zero ({cmax:.2e}). Stopping.")
                 break

            if cidx != -1: # Check if a variable was actually found to add
                if not useGram:
                    try:
                        x_cidx = X[:, cidx].reshape(-1, 1)
                        X_A = X[:, A] if A else np.zeros((n, 0))
                        R = cholinsert(R, x_cidx, X_A, delta)
                    except (np.linalg.LinAlgError, ValueError) as e:
                        warnings.warn(f"cholinsert failed at step {step+1} adding var {cidx}: {e}. Stopping.")
                        break

                if verbose:
                    print(f'{step+1}\t\t{cidx}\t\t\t\t\t{len(A) + 1}')

                # Update active/inactive sets
                A.append(cidx)
                I.pop(cmax_idx_in_I)
                A.sort() # Keep active set sorted
                I.sort() # Keep inactive set sorted
            else:
                 # If cidx == -1, it means I is empty, proceed to calculate gamma=1
                 pass
        else:
            # Reset drop flag as we are processing the drop from the *previous* step
            lassoCond = False

        # If active set is empty after potential drop, stop
        if not A:
            if verbose: print("Active set empty. Stopping.")
            break

        # Calculate OLS solution on active set: (X_A'X_A + delta*I) b = X_A' y
        X_A = X[:, A]
        X_A_T_y = X_A.T @ y

        try:
            if useGram:
                # Solve (Gram[A,A] + delta*I) b = X_A'y
                Gram_AA = Gram[np.ix_(A, A)]
                target_matrix = Gram_AA + delta * np.identity(len(A))
                b_OLS_A = solve(target_matrix, X_A_T_y, assume_a='sym')
            else:
                # Solve using Cholesky factor R: R'R b = X_A'y
                z = solve_triangular(R.T, X_A_T_y, lower=True, check_finite=False)
                b_OLS_A = solve_triangular(R, z, lower=False, check_finite=False)
        except np.linalg.LinAlgError as e:
            warnings.warn(f"Solving for b_OLS failed at step {step+1}: {e}. Stopping.")
            break

        # Calculate direction towards the OLS solution for the active set
        d = X_A @ b_OLS_A - mu

        # Calculate step length gamma = min(positive step lengths to next event)
        # --- Calculate gamma_tilde (step to zero crossing) ---
        b_current_A = b_current[A]
        diff_b = b_current_A - b_OLS_A
        valid_idx_tilde = np.abs(diff_b) > 1e-12
        gamma_tilde_vec = np.full_like(b_current_A, np.inf)
        if np.any(valid_idx_tilde):
             gamma_tilde_vec[valid_idx_tilde] = b_current_A[valid_idx_tilde] / diff_b[valid_idx_tilde]
        gamma_tilde_vec[gamma_tilde_vec <= 1e-12] = np.inf
        if not gamma_tilde_vec.size:
            gamma_tilde = np.inf
            dropIdx = -1
        else:
            gamma_tilde_idx_in_A = np.argmin(gamma_tilde_vec)
            gamma_tilde = gamma_tilde_vec[gamma_tilde_idx_in_A]
            dropIdx = gamma_tilde_idx_in_A # Index within A

        # --- Calculate gamma_eqi (step to next correlation equality) ---
        if not I: # If inactive set is empty, go all the way to OLS
            gamma_eqi = 1.0
        else:
            cd = X.T @ d # Correlation of direction d with all variables
            cd_I = cd[I]
            c_I = c[I] # Re-fetch inactive correlations

            # Use the calculation from the MATLAB code directly
            term1_num = c_I - cmax
            term1_den = cd_I - cmax
            term2_num = c_I + cmax
            term2_den = cd_I + cmax

            gammas1 = np.full_like(c_I, np.inf)
            gammas2 = np.full_like(c_I, np.inf)

            valid_den1 = np.abs(term1_den) > 1e-12
            gammas1[valid_den1] = term1_num[valid_den1] / term1_den[valid_den1]

            valid_den2 = np.abs(term2_den) > 1e-12
            gammas2[valid_den2] = term2_num[valid_den2] / term2_den[valid_den2]

            all_gammas = np.concatenate((gammas1, gammas2))
            # Filter only positive gammas
            positive_gammas = all_gammas[all_gammas > 1e-12]

            if not positive_gammas.size:
                 gamma_eqi = 1.0 # Go towards OLS if no positive step found
            else:
                 gamma_eqi = np.min(positive_gammas)

        # --- Choose overall step length gamma ---
        gamma = min(gamma_eqi, gamma_tilde, 1.0)

        # Check if a variable should be dropped (LASSO condition)
        if gamma_tilde < gamma_eqi - 1e-12:
            lassoCond = True
            # dropIdx is already set

        # --- Update coefficients and prediction ---
        delta_b_A = gamma * (b_OLS_A - b_current_A)
        b_current[A] += delta_b_A
        mu += gamma * d
        step += 1 # Increment step counter *after* update

        # --- Store path ---
        if storepath:
            if step >= b_path.shape[1]: # Resize if needed
                b_path = np.hstack((b_path, np.zeros((p, b_path.shape[1]))))
            b_path[:, step] = b_current

        # --- Check stopping criteria ---
        if stop != 0:
            if stop < 0: # Stop based on number of non-zeros
                # Count non-zeros in current active set coefs
                current_non_zeros = np.sum(np.abs(b_current[A]) > 1e-12)
                # If lasso condition is true, one var will be dropped *next*,
                # so check if current count meets criterion.
                stopCond = current_non_zeros >= abs(stop)
            else: # Stop based on L1 norm (using adjusted stop_adj)
                current_l1_norm = np.sum(np.abs(b_current))
                if current_l1_norm >= stop_adj:
                    # Interpolate to find exact solution at the boundary
                    if storepath:
                        l1_prev = np.sum(np.abs(b_path[:, step - 1]))
                        b_prev_interp = b_path[:, step - 1]
                    else:
                        l1_prev = np.sum(np.abs(b_prev))
                        b_prev_interp = b_prev

                    if abs(current_l1_norm - l1_prev) > 1e-12:
                        s = (stop_adj - l1_prev) / (current_l1_norm - l1_prev)
                        s = max(0, min(1, s)) # Clamp interpolation factor [0, 1]
                    else:
                        s = 1.0 # Already at the boundary or no change

                    b_interp = b_prev_interp + s * (b_current - b_prev_interp)

                    if storepath:
                         b_path[:, step] = b_interp # Overwrite current step with interpolated
                    b_current = b_interp # Update current b as well
                    stopCond = True

        # --- Drop variable if LASSO condition met AND not stopping ---
        if lassoCond and not stopCond:
             if dropIdx != -1 and dropIdx < len(A):
                 idx_to_drop_in_A = dropIdx
                 original_idx_dropped = A[idx_to_drop_in_A]

                 if verbose:
                     print(f'{step}\t\t\t\t{original_idx_dropped}\t\t\t{len(A) - 1}')

                 if not useGram:
                     try:
                         R = choldelete(R, idx_to_drop_in_A)
                     except (np.linalg.LinAlgError, ValueError, IndexError) as e:
                         warnings.warn(f"choldelete failed at step {step} dropping var {original_idx_dropped} (idx {idx_to_drop_in_A}): {e}. Stopping.")
                         stopCond = True # Force stop

                 # Update active/inactive sets only if choldelete succeeded or Gram is used
                 if not stopCond:
                     b_current[original_idx_dropped] = 0.0 # Set dropped coef to zero
                     if storepath: b_path[original_idx_dropped, step] = 0.0

                     I.append(original_idx_dropped)
                     A.pop(idx_to_drop_in_A)
                     I.sort()
                     # A remains sorted after pop
             else:
                 warnings.warn(f"LASSO condition met at step {step} but dropIdx ({dropIdx}) is invalid for active set size {len(A)}. Continuing without drop.")
             # Reset lassoCond *after* processing the drop for this iteration
             lassoCond = False


        # Update b_prev for next iteration's interpolation check if not storing path
        if not storepath:
            b_prev = b_current.copy()

    # --- Finalization ---
    final_steps_taken = step # Number of iterations run
    if storepath:
        # Trim path to actual steps taken (+1 for initial zero step)
        b_final = b_path[:, :final_steps_taken + 1]
    else:
        b_final = b_current # Return the final coefficient vector

    # Issue warning if max steps reached
    if final_steps_taken >= maxSteps:
        warnings.warn('LARS-EN forced exit. Maximum number of steps reached.', RuntimeWarning)

    # Return path or final vector, and number of steps
    return b_final, final_steps_taken
