import numpy as np
from scipy.linalg import solve_triangular
import warnings

def cholinsert(R_in, x, X, delta=0.0):
    """
    Fast update of the upper Cholesky factorization R = chol(X.T @ X + delta*I)
    when a new column x is added to X, forming [X, x].

    Args:
        R_in (np.ndarray or None): The current upper triangular Cholesky factor (PxP).
                                   If None or empty, assumes X was empty.
        x (np.ndarray): The new column vector to add (Nx1).
        X (np.ndarray or None): The original data matrix (NxP). Can be None if R_in is None.
        delta (float, optional): Regularization term added to the diagonal. Defaults to 0.0.

    Returns:
        np.ndarray: The updated upper triangular Cholesky factor ((P+1)x(P+1)).

    Raises:
        ValueError: If dimensions are incompatible or if the resulting diagonal
                    element R_kk^2 is negative (indicating non-positive definiteness).
    """
    # Ensure x is a 2D column vector
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2 or x.shape[1] != 1:
         raise ValueError(f"Input 'x' must be a column vector (shape Nx1), got {x.shape}")

    N = x.shape[0]

    # Calculate the new diagonal element for the augmented Gram matrix
    # diag_k = x.T @ x + delta
    diag_k = np.dot(x[:, 0], x[:, 0]) + delta # More efficient for vector dot product

    # Case 1: Original R is empty (X was empty)
    if R_in is None or R_in.size == 0:
        if X is not None and X.size != 0:
            warnings.warn("R_in is empty but X is not. Assuming X should be empty.")
        if diag_k < 0:
             raise ValueError("Cannot compute Cholesky factor: diag_k is negative.")
        # The new R is just a 1x1 matrix
        return np.array([[np.sqrt(diag_k)]])

    # Case 2: Original R is not empty
    R = np.array(R_in, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"Input R_in must be a square matrix, got shape {R.shape}")

    P = R.shape[1] # Number of columns in original X

    if X is None or X.size == 0:
         raise ValueError("R_in is provided but X is empty or None.")
    if X.ndim != 2:
        raise ValueError(f"Input X must be a 2D matrix (NxP), got {X.ndim} dimensions")
    if X.shape[0] != N:
        raise ValueError(f"Number of rows in x ({N}) does not match X ({X.shape[0]})")
    if X.shape[1] != P:
        raise ValueError(f"Number of columns in R_in ({P}) does not match X ({X.shape[1]})")


    # Calculate the off-diagonal elements for the new column
    # col_k = x.T @ X
    col_k = np.dot(x[:, 0], X) # Shape (P,)

    # Solve R.T @ R_k = col_k.T for R_k
    # We need col_k as a column vector for solve_triangular
    col_k_col = col_k.reshape(-1, 1) # Shape (P, 1)

    # solve_triangular solves a @ y = b
    # Here a = R.T (lower triangular), b = col_k_col
    try:
        R_k = solve_triangular(R.T, col_k_col, lower=True, check_finite=False) # Shape (P, 1)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(f"Failed to solve triangular system: {e}") from e


    # Calculate the square of the new diagonal element R_kk
    # R_kk^2 = diag_k - R_k.T @ R_k
    R_kk_squared = diag_k - np.dot(R_k[:, 0], R_k[:, 0]) # More efficient dot product

    # Check for potential numerical issues leading to negative sqrt argument
    if R_kk_squared < -np.finfo(float).eps * abs(diag_k): # Allow small negative due to precision
         raise ValueError(f"Cannot compute Cholesky factor: R_kk^2 ({R_kk_squared:.2e}) is negative. Matrix may not be positive definite.")
    elif R_kk_squared < 0:
        R_kk_squared = 0.0 # Clamp small negatives to zero

    R_kk = np.sqrt(R_kk_squared)

    # Construct the new R by augmenting
    # R_new = [ R   R_k ]
    #         [ 0  R_kk ]
    R_new = np.zeros((P + 1, P + 1), dtype=float)
    R_new[:P, :P] = R
    R_new[:P, P] = R_k[:, 0] # Assign the column vector R_k
    R_new[P, P] = R_kk

    return R_new

# Example Usage (optional, for testing)
# if __name__ == '__main__':
#     np.random.seed(1)
#     N = 10
#     P = 3
#     X = np.random.rand(N, P)
#     x_new = np.random.rand(N, 1)
#     delta_reg = 0.1

#     # --- Calculate R from scratch ---
#     A = X.T @ X + delta_reg * np.identity(P)
#     R_orig = np.linalg.cholesky(A).T # Upper Cholesky
#     print("Original R:\n", np.round(R_orig, 4))

#     # --- Calculate R_updated using cholinsert ---
#     R_updated_actual = cholinsert(R_orig, x_new, X, delta_reg)
#     print("\nUpdated R (cholinsert):\n", np.round(R_updated_actual, 4))

#     # --- Calculate R_updated expected (full calculation) ---
#     X_augmented = np.hstack((X, x_new))
#     A_augmented = X_augmented.T @ X_augmented + delta_reg * np.identity(P + 1)
#     R_updated_expected = np.linalg.cholesky(A_augmented).T # Upper Cholesky
#     print("\nUpdated R (expected):\n", np.round(R_updated_expected, 4))

#     print("\nAre they close?", np.allclose(R_updated_actual, R_updated_expected))

#     # --- Test empty R case ---
#     print("\n--- Testing empty R case ---")
#     R_empty_actual = cholinsert(None, x_new, None, delta_reg)
#     print("Updated R from empty (cholinsert):\n", np.round(R_empty_actual, 4))
#     A_empty_aug = x_new.T @ x_new + delta_reg * np.identity(1)
#     R_empty_expected = np.linalg.cholesky(A_empty_aug).T
#     print("Updated R from empty (expected):\n", np.round(R_empty_expected, 4))
#     print("Are they close?", np.allclose(R_empty_actual, R_empty_expected))
