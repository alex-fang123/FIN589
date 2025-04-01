import numpy as np

def _planerot(x):
    """
    Computes parameters for and applies a Givens plane rotation.
    Helper function mimicking MATLAB's planerot behavior needed in choldelete.

    Given a 2-element vector x = [f, g], computes c, s such that
    G = [[c, s], [-s, c]]
    G @ [f, g] = [r, 0]

    Args:
        x (np.ndarray): A 2-element numpy array [f, g].

    Returns:
        tuple: (G, y) where G is the 2x2 rotation matrix and y is the rotated vector [r, 0].
    """
    f = x[0]
    g = x[1]
    G = np.identity(2) # Default to identity if no rotation needed

    if g == 0.0:
        r = f
        y = np.array([r, 0.0])
    elif f == 0.0:
        # Non-standard Givens rotation to handle f=0 case simply
        # G = [[0, 1], [-1, 0]] -> results in [g, 0]
        c = 0.0
        s = 1.0 if g > 0 else -1.0 # Ensure r is positive
        r = np.abs(g)
        G = np.array([[c, s], [-s, c]])
        y = np.array([r, 0.0])
    else:
        r = np.hypot(f, g) # sqrt(f**2 + g**2)
        c = f / r
        s = g / r
        G = np.array([[c, s], [-s, c]])
        # y = G @ x # Theoretical result is [r, 0]
        y = np.array([r, 0.0])

    return G, y


def choldelete(R_in, j):
    """
    Fast downdate of Cholesky factorization R = chol(X.T @ X) when column j of X is removed.
    Assumes R_in is the upper triangular Cholesky factor.

    Args:
        R_in (np.ndarray): The original upper triangular Cholesky factor (NumPy array).
                           Must be square (PxP).
        j (int): The 0-based index of the column to remove relative to the original matrix X.

    Returns:
        np.ndarray: The updated upper triangular Cholesky factor ((P-1)x(P-1)).
    """
    # Ensure R is a NumPy array and make a copy to avoid modifying the input
    R = np.array(R_in, dtype=float)

    # Check if R is square
    P = R.shape[0]
    if R.shape[1] != P:
        raise ValueError(f"Input Cholesky factor R must be square (shape {R.shape})")

    # Check if j is a valid index
    if j < 0 or j >= P:
        raise IndexError(f"Index j={j} is out of bounds for R with {P} columns")

    # Remove column j (0-based index)
    R = np.delete(R, j, axis=1)
    rows, n = R.shape # n is the new number of columns (n = P-1)

    # Apply Givens rotations to zero out the subdiagonal elements created by deletion
    # Loop corresponds to MATLAB's k = j:n
    # Python k_py goes from j to n-1 (inclusive), which is j to P-2
    for k_py in range(j, n):
        # Indices for rows involved in rotation (0-based)
        # Corresponds to MATLAB's p = k:k+1
        row_indices = [k_py, k_py + 1]

        # Check if row index k_py+1 is valid (it should be, as max k_py = P-2)
        if row_indices[1] >= rows: # rows = P
             raise IndexError(f"Row index {row_indices[1]} out of bounds during rotation (should not happen).")

        # Extract the 2-element column vector for rotation: R[k_py, k_py] and R[k_py+1, k_py]
        col_vector = R[row_indices, k_py]

        # Compute Givens rotation matrix G and the resulting [r, 0] vector
        G, rotated_col = _planerot(col_vector)

        # Update the column k_py in R with the rotated values [r, 0]
        R[row_indices, k_py] = rotated_col

        # Apply the same rotation G to the rest of the columns (k_py+1 to n-1) in these two rows
        if k_py + 1 < n: # Check if there are columns to the right
            # Columns from k_py+1 up to n-1 (0-based)
            cols_to_rotate = slice(k_py + 1, n)
            # Extract the submatrix R(p, k+1:n)
            sub_matrix = R[row_indices, cols_to_rotate]
            # Apply rotation: G @ R(p, k+1:n)
            R[row_indices, cols_to_rotate] = G @ sub_matrix

    # Remove the last row (which has been zeroed out by the rotations)
    # Corresponds to MATLAB's R(end,:) = []
    R = np.delete(R, -1, axis=0) # Or R = R[:-1, :]

    return R

# Example Usage (optional, for testing)
# if __name__ == '__main__':
#     # Create a positive definite matrix A
#     np.random.seed(0)
#     X = np.random.rand(10, 5) # Example: 10 observations, 5 features
#     A = X.T @ X
#     # Get the upper Cholesky factor (MATLAB's chol returns upper)
#     # np.linalg.cholesky returns lower, so we transpose
#     R_upper = np.linalg.cholesky(A).T

#     print("Original Upper R:\n", np.round(R_upper, 4))

#     # --- Test Case 1: Remove middle column (j=2) ---
#     j_remove = 2
#     print(f"\n--- Deleting column j={j_remove} ---")
#     X_deleted = np.delete(X, j_remove, axis=1)
#     A_deleted_expected = X_deleted.T @ X_deleted
#     R_deleted_expected = np.linalg.cholesky(A_deleted_expected).T
#     R_deleted_actual = choldelete(R_upper, j_remove)
#     print("Actual R_deleted:\n", np.round(R_deleted_actual, 4))
#     print("Expected R_deleted:\n", np.round(R_deleted_expected, 4))
#     print("Are they close?", np.allclose(R_deleted_actual, R_deleted_expected))

#     # --- Test Case 2: Remove first column (j=0) ---
#     j_remove = 0
#     print(f"\n--- Deleting column j={j_remove} ---")
#     X_deleted = np.delete(X, j_remove, axis=1)
#     A_deleted_expected = X_deleted.T @ X_deleted
#     R_deleted_expected = np.linalg.cholesky(A_deleted_expected).T
#     R_deleted_actual = choldelete(R_upper, j_remove)
#     print("Are they close?", np.allclose(R_deleted_actual, R_deleted_expected))

#     # --- Test Case 3: Remove last column (j=4) ---
#     j_remove = 4
#     print(f"\n--- Deleting column j={j_remove} ---")
#     X_deleted = np.delete(X, j_remove, axis=1)
#     A_deleted_expected = X_deleted.T @ X_deleted
#     R_deleted_expected = np.linalg.cholesky(A_deleted_expected).T
#     R_deleted_actual = choldelete(R_upper, j_remove)
#     print("Are they close?", np.allclose(R_deleted_actual, R_deleted_expected))
