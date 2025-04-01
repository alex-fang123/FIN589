import numpy as np

def cvpartition_contiguous(n_samples, k):
    """
    Generates k contiguous, non-overlapping partitions (folds) for n_samples.

    Mimics the partitioning logic of the MATLAB cvpartition_contiguous function,
    but returns boolean masks compatible with the cross_validate.py structure.

    Args:
        n_samples (int): Total number of samples.
        k (int): Number of folds.

    Returns:
        list: A list of k boolean numpy arrays. Each array has length n_samples,
              where True indicates membership in that fold (test set in CV context).
    """
    if k <= 0 or k > n_samples:
        raise ValueError("Number of folds k must be positive and not exceed n_samples.")

    fold_size = n_samples // k
    indices = np.arange(n_samples) # 0-based indices

    partition_masks = []
    current_start = 0
    for i in range(k):
        # Calculate the end index for the current fold
        if i < k - 1:
            current_end = current_start + fold_size
        else:
            # Last fold takes the remainder
            current_end = n_samples

        # Create boolean mask for the current fold
        mask = np.zeros(n_samples, dtype=bool)
        if current_start < current_end: # Ensure start < end before slicing
             mask[current_start:current_end] = True
        partition_masks.append(mask)

        # Update start index for the next fold
        current_start = current_end

    return partition_masks

# Example Usage (optional, for testing)
# if __name__ == '__main__':
#     n = 10
#     k = 3
#     masks = cvpartition_contiguous(n, k)
#     print(f"Partitioning n={n} into k={k} folds:")
#     for i, mask in enumerate(masks):
#         print(f"Fold {i+1} mask: {mask.astype(int)}")
#         print(f"Fold {i+1} indices: {np.where(mask)[0]}")

#     n = 11
#     k = 3
#     masks = cvpartition_contiguous(n, k)
#     print(f"\nPartitioning n={n} into k={k} folds:")
#     for i, mask in enumerate(masks):
#         print(f"Fold {i+1} mask: {mask.astype(int)}")
#         print(f"Fold {i+1} indices: {np.where(mask)[0]}")
