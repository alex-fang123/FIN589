# ------------------- Setup -------------------
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import Lasso # Import Lasso for potential use if minimize fails
from sklearn.model_selection import KFold # For a more standard CV approach if needed

# Set date range
t0 = pd.to_datetime("1973-01-01")
tN = pd.to_datetime("2017-12-31")

# Set paths
datapath = "./my_scs/Data" # Updated path as requested
# instrpath = os.path.join(datapath, "instruments") # Not used in the provided snippet
ffact5 = "F-F_Research_Data_Factors.csv"
ff25 = "25_Portfolios_5x5_average_value_weighted_returns_monthly.csv"
date_format = "%Y/%m/%d" # Corrected date format based on file content

# ------------------- Load Data -------------------
# Read factor data
factor_path = os.path.join(datapath, ffact5)
# Try loading with potential skiprows if header is not on the first line
try:
    # Removed skiprows=3 as header is on the first line
    ff_data = pd.read_csv(factor_path)
except FileNotFoundError:
    print(f"Error: Factor file not found at {factor_path}")
    exit()
except Exception as e:
    print(f"Error reading factor file: {e}")
    # Attempt without skiprows as fallback
    try:
        ff_data = pd.read_csv(factor_path)
    except Exception as e2:
        print(f"Fallback reading failed: {e2}")
        exit()

# Handle potential unnamed columns or different date column names
if 'Unnamed: 0' in ff_data.columns:
    ff_data['Date'] = ff_data['Unnamed: 0']
elif 'date' in ff_data.columns:
     ff_data['Date'] = ff_data['date']
# Add more potential date column names if necessary

# Rename columns FIRST
ff_data.columns = ff_data.columns.str.strip().str.replace('-', '_').str.replace(' ', '_')

# Identify and convert date column AFTER renaming
date_col_name = None
if 'Date' in ff_data.columns:
    date_col_name = 'Date'
elif 'date' in ff_data.columns: # Check for lowercase 'date' after renaming
    date_col_name = 'date'
elif 'Unnamed_0' in ff_data.columns: # Check for 'Unnamed_0' after renaming
    date_col_name = 'Unnamed_0'

if date_col_name:
    ff_data['Date'] = pd.to_datetime(ff_data[date_col_name].astype(str), format=date_format, errors='coerce')
    # Optionally rename the identified date column to 'Date' if it wasn't already
    if date_col_name != 'Date':
        ff_data.rename(columns={date_col_name: 'Date'}, inplace=True)
else:
    print("Error: Could not identify the date column in factor data.")
    exit()

# Ensure numeric conversion for factors, coercing errors
factor_cols = ['Mkt_RF', 'SMB', 'HML', 'RF'] # Add RMW, CMA if using 5 factors
for col in factor_cols:
    if col in ff_data.columns:
        ff_data[col] = pd.to_numeric(ff_data[col], errors='coerce')
    else:
        print(f"Warning: Factor column '{col}' not found in factor data.")


# Read portfolio return data
ret_path = os.path.join(datapath, ff25)
# Try loading with potential skiprows
try:
    # Removed skiprows=15 as header is on the first line
    ret_data = pd.read_csv(ret_path)
except FileNotFoundError:
    print(f"Error: Portfolio return file not found at {ret_path}")
    exit()
except Exception as e:
    print(f"Error reading portfolio return file: {e}")
    # Attempt without skiprows as fallback
    try:
        ret_data = pd.read_csv(ret_path)
    except Exception as e2:
        print(f"Fallback reading failed: {e2}")
        exit()

# Handle potential unnamed columns or different date column names
if 'Unnamed: 0' in ret_data.columns:
    ret_data['Date'] = ret_data['Unnamed: 0']
elif 'date' in ret_data.columns:
     ret_data['Date'] = ret_data['date']
# Add more potential date column names if necessary

# Rename columns FIRST
ret_data.columns = ret_data.columns.str.strip().str.replace('-', '_').str.replace(' ', '_')

# Identify and convert date column AFTER renaming
date_col_name_ret = None
if 'Date' in ret_data.columns:
    date_col_name_ret = 'Date'
elif 'date' in ret_data.columns: # Check for lowercase 'date' after renaming
    date_col_name_ret = 'date'
elif 'Unnamed_0' in ret_data.columns: # Check for 'Unnamed_0' after renaming
    date_col_name_ret = 'Unnamed_0'

if date_col_name_ret:
    ret_data['Date'] = pd.to_datetime(ret_data[date_col_name_ret].astype(str), format=date_format, errors='coerce')
    # Optionally rename the identified date column to 'Date' if it wasn't already
    if date_col_name_ret != 'Date':
        ret_data.rename(columns={date_col_name_ret: 'Date'}, inplace=True)
else:
    print("Error: Could not identify the date column in return data.")
    exit()

# Ensure numeric conversion for returns, coercing errors
# Identify return columns AFTER potential date column rename
ret_cols_raw = [col for col in ret_data.columns if col != 'Date']
for col in ret_cols_raw:
    ret_data[col] = pd.to_numeric(ret_data[col], errors='coerce')


# Drop rows with NaT dates from parsing errors
ff_data.dropna(subset=['Date'], inplace=True)
ret_data.dropna(subset=['Date'], inplace=True)

# Filter by date range
if t0 is not None:
    ff_data = ff_data[ff_data['Date'] >= t0]
    ret_data = ret_data[ret_data['Date'] >= t0]
if tN is not None:
    ff_data = ff_data[ff_data['Date'] <= tN]
    ret_data = ret_data[ret_data['Date'] <= tN]

# Merge data on Date
# Ensure 'Date' column exists and handle potential missing columns after cleaning
if 'Date' not in ff_data.columns or 'Date' not in ret_data.columns:
    print("Error: 'Date' column missing in one of the dataframes after cleaning.")
    exit()

DATA = pd.merge(ff_data, ret_data, on="Date", how='inner') # Use inner merge to keep only matching dates

# Drop rows with any NaN values that might hinder calculations
DATA.dropna(inplace=True)

print("Available columns after merge and cleaning:")
print(DATA.columns)

# ------------------- Preprocess Returns -------------------
if 'Date' not in DATA.columns or 'RF' not in DATA.columns:
    print("Error: 'Date' or 'RF' column missing in merged DATA.")
    exit()

dates = DATA['Date']
# mkt = DATA['Mkt_RF'] / 100 # Not directly used in the objective function part
rf = DATA['RF'] / 100

# Identify return columns dynamically after merge
ret_cols = [col for col in ret_data.columns if col in DATA.columns and col != 'Date']
if not ret_cols:
    print("Error: No return columns found in merged DATA.")
    exit()

ret_matrix = DATA[ret_cols].values / 100
ret = ret_matrix - rf.values[:, np.newaxis]  # excess returns

# Clean labels for use as dictionary keys or variable names
labels = pd.Index(ret_cols).str.replace('.', '_', regex=False).str.replace(' ', '_')
labels = pd.Index([label if label.isidentifier() else f"X{label}" for label in labels])

if ret.shape[0] == 0:
    print("Error: No data remaining after preprocessing and date filtering.")
    exit()

# ------------------- LASSO Objective -------------------
# Note: Minimizing this objective with L1 using generic solvers like BFGS can be unstable.
# Consider using specialized libraries (like sklearn.linear_model.Lasso) if results are poor.
def lasso_objective(b, mu, Sigma, Sigma_inv, alpha):
    """Calculates the LASSO objective function based on HJ distance."""
    if Sigma_inv is None:
        # Fallback if Sigma_inv calculation failed earlier
        return np.inf
    diff = mu - Sigma @ b
    try:
        hj_dist = diff.T @ Sigma_inv @ diff
    except np.linalg.LinAlgError:
         # Handle cases where Sigma_inv might still be problematic
        print("Warning: HJ distance calculation failed (LinAlgError).")
        return np.inf # Return infinity to signal failure
    penalty = alpha * np.sum(np.abs(b))
    return hj_dist + penalty

# ------------------- Estimation Without CV -------------------
mu = np.nanmean(ret, axis=0)
# Calculate covariance matrix, handle potential NaN/inf values
if np.isnan(ret).any() or np.isinf(ret).any():
    print("Warning: NaN or Inf values found in excess returns before covariance calculation. Attempting to handle.")
    # Option 1: Impute (e.g., with mean/median) - Simple approach
    # ret = np.nan_to_num(ret, nan=np.nanmean(ret, axis=0))
    # Option 2: Drop rows/cols with NaNs (already done with DATA.dropna, but double-check)
    # Ensure no NaNs remain in mu
    if np.isnan(mu).any():
         print("Error: NaNs found in mean excess returns (mu). Cannot proceed.")
         exit()

try:
    Sigma = np.cov(ret, rowvar=False)
    # Check if Sigma is well-conditioned
    if np.linalg.cond(Sigma) > 1/np.finfo(Sigma.dtype).eps:
        print("Warning: Covariance matrix Sigma is ill-conditioned. Regularization might be needed or results unstable.")
        # Optional: Add regularization to Sigma (e.g., Sigma + reg_param * np.identity(Sigma.shape[0]))
    Sigma_inv = np.linalg.inv(Sigma)
except np.linalg.LinAlgError:
    print("Error: Covariance matrix Sigma is singular. Cannot compute inverse.")
    Sigma_inv = None # Set to None to handle in objective function
except ValueError:
    print("Error: Input contains NaN, infinity or is too large for dtype.")
    exit()


alpha_grid = np.logspace(-8, 0, 100) # LASSO penalties often smaller
results = []
obj_vals = np.empty(len(alpha_grid))
init_b = np.zeros_like(mu)

print("Starting estimation without CV...")
for i, alpha in enumerate(alpha_grid):
    if Sigma_inv is None:
        print(f"Skipping alpha = {alpha} due to singular Sigma.")
        obj_vals[i] = np.inf
        results.append(None) # Placeholder for failed optimization
        continue

    # Using SLSQP as it sometimes handles L1 better than BFGS, though not ideal.
    # Consider Coordinate Descent algorithms for robust LASSO.
    opt = minimize(lasso_objective, init_b, args=(mu, Sigma, Sigma_inv, alpha),
                   method='SLSQP', options={'maxiter': 1000, 'ftol': 1e-9}) # Increased maxiter and ftol
    results.append(opt)
    if opt.success:
        obj_vals[i] = opt.fun
    else:
        print(f"Warning: Optimization did not converge for alpha = {alpha}. Status: {opt.status}, Message: {opt.message}")
        obj_vals[i] = np.inf # Assign infinity if optimization failed

if np.all(np.isinf(obj_vals)):
    print("Error: All optimizations failed. Cannot determine best alpha.")
else:
    best_idx = np.nanargmin(obj_vals) # Use nanargmin to ignore inf values
    best_alpha = alpha_grid[best_idx]
    if results[best_idx] is not None:
        best_b = results[best_idx].x
        print(f"Optimal alpha (without CV): {best_alpha:.2e}")
        print(f"Objective value: {obj_vals[best_idx]:.6f}")
        coefficients = dict(zip(labels, np.round(best_b, 4)))
        print("Coefficients:")
        print(coefficients)

        # Create figure for combined plots
        fig, axs = plt.subplots(2, 1, figsize=(8, 10)) # 2 rows, 1 column

        # Plot for No CV on the first subplot
        axs[0].plot(np.log10(alpha_grid), obj_vals)
        axs[0].axvline(np.log10(best_alpha), color='red', linestyle='--')
        axs[0].set_title("Objective vs log10(alpha) - No CV")
        axs[0].set_xlabel("log10(alpha)")
        axs[0].set_ylabel("Objective Value")
        axs[0].set_ylim(np.nanmin(obj_vals), np.nanpercentile(obj_vals[np.isfinite(obj_vals)], 95)) # Adjust ylim
        axs[0].grid(True)
        # Removed plt.show() here
    else:
        print("Error: Best optimization result was None.")
        # Create figure even if first plot fails, to potentially show the second
        fig, axs = plt.subplots(2, 1, figsize=(8, 10))


# ------------------- Cross-Validation -------------------
def lasso_estimator(mu, Sigma, alpha):
    """Estimates LASSO coefficients using the objective function."""
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        print("Warning: Singular Sigma in lasso_estimator. Returning zeros.")
        return np.zeros(len(mu))

    # Initial guess
    init_b = np.zeros(len(mu))

    # Minimize the LASSO objective
    opt = minimize(lasso_objective, init_b, args=(mu, Sigma, Sigma_inv, alpha),
                   method='SLSQP', options={'maxiter': 1000, 'ftol': 1e-9})

    if not opt.success:
        print(f"Warning: Estimator optimization failed for alpha={alpha}. Returning zeros.")
        return np.zeros(len(mu))
    return opt.x

def cv_r2_oos_lasso(ret, alpha_grid, K=3):
    """Performs K-fold cross-validation for LASSO based on OOS R^2."""
    T, N = ret.shape
    if T < K:
        print(f"Warning: Number of samples ({T}) is less than K ({K}). Reducing K to {T}.")
        K = T
    if K <= 1:
        print("Error: K must be greater than 1 for cross-validation.")
        return {
            "best_alpha": np.nan, "best_r2": np.nan, "all_r2": np.full(len(alpha_grid), np.nan), "best_index": -1
        }

    kf = KFold(n_splits=K)
    r2_oos_values = np.full(len(alpha_grid), -np.inf) # Initialize with -inf

    print(f"\nStarting {K}-Fold Cross-Validation...")
    for g, alpha in enumerate(alpha_grid):
        r2_folds = []
        # print(f"  Testing alpha = {alpha:.2e}") # Reduce verbosity
        for fold, (train_index, test_index) in enumerate(kf.split(ret)):
            ret_train, ret_test = ret[train_index], ret[test_index]

            if ret_train.shape[0] < N or ret_test.shape[0] == 0: # Ensure enough samples
                # print(f"    Skipping fold {fold+1} for alpha {alpha:.2e}: Not enough samples in train/test split.")
                continue

            mu_train = np.nanmean(ret_train, axis=0)
            mu_test = np.nanmean(ret_test, axis=0)

            try:
                Sigma_train = np.cov(ret_train, rowvar=False)
                Sigma_test = np.cov(ret_test, rowvar=False)
                # Add checks for singularity/conditioning
                if np.linalg.cond(Sigma_train) > 1/np.finfo(Sigma_train.dtype).eps:
                     # print(f"    Warning: Sigma_train ill-conditioned in fold {fold+1} for alpha {alpha:.2e}.")
                     pass # Reduce verbosity
                if np.linalg.cond(Sigma_test) > 1/np.finfo(Sigma_test.dtype).eps:
                     # print(f"    Warning: Sigma_test ill-conditioned in fold {fold+1} for alpha {alpha:.2e}.")
                     pass # Reduce verbosity

                Sigma_train_inv = np.linalg.inv(Sigma_train) # Needed for estimator

            except np.linalg.LinAlgError:
                # print(f"    Skipping fold {fold+1} for alpha {alpha:.2e}: Singular covariance matrix.")
                continue
            except ValueError:
                 # print(f"    Skipping fold {fold+1} for alpha {alpha:.2e}: NaN/Inf in covariance input.")
                 continue


            b_hat = lasso_estimator(mu_train, Sigma_train, alpha)

            # Calculate OOS R^2
            pred = Sigma_test @ b_hat
            resid = mu_test - pred
            mu_test_norm_sq = mu_test.T @ mu_test
            if mu_test_norm_sq < 1e-12: # Avoid division by zero if test means are all zero
                 r2 = -np.inf if np.sum(resid**2) > 1e-12 else 1.0
            else:
                 r2 = 1 - (resid.T @ resid) / mu_test_norm_sq

            if np.isnan(r2):
                # print(f"    Warning: NaN R^2 encountered in fold {fold+1} for alpha {alpha:.2e}.")
                pass # Reduce verbosity
            else:
                r2_folds.append(r2)

        if r2_folds: # Only calculate mean if list is not empty
            r2_oos_values[g] = np.mean(r2_folds)
            # print(f"    Alpha {alpha:.2e}: Mean OOS R^2 = {r2_oos_values[g]:.4f}") # Reduce verbosity
        else:
            # print(f"    Alpha {alpha:.2e}: No valid folds completed.") # Reduce verbosity
            r2_oos_values[g] = -np.inf # Keep as -inf if no folds worked


    if np.all(np.isinf(r2_oos_values)) or np.all(np.isnan(r2_oos_values)):
         print("Error: Cross-validation failed for all alpha values.")
         best_idx = -1
         best_alpha_cv = np.nan
         best_r2_cv = np.nan
    else:
        best_idx = np.nanargmax(r2_oos_values) # Find index of max R^2, ignoring NaNs/-infs
        best_alpha_cv = alpha_grid[best_idx]
        best_r2_cv = r2_oos_values[best_idx]

    return {
        "best_alpha": best_alpha_cv,
        "best_r2": best_r2_cv,
        "all_r2": r2_oos_values,
        "best_index": best_idx
    }

# Run cross-validation
alpha_grid_cv = np.logspace(-8, 0, 50) # Grid for CV
cv_results = cv_r2_oos_lasso(ret, alpha_grid_cv, K=5) # Using K=5

if cv_results["best_index"] != -1:
    print(f"\nBest alpha (from CV): {cv_results['best_alpha']:.2e}")
    print(f"Best OOS R^2 (from CV): {cv_results['best_r2']:.4f}")

    # Plot for CV results on the second subplot
    valid_r2 = cv_results["all_r2"][np.isfinite(cv_results["all_r2"])]
    valid_alpha = alpha_grid_cv[np.isfinite(cv_results["all_r2"])]
    if len(valid_r2) > 0:
        axs[1].plot(np.log10(valid_alpha), valid_r2, marker='o')
        axs[1].axvline(np.log10(cv_results["best_alpha"]), color='blue', linestyle='--')
        axs[1].set_title("OOS R^2 vs log10(alpha) - Cross-Validation")
        axs[1].set_xlabel("log10(alpha)")
        axs[1].set_ylabel("Mean OOS R^2")
        axs[1].grid(True)
    else:
        print("No valid R^2 values to plot for CV.")
        axs[1].set_title("OOS R^2 vs log10(alpha) - Cross-Validation (No Data)")

    # Adjust layout and show the combined figure
    plt.tight_layout()
    plt.show()

    # Estimate final coefficients using the best alpha from CV
    print("\nEstimating final coefficients using best alpha from CV...")
    if Sigma_inv is not None:
        final_b = lasso_estimator(mu, Sigma, cv_results['best_alpha'])
        final_coefficients = dict(zip(labels, np.round(final_b, 4)))
        print("Final Coefficients (using CV best alpha):")
        print(final_coefficients)
    else:
        print("Cannot estimate final coefficients due to singular Sigma.")

else:
    print("\nCross-validation did not yield a best alpha.")

print("\nScript finished.")
