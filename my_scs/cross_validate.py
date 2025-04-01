import numpy as np
import pandas as pd
from scipy.linalg import pinv # Using scipy's pinv as in MATLAB example
import warnings
from math import sqrt
from .cvpartition_contiguous import cvpartition_contiguous # Import the actual function
from .regcov import regcov # Import the actual function

# Placeholder for tsmovavg - Needs proper implementation using pandas
def tsmovavg(data, window):
     """Placeholder for simple moving average using pandas."""
     if not isinstance(data, pd.DataFrame):
         # Assuming columns are time series
         data = pd.DataFrame(data)
     return data.rolling(window=window, min_periods=1).mean().to_numpy()


# --- Objective Functions ---

def bootstrp_obj_HJdist(y_hat, y, invX, phi, r, params):
    """GLS/HJdist objective: sqrt(alpha' * inv(Cov) * alpha * freq)"""
    alpha = y - y_hat
    # Ensure invX is valid before proceeding
    if invX is None or np.any(np.isnan(invX)) or np.any(np.isinf(invX)):
        return np.nan
    try:
        # Ensure vectors are column vectors (N, 1)
        alpha = alpha.reshape(-1, 1)
        term = alpha.T @ invX @ alpha
        if term < 0: # Handle potential numerical precision issues
             term = 0
        obj = sqrt(term.item() * params.get('freq', 1))
    except np.linalg.LinAlgError:
        obj = np.nan # Handle cases where invX might be singular despite pinv
    return obj

def bootstrp_obj_SSE(y_hat, y, invX, phi, r, params):
    """SSE objective: sqrt(mean(alpha^2) * freq)"""
    alpha = y - y_hat
    n_assets = len(y)
    if n_assets == 0:
        return np.nan
    obj = params.get('freq', 1) * sqrt(np.dot(alpha, alpha) / n_assets)
    return obj

def bootstrp_obj_CSR2(y_hat, y, invX, phi, r, params):
    """Cross-sectional R^2 objective: 1 - (alpha' * alpha) / (y' * y)"""
    alpha = y - y_hat
    y_ss = np.dot(y, y)
    if y_ss == 0:
        return np.nan # Avoid division by zero
    obj = 1 - np.dot(alpha, alpha) / y_ss
    return obj

def bootstrp_obj_GLSR2(y_hat, y, invX, phi, r, params):
    """Cross-sectional GLS R^2 objective: 1 - (alpha' * invX * alpha) / (y' * invX * y)"""
    alpha = y - y_hat
    # Ensure invX is valid
    if invX is None or np.any(np.isnan(invX)) or np.any(np.isinf(invX)):
        return np.nan
    try:
        # Ensure vectors are column vectors (N, 1)
        alpha = alpha.reshape(-1, 1)
        y_col = y.reshape(-1, 1)
        num = alpha.T @ invX @ alpha
        den = y_col.T @ invX @ y_col
        if den.item() <= 0: # Denominator must be positive
            return np.nan
        obj = 1 - num.item() / den.item()
    except np.linalg.LinAlgError:
        obj = np.nan
    return obj

def bootstrp_obj_SRexpl(y_hat, y, invX, phi, r, params):
    """Sharpe ratio explained objective: (SR_max^2 - SR_unexplained^2) * freq"""
    alpha = y - y_hat
    # Ensure invX is valid
    if invX is None or np.any(np.isnan(invX)) or np.any(np.isinf(invX)):
        return np.nan
    try:
        # Ensure vectors are column vectors (N, 1)
        alpha = alpha.reshape(-1, 1)
        y_col = y.reshape(-1, 1)
        sr_max_sq = y_col.T @ invX @ y_col
        sr_unexpl_sq = alpha.T @ invX @ alpha
        # Handle potential numerical precision issues
        if sr_max_sq < 0: sr_max_sq = 0
        if sr_unexpl_sq < 0: sr_unexpl_sq = 0

        obj = (sr_max_sq.item() - sr_unexpl_sq.item()) * params.get('freq', 1)
        # The original MATLAB code returns the difference, which can be negative.
        # It doesn't take sqrt here.
    except np.linalg.LinAlgError:
        obj = np.nan
    return obj


def bootstrp_obj_SR(y_hat, y, invX, phi, r, params):
    """Sharpe Ratio objective: sqrt(freq) * mean(portfolio_ret) / std(portfolio_ret)"""
    # The MATLAB code comments out a complex scaling part involving tsmovavg.
    # Implementing the simpler version: SR of the estimated portfolio r*phi
    portfolio_ret = r @ phi
    mean_ret = np.nanmean(portfolio_ret)
    std_ret = np.nanstd(portfolio_ret)
    if std_ret == 0 or np.isnan(std_ret) or np.isnan(mean_ret):
        return np.nan # Avoid division by zero or NaN result
    obj = sqrt(params.get('freq', 1)) * mean_ret / std_ret
    return obj

def bootstrp_obj_MVutil(y_hat, y, invX, phi, r, params):
    """Mean-Variance Utility objective"""
    # gamma = params.get('gamma', 1.0) # Assuming default gamma=1 if not provided
    # Using the log-utility version from the MATLAB code comments
    portfolio_ret = r @ phi
    pret_log = np.log(1 + portfolio_ret) # Calculate log returns

    # Check for invalid log returns (e.g., if 1+portfolio_ret <= 0)
    if np.any(np.isnan(pret_log)) or np.any(np.isinf(pret_log)):
        # Original MATLAB code returns -1e9 for invalid returns
        return -1e9

    mean_log_ret = np.mean(pret_log)
    var_log_ret = np.var(pret_log) # Use variance of log returns

    # The MATLAB comment mentions gamma=2 for risk aversion in log utility
    # obj = params.freq * (mean_log_ret - (gamma/2) * var_log_ret)
    # The code uses 2*0.5 = 1 as the coefficient for variance
    obj = params.get('freq', 1) * (mean_log_ret - 0.5 * var_log_ret)
    return obj


# --- Core Logic Handler ---

def bootstrp_handler(idx_test_bool, params):
    """Executes the core estimation and objective calculation logic."""

    # Map objective strings to functions
    objective_map = {
        'SSE': bootstrp_obj_SSE,
        'GLS': bootstrp_obj_HJdist, # GLS and HJdist seem equivalent here
        'CSR2': bootstrp_obj_CSR2,
        'GLSR2': bootstrp_obj_GLSR2,
        'SRexpl': bootstrp_obj_SRexpl,
        'SR': bootstrp_obj_SR,
        'MVU': bootstrp_obj_MVutil
    }
    objective_func = objective_map.get(params.get('objective', 'SSE'), bootstrp_obj_SSE)

    # Initialize parameters
    ret = params['ret']
    FUN = params['fun'] # The user-provided estimation function
    n_samples = ret.shape[0]
    idx_all = np.arange(n_samples)

    # Get training indices from the boolean test mask
    idx_train = idx_all[~idx_test_bool]
    idx_test = idx_all[idx_test_bool] # Use original numeric indices for slicing
    n_test = len(idx_test)

    res = [np.nan, np.nan] # Default result [IS, OOS]

    if n_test > 0 and len(idx_train) > 0: # Need both training and test samples
        # IS/OOS returns
        r_train = ret[idx_train, :]
        r_test = ret[idx_test, :]

        # Use cache if available for the current CV iteration
        cv_iter = params.get('cv_iteration', 0) # Assume 0 if not in CV loop
        cache = params.get('cv_cache', [])
        cvdata = None

        if len(cache) > cv_iter and cache[cv_iter] is not None:
            cvdata = cache[cv_iter]
        else:
            # Compute means and covariances if not cached
            cvdata = {}
            cvdata['y'] = np.mean(r_train, axis=0) # Mean for training data (IS)
            cvdata['X'] = regcov(r_train)         # Cov for training data (IS)
            cvdata['y_test'] = np.mean(r_test, axis=0) # Mean for test data (OOS)
            cvdata['X_test'] = regcov(r_test)      # Cov for test data (OOS)

            if params.get('objective') in ['GLS', 'GLSR2', 'SRexpl']:
                # Use pseudo-inverse for potentially singular matrices
                cvdata['invX'] = pinv(cvdata['X']) if cvdata['X'] is not None and not np.all(np.isnan(cvdata['X'])) else np.full_like(cvdata['X'], np.nan)
                cvdata['invX_test'] = pinv(cvdata['X_test']) if cvdata['X_test'] is not None and not np.all(np.isnan(cvdata['X_test'])) else np.full_like(cvdata['X_test'], np.nan)
            else:
                 cvdata['invX'] = None
                 cvdata['invX_test'] = None

            # Store in cache (ensure cache list is long enough)
            if 'cv_cache' not in params: params['cv_cache'] = []
            while len(params['cv_cache']) <= cv_iter:
                params['cv_cache'].append(None)
            params['cv_cache'][cv_iter] = cvdata

        # Extract data from cache/calculation
        X = cvdata['X']
        y = cvdata['y']
        X_test = cvdata['X_test']
        y_test = cvdata['y_test']
        invX = cvdata.get('invX') # Use .get for safety
        invX_test = cvdata.get('invX_test')

        # Check if covariance matrices are valid before proceeding
        if np.any(np.isnan(X)) or np.any(np.isnan(y)) or \
           np.any(np.isnan(X_test)) or np.any(np.isnan(y_test)):
             warnings.warn(f"NaN values encountered in means or covariances for fold {cv_iter}. Skipping.")
             return res, params # Return NaN results

        # Estimate the model using the provided function FUN
        # FUN is expected to take (Cov_train, mean_train, params)
        # and return (phi, updated_params) or potentially (phi, updated_params, se)
        try:
            estimator_result = FUN(X, y, params)
            if len(estimator_result) == 2:
                phi, params = estimator_result
            elif len(estimator_result) == 3:
                # If 3 values returned (like l2est), ignore the third (se)
                phi, params, _ = estimator_result
            else:
                raise ValueError(f"Estimator function FUN returned {len(estimator_result)} values, expected 2 or 3.")
        except Exception as e:
             warnings.warn(f"Error during model estimation (FUN) in fold {cv_iter}: {e}")
             return res, params # Return NaN results if estimation fails

        # Check if phi is valid
        if phi is None or np.any(np.isnan(phi)) or np.any(np.isinf(phi)):
             warnings.warn(f"Invalid 'phi' returned by estimation function in fold {cv_iter}. Skipping.")
             return res, params

        # Ensure phi is a column vector
        phi = phi.reshape(-1, 1)

        # Store results if not a cache run
        if not params.get('cache_run', False):
            if 'cv_phi' not in params: params['cv_phi'] = []
            while len(params['cv_phi']) <= cv_iter: params['cv_phi'].append(None)
            params['cv_phi'][cv_iter] = phi

            # Calculate OOS MVE portfolio returns and store
            oos_mve_returns = r_test @ phi
            if 'cv_MVE' not in params: params['cv_MVE'] = []
            while len(params['cv_MVE']) <= cv_iter: params['cv_MVE'].append(None)
            params['cv_MVE'][cv_iter] = oos_mve_returns

            # Estimate the IS/OOS MVE portfolio expected returns (y_hat)
            # y_hat = Cov * phi
            y_hat = X @ phi
            y_hat_test = X_test @ phi

            # Rescale if requested (ignore_scale=True)
            b = 1.0
            b_test = 1.0
            if params.get('ignore_scale', False):
                try:
                    # OLS: y = b * y_hat + error => b = (y_hat' * y_hat)^(-1) * y_hat' * y
                    # Using numpy.linalg.lstsq for robustness: solves y_hat * b = y
                    b_res = np.linalg.lstsq(y_hat, y, rcond=None)
                    b = b_res[0].item() if len(b_res[0]) > 0 else 1.0

                    b_test_res = np.linalg.lstsq(y_hat_test, y_test, rcond=None)
                    b_test = b_test_res[0].item() if len(b_test_res[0]) > 0 else 1.0

                except np.linalg.LinAlgError:
                    warnings.warn(f"OLS rescaling failed in fold {cv_iter}. Using b=1.")
                    b = 1.0
                    b_test = 1.0

            # Calculate IS and OOS objective values, suppressing potential runtime warnings
            # Ensure vectors passed to objective are 1D arrays
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                res_is = objective_func(y_hat.flatten() * b, y.flatten(), invX, phi.flatten(), r_train, params)
                res_oos = objective_func(y_hat_test.flatten() * b_test, y_test.flatten(), invX_test, phi.flatten(), r_test, params)
            res = [res_is, res_oos]

    return res, params


# --- Cross-Validation Method Handlers ---

def cross_validate_cv_handler(params):
    """Handler for k-fold cross-validation."""
    k = params.get('kfold', 2)
    ret = params['ret']
    n_samples = ret.shape[0]

    # Use placeholder for contiguous partitioning
    cv_partitions = cvpartition_contiguous(n_samples, k)

    obj_folds = np.full((k, 2), np.nan) # Store IS/OOS for each fold
    params['cv_idx_test'] = [] # Store test indices used

    for i in range(k):
        idx_test_bool = cv_partitions[i] # Get boolean mask for test set
        params['cv_idx_test'].append(np.where(idx_test_bool)[0]) # Store numeric indices
        params['cv_iteration'] = i # Pass current fold index (Corrected indent)

        # Call the core handler
        fold_res, params = bootstrp_handler(idx_test_bool, params)
        obj_folds[i, :] = fold_res

    # Average IS/OOS statistics across folds, suppressing RuntimeWarning for mean/std of empty slice
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_obj = np.nanmean(obj_folds, axis=0)
        # Calculate standard error of the mean
        # Adjust count for NaNs when calculating SE
        valid_counts = np.sum(~np.isnan(obj_folds), axis=0)
        # Avoid division by zero if a column is all NaNs
        std_err_obj = np.full_like(mean_obj, np.nan)
        for j in range(mean_obj.shape[0]): # Iterate through IS/OOS columns
            if valid_counts[j] > 1: # Need at least 2 points for std dev
                std_dev = np.nanstd(obj_folds[:, j], axis=0)
                std_err_obj[j] = std_dev / sqrt(valid_counts[j])
            elif valid_counts[j] == 1: # If only one valid point, SE is NaN or Inf? Set to NaN.
                 std_err_obj[j] = np.nan
            # else: # if valid_counts[j] == 0, std_err_obj[j] remains NaN

    # Result format: [mean_IS, mean_OOS, stderr_IS, stderr_OOS]
    # The MATLAB code returns [mean_IS, mean_OOS, std_IS/sqrt(k), std_OOS/sqrt(k)]
    # Let's match that format
    final_obj = np.concatenate((mean_obj, std_err_obj))

    # Return averaged objectives, updated params, and per-fold objectives
    return final_obj, params, obj_folds


def cross_validate_ssplit_handler(params):
    """Handler for single sample split."""
    splitdate_str = params.get('splitdate', '2000-01-01') # Use ISO format
    dates = params['dd'] # Assumes dates are comparable (e.g., datetime objects)

    try:
        # Convert split date string to datetime object
        split_datetime = pd.to_datetime(splitdate_str)
        # Ensure dates are also datetime objects if they aren't already
        if not isinstance(dates[0], (np.datetime64, pd.Timestamp)):
             dates_dt = pd.to_datetime(dates)
        else:
             dates_dt = dates
    except ValueError:
        raise ValueError(f"Invalid splitdate format: {splitdate_str}. Use YYYY-MM-DD or similar.")
    except IndexError:
         raise ValueError("Dates array appears to be empty.")


    # Create boolean mask for the test set (dates >= split_datetime)
    idx_test_bool = dates_dt >= split_datetime

    if not np.any(idx_test_bool):
        warnings.warn(f"Split date {splitdate_str} results in an empty test set.")
        return [np.nan, np.nan], params, None # Return NaN, params, no fold data
    if np.all(idx_test_bool):
         warnings.warn(f"Split date {splitdate_str} results in an empty training set.")
         return [np.nan, np.nan], params, None

    params['cv_iteration'] = 0 # Only one 'fold'
    obj, params = bootstrp_handler(idx_test_bool, params)

    # For sample split, there's only one result, no averaging needed.
    # Return IS/OOS result, updated params. obj_folds is None or the single result.
    return obj, params, np.array([obj]) if obj is not None else None


def cross_validate_bootstrap_handler(params):
    """Handler for bootstrap (marked as not implemented properly in MATLAB)."""
    # Mirroring the MATLAB comment and structure
    warnings.warn("Bootstrap method was marked 'Not implemented properly!' in original MATLAB code.")
    raise NotImplementedError("Bootstrap cross-validation handler is not fully implemented.")
    # If implementation was desired, it would involve resampling indices
    # with replacement and calling bootstrp_handler repeatedly.
    # Example structure (needs proper implementation):
    # n_iter = params.get('niter', 100)
    # n_samples = params['ret'].shape[0]
    # results = []
    # for _ in range(n_iter):
    #     idx_boot = np.random.choice(n_samples, n_samples, replace=True)
    #     # Need to define how train/test split works in bootstrap context
    #     # This part is unclear from the original code.
    #     # Assuming OOB (Out-of-Bag) for testing:
    #     idx_test_bool = np.ones(n_samples, dtype=bool)
    #     idx_test_bool[np.unique(idx_boot)] = False # OOB samples are test
    #     params['cv_iteration'] = _ # Pass iteration?
    #     res, params = bootstrp_handler(idx_test_bool, params)
    #     results.append(res)
    # obj = np.nanmean(results, axis=0)
    # return obj, params, np.array(results)


# --- Main Function ---

def cross_validate(FUN, dates, r, params=None):
    """
    Computes IS/OOS values of an objective function using cross-validation.

    Args:
        FUN (callable): Handle to a function which estimates model parameters.
                        Expected signature: FUN(Cov_train, mean_train, params) -> (phi, updated_params)
        dates (array-like): T x 1 vector of dates.
        r (np.ndarray): T x N matrix of returns.
        params (dict, optional): Dictionary containing extra arguments:
            - 'method' (str): 'CV' (default), 'ssplit', 'bootstrap'.
            - 'objective' (str): 'SSE' (default), 'GLS', 'CSR2', 'GLSR2', 'SR', 'MVU', 'SRexpl'.
            - 'ignore_scale' (bool): Rescale MVE portfolio for best fit. Default: False.
            - 'kfold' (int): Number of folds for 'CV'. Default: 2.
            - 'splitdate' (str): Date for 'ssplit' (e.g., 'YYYY-MM-DD'). Default: '2000-01-01'.
            - 'freq' (int): Annualization frequency (e.g., 12 for monthly). Default: 1.
            - 'gamma' (float): Risk aversion for 'MVU'. Default: (not specified, assumed in obj func).
            - Other parameters needed by FUN.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: [mean_IS, mean_OOS, stderr_IS, stderr_OOS] for CV,
                          or [IS, OOS] for ssplit.
            - dict: The updated params dictionary (may contain cached items).
            - np.ndarray or None: Per-fold results [IS, OOS] (k x 2 for CV, 1 x 2 for ssplit).
    """
    if not callable(FUN):
        raise TypeError("FUN argument must be a callable function handle.")
    if params is None:
        params = {}

    # Ensure returns are numpy array
    r = np.asarray(r)
    # Ensure dates are suitable for comparison (e.g., pandas Timestamps)
    try:
        # Convert dates to datetime objects if they are not already
        if not isinstance(dates, (pd.DatetimeIndex, np.ndarray)) or not np.issubdtype(dates.dtype, np.datetime64):
             dates_pd = pd.to_datetime(dates)
        else:
             dates_pd = dates # Assume already compatible
    except Exception as e:
         warnings.warn(f"Could not convert 'dates' to datetime objects: {e}. Ensure they are comparable.")
         dates_pd = dates # Proceed with original dates if conversion fails


    # Select cross-validation method handler
    method = params.get('method', 'CV')
    handler_map = {
        'CV': cross_validate_cv_handler,
        'ssplit': cross_validate_ssplit_handler,
        'bootstrap': cross_validate_bootstrap_handler
    }
    if method not in handler_map:
        raise ValueError(f"Unknown cross-validation method: {method}")

    selected_handler = handler_map[method]

    # Prepare params for the handler
    params['dd'] = dates_pd # Use potentially converted dates
    params['ret'] = r
    params['fun'] = FUN
    params.setdefault('freq', 1) # Default annualization frequency
    params.setdefault('ignore_scale', False)

    # Execute selected method
    obj, params, obj_folds = selected_handler(params)

    return obj, params, obj_folds

# Example Usage (Illustrative - requires a concrete FUN and data)
# if __name__ == '__main__':
#     # --- Dummy Data ---
#     T = 120 # Time periods
#     N = 10  # Assets
#     np.random.seed(0)
#     dummy_r = np.random.randn(T, N) * 0.05 + 0.005 # Monthly returns
#     dummy_dates = pd.date_range(start='2010-01-01', periods=T, freq='M')

#     # --- Dummy Estimation Function (e.g., simple mean) ---
#     def estimate_mean_portfolio(Cov_train, mean_train, params):
#         # Ignores Cov_train, just returns mean vector as 'phi'
#         phi = mean_train
#         # Ensure phi is a column vector
#         return phi.reshape(-1, 1), params

#     # --- Run Cross-Validation ---
#     cv_params = {
#         'method': 'CV',
#         'objective': 'SR', # Sharpe Ratio
#         'kfold': 5,
#         'freq': 12 # Annualize Sharpe Ratio
#     }

#     print("Running K-Fold CV...")
#     obj_cv, updated_params_cv, obj_folds_cv = cross_validate(
#         estimate_mean_portfolio, dummy_dates, dummy_r, cv_params
#     )
#     print("CV Results (Mean IS, Mean OOS, SE IS, SE OOS):", np.round(obj_cv, 4))
#     print("CV Per-Fold Results:\n", np.round(obj_folds_cv, 4))

#     # --- Run Sample Split ---
#     ssplit_params = {
#         'method': 'ssplit',
#         'objective': 'CSR2', # Cross-Sectional R2
#         'splitdate': '2015-01-01',
#         'freq': 1 # No annualization for R2
#     }
#     print("\nRunning Sample Split...")
#     obj_ss, updated_params_ss, obj_folds_ss = cross_validate(
#         estimate_mean_portfolio, dummy_dates, dummy_r, ssplit_params
#     )
#     print("Sample Split Results (IS, OOS):", np.round(obj_ss, 4))
