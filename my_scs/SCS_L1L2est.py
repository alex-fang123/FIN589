import numpy as np
import pandas as pd
import warnings
from math import sqrt, log10, floor, ceil, log
from scipy.linalg import pinv, svd
import datetime # Ensure datetime is imported

# Import previously translated functions
try:
    from .parse_config import parse_config
    from .datenum2 import datenum2 # Assuming datenum2 returns comparable objects like pd.Timestamp
    from .demarket import demarket
    # from .demarketcond import demarketcond # Missing
    # from .devolcond import devolcond # Missing
    from .regcov import regcov
    from .l2est import l2est
    from .cross_validate import cross_validate
    from .elasticnet_sdf_HJdist import elasticnet_sdf_HJdist
    from .anomnames import anomnames
    from .anomdescr import anomdescr # Needed for table_L2coefs
    from .plotting_utils import plot_dof, plot_L2coefpaths, plot_L2cv, plot_L1L2map, table_L2coefs # Import plotting utils
except ImportError:
    # Fallback for running script directly
    from parse_config import parse_config
    from datenum2 import datenum2
    from demarket import demarket
    from regcov import regcov
    from l2est import l2est
    from cross_validate import cross_validate
    from elasticnet_sdf_HJdist import elasticnet_sdf_HJdist
    from anomnames import anomnames
    from anomdescr import anomdescr
    from plotting_utils import plot_dof, plot_L2coefpaths, plot_L2cv, plot_L1L2map, table_L2coefs
    warnings.warn("Running SCS_L1L2est directly, ensure conditional functions (demarketcond, etc.) are handled if needed.")


def SCS_L1L2est(dates, re, market, freq, anomalies, p=None, verbose=False):
    """
    Python translation of SCS_L1L2est.m. Computes L2 and L1+L2 (Elastic Net)
    shrinkage estimators for SDF parameters based on Kozak, Nagel, Santosh (2019).

    Args:
        dates (array-like): T x 1 time series of dates (parsable by datenum2/pandas).
        re (np.ndarray): T x N matrix of excess returns time series.
        market (np.ndarray): T x 1 matrix of market's excess returns time series.
        freq (int): Number of observations per year (e.g., 12, 52, 252).
        anomalies (list): List of anomaly names (strings, N).
        p (dict, optional): Dictionary containing extra parameters. Defaults to None.
            See P_defaults below for options.
        verbose (bool, optional): Print progress messages. Defaults to False.


    Returns:
        dict: Dictionary 'estimates' containing results:
            - 'optimal_model_L2': Dict with L2 results (coefficients, objective, kappa).
            - 'optimal_model_L1L2': Dict with L1+L2 results (coefficients, objective, L1/L2 params).
            - 'coeffsPaths_L2': L2 coefficient paths (N x gridsize).
            - 'df_L2': Effective degrees of freedom for L2 path (gridsize,).
            - 'objL2_IS': In-sample objective for L2 path (gridsize,).
            - 'objL2_OOS': Out-of-sample objective for L2 path (gridsize,).
            - 'cv_test_L1L2': OOS objective grid for L1+L2 (nl x L1rn).
            - 'cv_test_se_L1L2': Std Err grid for L1+L2 OOS objective (nl x L1rn).
            - 'L1_grid': Grid used for L1 penalty (# non-zeros).
            - 'L2_grid_kappa': Grid used for L2 penalty (kappa values).
            - Various parameters used during estimation.
    """
    # --- Validate Arguments ---
    re = np.asarray(re)
    market = np.asarray(market)
    if np.any(np.isnan(re)):
         warnings.warn('Missing observations found in returns (re). Results may be unreliable.')
         # Consider adding imputation or row removal here if needed.

    # --- Default Parameters ---
    P_defaults = {
        'gridsize': 20,
        'method': 'CV', # CV method ('CV', 'ssplit')
        'objective': 'CSR2', # CV objective
        'ignore_scale': False,
        'kfold': 5,
        'oos_test_date': None,
        'freq': freq,
        'rotate_PC': False,
        'demarket_conditionally': False,
        'demarket_unconditionally': True,
        'devol_conditionally': False,
        'devol_unconditionally': True,
        'plot_dof': False,
        'plot_coefpaths': False,
        'plot_objective': False,
        'table_coefs': False,
        'L1_grid_size': 50,
    }
    if p is None: p = {}
    # Determine default last date *before* parse_config, store as Timestamp
    default_last_date_ts = None
    try:
        pd_dates = pd.to_datetime(dates)
        if not pd_dates.empty:
            default_last_date_ts = pd_dates[-1]
            P_defaults['oos_test_date'] = default_last_date_ts # Store Timestamp in defaults
    except Exception:
         warnings.warn("Could not determine last date. Default oos_test_date remains None.")
         # P_defaults['oos_test_date'] remains None

    # Apply defaults. If user provided oos_test_date=None, it might override the Timestamp default.
    p, _, _ = parse_config(p, P_defaults)
    if verbose: print(f"  Debug: Params *after* parse_config: oos_test_date='{p.get('oos_test_date')}' (type: {type(p.get('oos_test_date'))})")

    min_objectives = {'GLS', 'SSE'}
    optfunc = np.nanmin if p['objective'] in min_objectives else np.nanmax
    optfunc_idx = np.nanargmin if p['objective'] in min_objectives else np.nanargmax

    # --- Initialize; compute means, cov, SVD decomposition ---
    try:
        # Convert dates first (might be redundant if done above, but safe)
        numeric_dates = pd.to_datetime(dates)
        if verbose: print(f"  Debug: Numeric dates range: {numeric_dates.min()} to {numeric_dates.max()}")

        # Determine tT0 *after* parsing config, handling different types
        oos_test_date_param = p.get('oos_test_date')
        if verbose: print(f"  Debug: Effective oos_test_date from params: {oos_test_date_param} (type: {type(oos_test_date_param)})")

        # Explicitly check for None or empty string to use last date
        if oos_test_date_param is None or oos_test_date_param == '':
            tT0 = numeric_dates[-1]
            if verbose: print(f"  Debug: Using last date as tT0: {tT0}")
        elif isinstance(oos_test_date_param, (pd.Timestamp, datetime.datetime)):
            tT0 = pd.Timestamp(oos_test_date_param) # Ensure it's pandas Timestamp
            if verbose: print(f"  Debug: Using provided Timestamp as tT0: {tT0}")
        elif isinstance(oos_test_date_param, str) and oos_test_date_param != '':
            try:
                tT0 = pd.to_datetime(oos_test_date_param)
                if verbose: print(f"  Debug: Parsed oos_test_date string as tT0: {tT0}")
            except Exception as date_err:
                 warnings.warn(f"Could not parse oos_test_date string '{oos_test_date_param}': {date_err}. Defaulting to use all data for training.")
                 tT0 = numeric_dates[-1]
        else: # Handle other unexpected types or empty strings
             warnings.warn(f"Unexpected type or empty string for oos_test_date: {oos_test_date_param}. Defaulting to use all data for training.")
             tT0 = numeric_dates[-1]

    except Exception as e:
        raise ValueError(f"Error processing dates: {e}")

    if verbose: print(f"  Debug: Comparing numeric_dates (type: {type(numeric_dates)}, example: {numeric_dates[0]}) <= tT0 (type: {type(tT0)}, value: {tT0})")
    # Ensure comparison works even if tT0 is NaT (though unlikely now)
    try:
        # Ensure tT0 is valid before comparison
        if pd.isna(tT0):
            raise ValueError("tT0 is NaT (Not a Time), cannot perform date comparison.")
        idx_train = np.where(numeric_dates <= tT0)[0]
    except TypeError as te:
         raise ValueError(f"Date comparison failed. Check date types. numeric_dates type: {type(numeric_dates)}, tT0 type: {type(tT0)}. Error: {te}")
    idx_test = np.where(numeric_dates > tT0)[0]
    if verbose: print(f"  Debug: len(idx_train)={len(idx_train)}, len(idx_test)={len(idx_test)}")


    if len(idx_train) == 0:
        raise ValueError(f"Training set is empty based on oos_test_date '{p.get('oos_test_date')}'. Check date formats and comparison.")

    # --- Preprocessing ---
    r0 = re.copy()
    mkt0 = market.copy().reshape(-1, 1)

    if p['demarket_conditionally']:
        warnings.warn("Conditional de-marketing (demarketcond) is not implemented. Skipping.")
    elif p['demarket_unconditionally']:
        if verbose: print("Performing unconditional de-marketing...")
        try:
            r_train_dm, b_train = demarket(re[idx_train,:], mkt0[idx_train,:])
            r0[idx_train,:] = r_train_dm
            if len(idx_test) > 0:
                r_test_dm, _ = demarket(re[idx_test,:], mkt0[idx_test,:], b=b_train)
                r0[idx_test,:] = r_test_dm
            if verbose: print("De-marketing complete.")
        except Exception as e:
            warnings.warn(f"Unconditional de-marketing failed: {e}")

    if p['devol_conditionally']:
        warnings.warn("Conditional de-volatilization (devolcond) is not implemented. Skipping.")
    elif p['devol_unconditionally']:
        if verbose: print("Performing unconditional de-volatilization...")
        try:
            std_r0_train = np.nanstd(r0[idx_train,:], axis=0, ddof=1)
            std_mkt_train = np.nanstd(mkt0[idx_train], ddof=1)
            std_r0_train[std_r0_train < 1e-12] = 1.0
            if std_mkt_train < 1e-12: std_mkt_train = 1.0
            r0 = (r0 / std_r0_train) * std_mkt_train
            if verbose: print("De-volatilization complete.")
        except Exception as e:
            warnings.warn(f"Unconditional de-volatilization failed: {e}")


    dd = numeric_dates[idx_train]
    r_train = r0[idx_train,:]
    r_test = r0[idx_test,:] if len(idx_test) > 0 else np.empty((0, r0.shape[1]))

    T, n = r_train.shape
    p['T'] = T
    p['n'] = n
    if verbose: print(f"Training sample size: T={T}, N={n}")

    if p['rotate_PC']:
        if verbose: print("Rotating returns into PC space...")
        try:
            X_cov_pc = regcov(r_train)
            U_pc, s_pc, Vh_pc = svd(X_cov_pc)
            Q_pc = U_pc
            r_train = r_train @ Q_pc
            if len(idx_test) > 0: r_test = r_test @ Q_pc
            anomalies = [f'PC{i+1}' for i in range(n)]
            p['Q_pc'] = Q_pc
            if verbose: print("PC rotation complete.")
        except Exception as e:
            warnings.warn(f"PC rotation failed: {e}. Skipping rotation.")
            p['rotate_PC'] = False

    if verbose: print("Calculating final moments...")
    X = regcov(r_train)
    y = np.mean(r_train, axis=0)
    if verbose: print("Moment calculation complete.")

    try:
        Q_svd, s_svd, Vh_svd = svd(X)
        d_svd = s_svd
        p['Q_svd'] = Q_svd
        p['d_svd'] = d_svd
    except np.linalg.LinAlgError:
        warnings.warn("SVD of final training covariance failed.")
        p['Q_svd'], p['d_svd'], d_svd = None, None, None

    # --- L2 Grid Setup ---
    if verbose: print("Setting up L2 grid (kappa)...")
    def kappa2pen(kappa, T_local, X_local, params_local):
        kappa = np.asarray(kappa)
        if np.any(kappa <= 0): return np.inf
        trace_X = np.trace(X_local)
        if T_local <= 0 or trace_X <= 0: return np.inf
        return params_local.get('freq', 1) * trace_X / T_local / (kappa**2)

    test_kappas = 2.0**np.arange(21)
    test_pens = kappa2pen(test_kappas, T, X, p)
    z_coeffs = np.zeros((n, len(test_pens)))
    params_l2_test = p.copy()
    for i, pen in enumerate(test_pens):
        if np.isinf(pen): continue
        params_l2_test['L2pen'] = pen
        try:
             b_test, _, _ = l2est(X, y, params_l2_test, compute_errors=False)
             z_coeffs[:, i] = b_test
        except (ValueError, np.linalg.LinAlgError): pass

    rel_change = np.mean(np.abs(z_coeffs[:, 1:] - z_coeffs[:, :-1]) / (1 + np.abs(z_coeffs[:, :-1]) + 1e-9), axis=0) # Add epsilon for stability
    stable_idx = np.where(rel_change < 0.01)[0]
    kappa_high_idx = stable_idx[0] + 1 if len(stable_idx) > 0 else len(test_kappas) - 1
    kappa_high = test_kappas[min(kappa_high_idx, len(test_kappas)-1)] # Ensure index is valid
    kappa_low = 0.01

    L2_grid_kappa = np.logspace(log10(max(kappa_low, kappa_high)), log10(kappa_low), p['gridsize']) # Ensure high >= low
    l_grid = kappa2pen(L2_grid_kappa, T, X, p)
    cv_fold_factor = (1 - 1/p['kfold']) if p['kfold'] > 1 else 1.0
    l_grid_cv = l_grid / cv_fold_factor
    nl = len(l_grid)
    if verbose: print(f"L2 grid (kappa) from {kappa_low:.3f} to {kappa_high:.3f} ({nl} points).")

    # --- Estimate L2 Model Path ---
    if verbose: print("Estimating L2 path...")
    phi_L2 = np.full((n, nl), np.nan)
    se_L2 = np.full((n, nl), np.nan)
    objL2 = np.full((nl, 4), np.nan)
    objL2_folds = np.full((nl, p['kfold']), np.nan)
    MVE_L2_cv_folds = [[] for _ in range(nl)]

    params_cv_l2 = p.copy()
    params_cv_l2['objective'] = p['objective']
    params_cv_l2['method'] = p['method']
    params_cv_l2['kfold'] = p['kfold']
    params_cv_l2['ignore_scale'] = p['ignore_scale']

    for i in range(nl):
        params_l2_full = p.copy()
        params_l2_full['L2pen'] = l_grid[i]
        try:
            phi_L2[:, i], _, se_L2[:, i] = l2est(X, y, params_l2_full, compute_errors=True)
        except (ValueError, np.linalg.LinAlgError) as e:
            warnings.warn(f"l2est failed for full sample at L2 grid point {i} (kappa={L2_grid_kappa[i]:.3f}): {e}")
            continue

        params_cv_l2['L2pen'] = l_grid_cv[i]
        try:
            obj_agg, params_updated, obj_folds_raw = cross_validate(l2est, dd, r_train, params_cv_l2)
            objL2[i, :] = obj_agg
            if obj_folds_raw is not None and obj_folds_raw.shape == (p['kfold'], 2):
                 objL2_folds[i, :] = obj_folds_raw[:, 1]
            if 'cv_MVE' in params_updated and isinstance(params_updated['cv_MVE'], list):
                 MVE_L2_cv_folds[i] = params_updated.get('cv_MVE', [])
            params_cv_l2 = params_updated
        except (ValueError, np.linalg.LinAlgError, NotImplementedError) as e:
            warnings.warn(f"cross_validate failed for l2est at L2 grid point {i} (kappa={L2_grid_kappa[i]:.3f}): {e}")

    if verbose: print("L2 path estimation complete.")

    df_L2 = np.sum(d_svd**2 / (d_svd**2 + l_grid[:, np.newaxis]), axis=1) if d_svd is not None else np.full(nl, np.nan) # Ensure broadcasting

    objL2_OOS = objL2[:, 1]
    if np.all(np.isnan(objL2_OOS)):
         warnings.warn("All OOS objectives for L2 path are NaN. Cannot select optimal L2 model.")
         iL2opt = 0
         objL2opt = np.nan
    else:
         iL2opt = optfunc_idx(objL2_OOS)
         objL2opt = objL2_OOS[iL2opt]

    bL2opt = phi_L2[:, iL2opt]
    kappaL2opt = L2_grid_kappa[iL2opt]
    dfL2opt = df_L2[iL2opt] if not np.isnan(df_L2).all() else np.nan

    optimal_model_L2 = {
        'coefficients': bL2opt, 'se': se_L2[:, iL2opt], 'objective': objL2opt,
        'dof': dfL2opt, 'kappa': kappaL2opt
    }
    try:
        mve_returns_opt_l2 = []
        if iL2opt < len(MVE_L2_cv_folds) and MVE_L2_cv_folds[iL2opt]:
             mve_returns_opt_l2 = np.concatenate([fold_returns.flatten() for fold_returns in MVE_L2_cv_folds[iL2opt] if fold_returns is not None and fold_returns.size > 0])
        if len(mve_returns_opt_l2) > 1:
             sr_opt_l2 = np.mean(mve_returns_opt_l2) / np.std(mve_returns_opt_l2) * sqrt(p['freq'])
             optimal_model_L2['SR'] = sr_opt_l2
        else: optimal_model_L2['SR'] = np.nan
    except Exception as e:
        warnings.warn(f"Could not calculate SR for optimal L2 model: {e}")
        optimal_model_L2['SR'] = np.nan

    # --- Estimate L1+L2 Model Path (Elastic Net) ---
    if verbose: print("Estimating L1+L2 path (Elastic Net)...")
    L1_grid = np.unique(np.round(np.logspace(log10(1), log10(max(1,n-1)), p['L1_grid_size'])).astype(int))
    L1_grid = L1_grid[L1_grid > 0]
    L1rn = len(L1_grid)

    cv_test_L1L2 = np.full((nl, L1rn), np.nan)
    cv_test_se_L1L2 = np.full((nl, L1rn), np.nan)
    cv_folds_L1L2 = np.full((nl, L1rn, p['kfold']), np.nan)

    params_cache_en = p.copy()
    params_cache_en['cache_run'] = True
    params_cache_en['objective'] = p['objective']
    params_cache_en['method'] = p['method']
    params_cache_en['kfold'] = p['kfold']
    try:
        _, params_cached, _ = cross_validate(elasticnet_sdf_HJdist, dd, r_train, params_cache_en)
        params_cached['cache_run'] = False
        params_cached['use_precomputed'] = False
        if verbose: print("Pre-caching for Elastic Net CV complete.")
    except Exception as e:
        warnings.warn(f"Pre-caching for Elastic Net failed: {e}. Proceeding without cache.")
        params_cached = p.copy()

    for i in range(nl): # Outer loop over L2 grid
        if verbose: print(f"  L1+L2: Processing L2 grid point {i+1}/{nl} (kappa={L2_grid_kappa[i]:.3f})...")
        params_en_cv = params_cached.copy()
        params_en_cv['L2pen'] = l_grid_cv[i]
        params_en_cv['storepath'] = True # Store path within CV for efficiency

        for j in range(L1rn - 1, -1, -1): # Inner loop over L1 grid
            stop_val = -L1_grid[j]
            params_en_cv['stop'] = stop_val
            try:
                obj_agg, params_updated, obj_folds_raw = cross_validate(
                    elasticnet_sdf_HJdist, dd, r_train, params_en_cv
                )
                cv_test_L1L2[i, j] = obj_agg[1]
                cv_test_se_L1L2[i, j] = obj_agg[3]
                if obj_folds_raw is not None and obj_folds_raw.shape == (p['kfold'], 2):
                     cv_folds_L1L2[i, j, :] = obj_folds_raw[:, 1]
                params_en_cv = params_updated
                params_en_cv['use_precomputed'] = True # Use computed path for next j
            except (ValueError, np.linalg.LinAlgError, NotImplementedError) as e:
                 warnings.warn(f"cross_validate failed for elasticnet at L2={i}, L1={j}: {e}")

        params_cached['use_precomputed'] = False # Reset for next L2 iteration

    if verbose: print("L1+L2 path estimation complete.")

    # Optimal L1+L2 model selection
    optimal_model_L1L2 = {'objective': np.nan} # Default
    if not np.all(np.isnan(cv_test_L1L2)):
         opt_l2_indices = optfunc_idx(cv_test_L1L2, axis=0)
         opt_l2_values = np.array([cv_test_L1L2[opt_l2_indices[j], j] for j in range(L1rn)])
         cv_L1opt_idx = optfunc_idx(opt_l2_values)
         cv_L2opt_idx = opt_l2_indices[cv_L1opt_idx]

         objL1L2opt = cv_test_L1L2[cv_L2opt_idx, cv_L1opt_idx]
         kappaL1L2opt = L2_grid_kappa[cv_L2opt_idx]
         num_nonzeroL1L2opt = L1_grid[cv_L1opt_idx]

         params_final = p.copy()
         params_final['L2pen'] = l_grid[cv_L2opt_idx]
         params_final['stop'] = -num_nonzeroL1L2opt
         params_final['storepath'] = False
         try:
             # elasticnet_sdf_HJdist returns (b, params)
             bL1L2opt_path, _ = elasticnet_sdf_HJdist(X, y, params_final)
             # If path was returned (e.g., stop condition not met exactly), take last coefs
             bL1L2opt = bL1L2opt_path if bL1L2opt_path.ndim == 1 else bL1L2opt_path[:, -1]
         except Exception as e:
             warnings.warn(f"Final estimation of optimal L1L2 model failed: {e}")
             bL1L2opt = np.full(n, np.nan)

         optimal_model_L1L2 = {
             'coefficients': bL1L2opt, 'objective': objL1L2opt, 'kappa': kappaL1L2opt,
             'num_nonzero': num_nonzeroL1L2opt, 'L2_idx': cv_L2opt_idx, 'L1_idx': cv_L1opt_idx
         }
         try:
             M_opt_folds = cv_folds_L1L2[cv_L2opt_idx, cv_L1opt_idx, :]
             best_obj_per_fold = optfunc(optfunc(cv_folds_L1L2, axis=0), axis=0)
             valid_folds = ~np.isnan(M_opt_folds) & ~np.isnan(best_obj_per_fold)
             if np.any(valid_folds):
                 bias_L1L2 = best_obj_per_fold[valid_folds] - M_opt_folds[valid_folds]
                 optimal_model_L1L2['R2oos_bias'] = np.nanmean(bias_L1L2)
                 optimal_model_L1L2['R2oos_bias_se'] = np.nanstd(bias_L1L2) / sqrt(np.sum(valid_folds))
             else:
                 optimal_model_L1L2['R2oos_bias'] = np.nan
                 optimal_model_L1L2['R2oos_bias_se'] = np.nan
         except Exception as e:
             warnings.warn(f"Could not calculate R2oos bias: {e}")
             optimal_model_L1L2['R2oos_bias'] = np.nan
             optimal_model_L1L2['R2oos_bias_se'] = np.nan
    else:
        warnings.warn("All OOS objectives for L1+L2 grid are NaN. Cannot select optimal model.")


    # --- Prepare Output ---
    estimates = p.copy()
    estimates['optimal_model_L2'] = optimal_model_L2
    estimates['optimal_model_L1L2'] = optimal_model_L1L2
    estimates['coeffsPaths_L2'] = phi_L2
    estimates['df_L2'] = df_L2
    estimates['objL2_IS'] = objL2[:, 0]
    estimates['objL2_OOS'] = objL2[:, 1]
    estimates['objL2_OOS_SE'] = objL2[:, 3]
    estimates['objL2_folds'] = objL2_folds
    estimates['cv_test_L1L2'] = cv_test_L1L2
    estimates['cv_test_se_L1L2'] = cv_test_se_L1L2
    estimates['cv_folds_L1L2'] = cv_folds_L1L2
    estimates['L1_grid'] = L1_grid
    estimates['L2_grid_kappa'] = L2_grid_kappa
    estimates['anomalies'] = anomalies

    # Clean up large arrays from estimates dict
    keys_to_remove = ['ret', 'dd', 'fun', 'elasticnet_cache', 'bpath', 'cv_cache',
                      'cv_phi', 'cv_MVE', 'Q_svd', 'd_svd', 'Q_pc']
    for key in keys_to_remove:
        estimates.pop(key, None)

    if verbose: print("SCS_L1L2est finished.")

    # --- Plotting and Table Output (Optional) ---
    figures = [] # Initialize list to store figure objects
    # Note: Using L2 path results for L2-specific plots/tables
    if p.get('plot_dof', False) and not np.isnan(estimates.get('df_L2', np.nan)).all():
        try:
            fig_dof = plot_dof(estimates['df_L2'], estimates['L2_grid_kappa'], p)
            figures.append(fig_dof)
        except Exception as plot_e:
            warnings.warn(f"Failed to generate DoF plot: {plot_e}")

    if p.get('plot_coefpaths', False):
        try:
            # Plot L2 coefficients path
            fig_l2coeffs = plot_L2coefpaths(estimates['L2_grid_kappa'], estimates['coeffsPaths_L2'],
                             estimates.get('optimal_model_L2', {}).get('L2_idx', 0), # Use optimal L2 index if available
                             anomalies, r'SDF Coefficient, $b$ (L2 Path)', p)
            figures.append(fig_l2coeffs)
            # Plot L2 t-stats path (requires SEs which are not stored in estimates by default)
            # If SEs were stored in coeffsPaths_L2_se, you could plot them:
            # if 'coeffsPaths_L2_se' in estimates and not np.all(np.isnan(estimates['coeffsPaths_L2_se'])):
            #      with np.errstate(divide='ignore', invalid='ignore'):
            #          tstats_paths = estimates['coeffsPaths_L2'] / estimates['coeffsPaths_L2_se']
            #      fig_l2tstats = plot_L2coefpaths(estimates['L2_grid_kappa'], tstats_paths,
            #                       estimates.get('optimal_model_L2', {}).get('L2_idx', 0),
            #                       anomalies, r'SDF Coefficient $t$-statistic (L2 Path)', p)
            #      figures.append(fig_l2tstats)
        except Exception as plot_e:
            warnings.warn(f"Failed to generate L2 coefficient path plots: {plot_e}")

    if p.get('plot_objective', False):
        try:
            # Plot L2 CV objective
            objL2_plot = np.column_stack((
                estimates['objL2_IS'],
                estimates['objL2_OOS'],
                np.full_like(estimates['objL2_OOS'], np.nan), # Placeholder for IS SE if needed
                estimates['objL2_OOS_SE']
            ))
            fig_l2cv = plot_L2cv(estimates['L2_grid_kappa'], objL2_plot, p)
            figures.append(fig_l2cv)

            # Plot L1L2 Contour Map
            if 'cv_test_L1L2' in estimates and not np.all(np.isnan(estimates['cv_test_L1L2'])):
                 fig_l1l2map = plot_L1L2map(estimates['L2_grid_kappa'], estimates['L1_grid'],
                              estimates['cv_test_L1L2'], 'elasticnet_contour', p)
                 figures.append(fig_l1l2map)
        except Exception as plot_e:
            warnings.warn(f"Failed to generate objective plots: {plot_e}")

    if p.get('table_coefs', False):
        # table_L2coefs just prints, doesn't return a figure
        try:
            # Show table for the optimal L1+L2 model
            opt_model = estimates.get('optimal_model_L1L2')
            if opt_model and 'coefficients' in opt_model:
                 # table_L2coefs expects phi, se - L1L2 model doesn't store SE path by default
                 # We can show coefficients but maybe not t-stats unless SEs are calculated separately
                 warnings.warn("Standard errors for the optimal L1L2 model are not readily available for t-stat calculation in the table.")
                 table_L2coefs(opt_model['coefficients'], np.full_like(opt_model['coefficients'], np.nan), anomalies, p)
            else:
                 warnings.warn("Optimal L1+L2 model coefficients not found for table.")
        except Exception as table_e:
            warnings.warn(f"Failed to generate coefficients table: {table_e}")

    estimates['figures'] = figures # Add collected figures to output
    return estimates
