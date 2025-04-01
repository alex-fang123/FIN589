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
    # from .elasticnet_sdf_HJdist import elasticnet_sdf_HJdist # Removed for L2 only
    from .anomnames import anomnames
    from .anomdescr import anomdescr # Needed for table_L2coefs
    from .plotting_utils import plot_dof, plot_L2coefpaths, plot_L2cv, table_L2coefs # Import plotting utils
except ImportError:
    # Fallback for running script directly
    from parse_config import parse_config
    from datenum2 import datenum2
    from demarket import demarket
    from regcov import regcov
    from l2est import l2est
    from cross_validate import cross_validate
    # from elasticnet_sdf_HJdist import elasticnet_sdf_HJdist # Removed for L2 only
    from anomnames import anomnames
    from anomdescr import anomdescr
    from plotting_utils import plot_dof, plot_L2coefpaths, plot_L2cv, table_L2coefs
    warnings.warn("Running SCS_L2est directly, ensure conditional functions (demarketcond, etc.) are handled if needed.")


def SCS_L2est(dates, re, market, freq, anomalies, p=None, verbose=False):
    """
    Python translation of SCS_L2est.m. Computes L2 (Ridge)
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
            - 'optimal_model_L2': Dict with L2 results (coefficients, se, objective, dof, kappa, SR).
            - 'coeffsPaths_L2': L2 coefficient paths (N x gridsize).
            - 'sePaths_L2': Standard error paths for L2 coefficients (N x gridsize).
            - 'df_L2': Effective degrees of freedom for L2 path (gridsize,).
            - 'objL2_IS': In-sample objective for L2 path (gridsize,).
            - 'objL2_OOS': Out-of-sample objective for L2 path (gridsize,).
            - 'objL2_OOS_SE': Standard error for OOS objective (gridsize,).
            - 'objL2_folds': OOS objective value for each fold on L2 path (gridsize x kfold).
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
        # 'L1_grid_size': 50, # Removed L1 parameter
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

        if oos_test_date_param is None:
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

    # --- Prepare Output ---
    estimates = p.copy()
    estimates['optimal_model_L2'] = optimal_model_L2
    # estimates['optimal_model_L1L2'] = optimal_model_L1L2 # Removed L1L2
    estimates['coeffsPaths_L2'] = phi_L2
    estimates['sePaths_L2'] = se_L2 # Add SE path
    estimates['df_L2'] = df_L2
    estimates['objL2_IS'] = objL2[:, 0]
    estimates['objL2_OOS'] = objL2[:, 1]
    estimates['objL2_OOS_SE'] = objL2[:, 3]
    estimates['objL2_folds'] = objL2_folds
    # estimates['cv_test_L1L2'] = cv_test_L1L2 # Removed L1L2
    # estimates['cv_test_se_L1L2'] = cv_test_se_L1L2 # Removed L1L2
    # estimates['cv_folds_L1L2'] = cv_folds_L1L2 # Removed L1L2
    # estimates['L1_grid'] = L1_grid # Removed L1L2
    estimates['L2_grid_kappa'] = L2_grid_kappa
    estimates['anomalies'] = anomalies

    # Clean up large arrays from estimates dict
    keys_to_remove = ['ret', 'dd', 'fun', 'elasticnet_cache', 'bpath', 'cv_cache',
                      'cv_phi', 'cv_MVE', 'Q_svd', 'd_svd', 'Q_pc']
    for key in keys_to_remove:
        estimates.pop(key, None)

    if verbose: print("SCS_L2est finished.") # Updated message

    # --- Plotting and Table Output (Optional) ---
    figures = [] # Initialize list to store figure objects
    if p.get('plot_dof', False) and not np.isnan(df_L2).all():
        try:
            fig_dof = plot_dof(df_L2, L2_grid_kappa, p)
            figures.append(fig_dof)
        except Exception as plot_e:
            warnings.warn(f"Failed to generate DoF plot: {plot_e}")

    if p.get('plot_coefpaths', False):
        try:
            # Plot coefficients
            fig_coeffs = plot_L2coefpaths(L2_grid_kappa, phi_L2, iL2opt, anomalies, r'SDF Coefficient, $b$', p)
            figures.append(fig_coeffs)
            # Plot t-stats (if SEs are available)
            if not np.all(np.isnan(se_L2)):
                 with np.errstate(divide='ignore', invalid='ignore'):
                     tstats_paths = phi_L2 / se_L2
                 fig_tstats = plot_L2coefpaths(L2_grid_kappa, tstats_paths, iL2opt, anomalies, r'SDF Coefficient $t$-statistic', p)
                 figures.append(fig_tstats)
        except Exception as plot_e:
            warnings.warn(f"Failed to generate coefficient path plots: {plot_e}")

    if p.get('plot_objective', False):
        try:
            fig_cv = plot_L2cv(L2_grid_kappa, objL2, p)
            figures.append(fig_cv)
        except Exception as plot_e:
            warnings.warn(f"Failed to generate objective plot: {plot_e}")

    if p.get('table_coefs', False):
        # table_L2coefs just prints, doesn't return a figure
        try:
            table_L2coefs(optimal_model_L2['coefficients'], optimal_model_L2['se'], anomalies, p)
        except Exception as table_e:
            warnings.warn(f"Failed to generate coefficients table: {table_e}")

    estimates['figures'] = figures # Add collected figures to output
    return estimates
