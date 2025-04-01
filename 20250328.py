from my_scs import test
from my_scs import load_ff25
from my_scs import load_ff_anomalies
from my_scs import load_managed_portfolios

test.hello()

# Load some data to potentially use in tests
try:
    print("\n--- Loading FF25 Data ---")
    # Assuming load_ff25 returns dates, returns, market, other_data, labels
    ff_dates, ff_re, ff_mkt, ff_DATA, ff_labels = load_ff25.load_ff25('./my_scs/Data/', daily=True)
    print("FF25 Data loaded successfully.")
    ff_freq = 252
    ff_anomalies = ff_labels # Use labels as anomalies for testing
except Exception as e:
    print(f"Error loading FF25 data: {e}")
    ff_dates, ff_re, ff_mkt, ff_DATA, ff_labels, ff_anomalies = None, None, None, None, None, None
    ff_freq = 252 # Assume daily freq even if load fails

try:
    print("\n--- Loading Managed Portfolio Data ---")
    # Assuming load_managed_portfolios returns dates, returns, market, names, other_data
    mp_dates, mp_re, mp_mkt, mp_names, mp_DATA = load_managed_portfolios.load_managed_portfolios(
        "./my_scs/Data/Instruments/managed_portfolios_anom_d_50.csv", daily=True)
    print("Managed Portfolio Data loaded successfully.")
    mp_freq = 252
    mp_anomalies = mp_names
except Exception as e:
    print(f"Error loading Managed Portfolio data: {e}")
    mp_dates, mp_re, mp_mkt, mp_names, mp_anomalies = None, None, None, None, None
    mp_freq = 252 # Assume daily freq

# Use managed portfolio data if available, otherwise fallback to FF25
test_dates = mp_dates if mp_dates is not None else ff_dates
test_re = mp_re if mp_re is not None else ff_re
test_mkt = mp_mkt if mp_mkt is not None else ff_mkt
test_anomalies = mp_anomalies if mp_anomalies is not None else ff_anomalies
test_freq = mp_freq if mp_dates is not None else ff_freq

print(f"\nUsing data for tests (shape {test_re.shape if test_re is not None else 'N/A'})")

# --- Import Newly Translated Modules ---
import numpy as np
import pandas as pd
import datetime
import warnings # Import warnings module
import matplotlib.pyplot as plt # Import matplotlib
from my_scs import anomnames
from my_scs import check_params
from my_scs import choldelete
from my_scs import cholinsert
from my_scs import cross_validate
from my_scs import cvpartition_contiguous
from my_scs import datenum2
from my_scs import demarket
from my_scs import elasticnet_sdf_HJdist
from my_scs import l2est
from my_scs import larsen
from my_scs import parse_config
from my_scs import regcov
from my_scs import SCS_L2est
from my_scs import SCS_L1L2est
from my_scs import anomdescr
# characteristics_names_map is imported within anomdescr

print("\n--- Testing Translated Functions ---")

# --- Test anomnames ---
try:
    print("\nTesting anomnames...")
    sample_codes = ['rme', 're_size', 'r2_value', 'rX_prof_inv', 'r_beta', 'other_code']
    formatted = anomnames.anomnames(sample_codes)
    print(f"  Input: {sample_codes}")
    print(f"  Output: {formatted}")
    print("  anomnames OK")
except Exception as e:
    print(f"  anomnames FAILED: {e}")

# --- Test check_params ---
try:
    print("\nTesting check_params...")
    user_p = {'a': 1, 'c': 30}
    def_p = {'a': 10, 'b': 20, 'c': 300, 'd': 40}
    req_p = ['a']
    final_p, over_p, dflt_p = check_params.check_params(user_p, def_p, req_p)
    print(f"  Input: s={user_p}, defaults={def_p}, required={req_p}")
    print(f"  Output: final={final_p}, overridden={over_p}, defaulted={dflt_p}")
    # Test missing required
    user_p_missing = {'c': 30}
    try:
        check_params.check_params(user_p_missing, def_p, req_p)
        print("  check_params FAILED (did not raise error for missing required)")
    except ValueError as ve:
        print(f"  Caught expected error for missing required: {ve}")
        print("  check_params OK")
except Exception as e:
    print(f"  check_params FAILED: {e}")

# --- Test Cholesky functions ---
try:
    print("\nTesting cholinsert/choldelete...")
    # Create a small positive definite matrix A and its Cholesky factor R
    np.random.seed(1)
    X_chol = np.random.rand(10, 4)
    A_chol = X_chol.T @ X_chol + 0.1 * np.identity(4) # Add regularization
    R_chol = np.linalg.cholesky(A_chol).T # Upper Cholesky
    print(f"  Original R shape: {R_chol.shape}")

    # Test cholinsert
    x_ins = np.random.rand(10, 1)
    R_ins = cholinsert.cholinsert(R_chol, x_ins, X_chol, delta=0.1)
    print(f"  cholinsert output shape: {R_ins.shape} (Expected: {(5, 5)})")

    # Test choldelete (using the inserted matrix's factor)
    R_del = choldelete.choldelete(R_ins, j=2) # Delete 3rd column (index 2)
    print(f"  choldelete output shape: {R_del.shape} (Expected: {(4, 4)})")
    print("  cholinsert/choldelete OK (basic run)")
except Exception as e:
    print(f"  cholinsert/choldelete FAILED: {e}")

# --- Test cvpartition_contiguous ---
try:
    print("\nTesting cvpartition_contiguous...")
    n_cv = 15
    k_cv = 4
    masks = cvpartition_contiguous.cvpartition_contiguous(n_cv, k_cv)
    print(f"  Generated {len(masks)} masks for n={n_cv}, k={k_cv}")
    print(f"  Mask shapes: {[m.shape for m in masks]}")
    print(f"  Samples per fold: {[np.sum(m) for m in masks]}")
    print("  cvpartition_contiguous OK")
except Exception as e:
    print(f"  cvpartition_contiguous FAILED: {e}")

# --- Test datenum2 ---
try:
    print("\nTesting datenum2...")
    dt_obj = datetime.datetime(2024, 3, 15, 10, 30)
    dt_str = "2024-03-16"
    dt_ymd = (2024, 3, 17)
    print(f"  Input datetime obj: {datenum2.datenum2(dt_obj)}")
    print(f"  Input str: {datenum2.datenum2(dt_str)}")
    print(f"  Input tuple: {datenum2.datenum2(*dt_ymd)}")
    print("  datenum2 OK")
except Exception as e:
    print(f"  datenum2 FAILED: {e}")

# --- Test demarket ---
try:
    print("\nTesting demarket...")
    if test_re is not None and test_mkt is not None and test_re.shape[1] >= 5:
        rme, b = demarket.demarket(test_re[:, :5], test_mkt) # Use first 5 assets
        print(f"  rme shape: {rme.shape}")
        print(f"  b shape: {b.shape}")
        print("  demarket OK")
    else:
        print("  Skipping demarket test (missing or insufficient data).")
except Exception as e:
    print(f"  demarket FAILED: {e}")

# --- Test regcov ---
try:
    print("\nTesting regcov...")
    if test_re is not None and test_re.shape[1] >= 5:
        X_reg = regcov.regcov(test_re[:, :5]) # Use first 5 assets
        print(f"  Regularized Cov shape: {X_reg.shape}")
        print("  regcov OK")
    else:
        print("  Skipping regcov test (missing or insufficient data).")
except Exception as e:
    print(f"  regcov FAILED: {e}")

# --- Prepare data for estimator tests ---
X_est, y_est = None, None
if test_re is not None:
    try:
        print("\nPreparing data for estimator tests...")
        # Use a small subset of assets for faster testing
        n_assets_test = 10
        if test_re.shape[1] < n_assets_test:
             n_assets_test = test_re.shape[1]
        re_subset = test_re[:, :n_assets_test]
        X_est = regcov.regcov(re_subset)
        y_est = np.mean(re_subset, axis=0)
        # Handle potential NaNs from regcov if T is small
        if np.any(np.isnan(X_est)) or np.any(np.isnan(y_est)):
            print("  NaNs found in X_est or y_est, skipping estimator tests.")
            X_est, y_est = None, None
        else:
            print(f"  Using X_est shape {X_est.shape}, y_est shape {y_est.shape}")
    except Exception as e:
        print(f"  Data preparation failed: {e}")
        X_est, y_est = None, None

# --- Test l2est ---
try:
    print("\nTesting l2est...")
    if X_est is not None and y_est is not None and test_re is not None:
        params_l2 = {'L2pen': 0.1, 'T': test_re.shape[0]}
        b_l2, _, se_l2 = l2est.l2est(X_est, y_est, params_l2, compute_errors=True)
        print(f"  b shape: {b_l2.shape}")
        print(f"  se shape: {se_l2.shape}")
        print("  l2est OK")
    else:
        print("  Skipping l2est test (missing data).")
except Exception as e:
    print(f"  l2est FAILED: {e}")

# --- Test elasticnet_sdf_HJdist ---
try:
    print("\nTesting elasticnet_sdf_HJdist...")
    if X_est is not None and y_est is not None:
        params_en = {'L2pen': 0.01, 'stop': -5, 'storepath': False, 'verbose': False} # Stop at 5 vars
        # Need to ensure cache lists exist if function expects them
        params_en['elasticnet_cache'] = []
        params_en['bpath'] = []
        b_en, p_en_out = elasticnet_sdf_HJdist.elasticnet_sdf_HJdist(X_est, y_est, params_en)
        print(f"  b shape: {b_en.shape}")
        print(f"  Non-zeros: {np.sum(np.abs(b_en) > 1e-9)}")
        print("  elasticnet_sdf_HJdist OK")
    else:
        print("  Skipping elasticnet_sdf_HJdist test (missing data).")
except Exception as e:
    print(f"  elasticnet_sdf_HJdist FAILED: {e}")

# --- Test larsen ---
try:
    print("\nTesting larsen...")
    # larsen needs X (n x p), y (n,)
    # Use simpler dummy data for direct larsen test
    np.random.seed(2)
    n_lars, p_lars = 50, 10
    X_lars = np.random.randn(n_lars, p_lars)
    y_lars = X_lars[:, 0] * 2 - X_lars[:, 2] * 1.5 + np.random.randn(n_lars) * 0.5
    delta_lars = 0.1
    stop_lars = -3 # Stop at 3 vars
    b_lars_path, steps_lars = larsen.larsen(X_lars, y_lars, delta_lars, stop=stop_lars, storepath=True, verbose=False)
    print(f"  Path shape: {b_lars_path.shape}")
    print(f"  Steps taken: {steps_lars}")
    print(f"  Final b non-zeros: {np.sum(np.abs(b_lars_path[:, -1]) > 1e-9)}")
    print("  larsen OK")
except Exception as e:
    print(f"  larsen FAILED: {e}")

# --- Test parse_config ---
try:
    print("\nTesting parse_config...")
    user_c = {'param_a': 5}
    def_c = {'param_a': 10, 'param_b': 'hello'}
    final_c, over_c, dflt_c = parse_config.parse_config(user_c, def_c)
    print(f"  Input: cfg={user_c}, CFG={def_c}")
    print(f"  Output: final={final_c}, overridden={over_c}, defaulted={dflt_c}")
    print("  parse_config OK")
except Exception as e:
    print(f"  parse_config FAILED: {e}")

# --- Test cross_validate ---
try:
    print("\nTesting cross_validate...")
    if test_dates is not None and test_re is not None and X_est is not None and y_est is not None:
        # Define a simple dummy estimation function for testing CV
        def dummy_estimator(X_cv, y_cv, params_cv):
            # Example: just return ridge solution
            b_cv, _, _ = l2est.l2est(X_cv, y_cv, params_cv, compute_errors=False)
            return b_cv.reshape(-1, 1), params_cv # Return column vector

        # Ensure r_train exists and has enough samples for kfold=3
        r_train_cv = test_re[:len(test_dates), :X_est.shape[1]] # Use subset matching X_est dim
        dates_cv = test_dates[:len(test_dates)]
        if len(dates_cv) > 5: # Need enough samples for CV
            params_cv = {'L2pen': 0.1, 'T': r_train_cv.shape[0], 'freq': test_freq,
                         'objective': 'CSR2', 'method': 'CV', 'kfold': 3}
            cv_dates_pd = pd.to_datetime(dates_cv) # Match length
            obj_cv, p_cv_out, obj_folds_cv = cross_validate.cross_validate(
                dummy_estimator, cv_dates_pd, r_train_cv, params_cv
            )
            print(f"  CV Results (Mean IS, Mean OOS, SE IS, SE OOS): {obj_cv}")
            print(f"  CV Fold results shape: {obj_folds_cv.shape if obj_folds_cv is not None else 'N/A'}")
            print("  cross_validate OK")
        else:
            print("  Skipping cross_validate test (not enough data samples).")
    else:
        print("  Skipping cross_validate test (missing data).")
except Exception as e:
    print(f"  cross_validate FAILED: {e}")

# --- Test anomdescr ---
try:
    print("\nTesting anomdescr...")
    # Requires characteristics_names_map to be working
    from my_scs import characteristics_names_map
    # Check if the map function returns a dict (basic check)
    if isinstance(characteristics_names_map.characteristics_names_map(), dict):
        sample_codes_desc = ['rme', 're_size', 'r2_value', 'rX_prof_inv', 'r_beta', 'unknown_code']
        # Suppress warnings during this specific test as 'unknown_code' is expected to warn
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            descriptions = anomdescr.anomdescr(sample_codes_desc)
        print(f"  Input: {sample_codes_desc}")
        print(f"  Output: {descriptions}")
        print("  anomdescr OK")
    else:
        print("  Skipping anomdescr test (characteristics_names_map did not return dict).")
except ImportError:
     print("  Skipping anomdescr test (characteristics_names_map not found).")
except Exception as e:
    print(f"  anomdescr FAILED: {e}")

# --- Test SCS_L2est ---
try:
    print("\nTesting SCS_L2est...")
    if test_dates is not None and test_re is not None and test_mkt is not None and test_anomalies is not None:
        # Enable plotting flags for SCS_L2est
        params_scs = {'gridsize': 5, 'kfold': 2, 'objective': 'CSR2',
                      'plot_dof': True, 'plot_coefpaths': True, 'plot_objective': True, 'table_coefs': True}
        estimates_l2 = SCS_L2est.SCS_L2est(test_dates, test_re, test_mkt, test_freq, test_anomalies, params_scs, verbose=False)
        print(f"  SCS_L2est returned keys: {list(estimates_l2.keys())}")
        print(f"  Optimal L2 Kappa: {estimates_l2.get('optimal_model_L2', {}).get('kappa')}")
        print("  SCS_L2est OK")
    else:
        print("  Skipping SCS_L2est test (missing data).")
except Exception as e:
    print(f"  SCS_L2est FAILED: {e}")

# --- Test SCS_L1L2est ---
try:
    print("\nTesting SCS_L1L2est...")
    if test_dates is not None and test_re is not None and test_mkt is not None and test_anomalies is not None:
        # Enable plotting flags for SCS_L1L2est
        params_scs12 = {'gridsize': 5, 'kfold': 2, 'objective': 'CSR2', 'L1_grid_size': 5,
                        'plot_dof': True, 'plot_coefpaths': True, 'plot_objective': True, 'table_coefs': True}
        estimates_l1l2 = SCS_L1L2est.SCS_L1L2est(test_dates, test_re, test_mkt, test_freq, test_anomalies, params_scs12, verbose=False)
        print(f"  SCS_L1L2est returned keys: {list(estimates_l1l2.keys())}")
        print(f"  Optimal L1L2 Kappa: {estimates_l1l2.get('optimal_model_L1L2', {}).get('kappa')}")
        print(f"  Optimal L1L2 Nonzero: {estimates_l1l2.get('optimal_model_L1L2', {}).get('num_nonzero')}")
        print("  SCS_L1L2est OK")
    else:
        print("  Skipping SCS_L1L2est test (missing data).")
except Exception as e:
    print(f"  SCS_L1L2est FAILED: {e}")

print("\n--- Function Testing Complete ---")

# --- Save Figures to PDF ---
all_figures = []
if 'estimates_l2' in locals() and 'figures' in estimates_l2:
    all_figures.extend(estimates_l2['figures'])
if 'estimates_l1l2' in locals() and 'figures' in estimates_l1l2:
    all_figures.extend(estimates_l1l2['figures'])

if all_figures:
    pdf_filename = "scs_plots.pdf"
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(pdf_filename) as pdf:
            for fig in all_figures:
                pdf.savefig(fig)
                plt.close(fig) # Close figure after saving to PDF
        print(f"\nAll generated plots saved to {pdf_filename}")
    except Exception as pdf_e:
        print(f"\nError saving plots to PDF: {pdf_e}")
        # Optionally show plots if saving failed
        # print("Attempting to show plots instead...")
        # plt.show()
else:
    print("\nNo figures were generated to save.")

# plt.show() # Removed - plots are now saved to PDF file
