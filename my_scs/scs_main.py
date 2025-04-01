import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime
import time
import warnings

# Import previously translated functions and assume data loaders exist
try:
    from .parse_config import parse_config
    from .datenum2 import datenum2 # Or rely on pandas directly
    # Assume data loaders return (dates, returns, market_return, labels/anomalies, Optional[other_data])
    from .load_ff_anomalies import load_ff_anomalies
    from .load_ff25 import load_ff25
    from .load_managed_portfolios import load_managed_portfolios
    from .SCS_L2est import SCS_L2est
    from .SCS_L1L2est import SCS_L1L2est
except ImportError:
    # Fallback for running script directly
    from parse_config import parse_config
    from datenum2 import datenum2
    from load_ff_anomalies import load_ff_anomalies
    from load_ff25 import load_ff25
    from load_managed_portfolios import load_managed_portfolios
    from SCS_L2est import SCS_L2est
    from SCS_L1L2est import SCS_L1L2est
    warnings.warn("Running scs_main directly. Ensure data loaders are available.")

def main():
    """
    Main execution script translated from scs_main.m.
    """
    start_time = time.time()
    print("Starting SCS analysis...")

    # --- Options ---
    daily = True
    interactions = False # Set to True to include interaction terms (requires specific data)
    rotate_PC = False # Set to True to rotate returns into PC space
    withhold_test_sample = False # Set to True to use a separate OOS test period

    # Data provider ('anom' or 'ff25')
    dataprovider = 'anom'
    # dataprovider = 'ff25'

    # Sample dates (use YYYY-MM-DD format for pandas compatibility)
    t0_str = '1963-07-01'
    tN_str = '2017-12-31'
    oos_test_date_str = '2005-01-01' # Used if withhold_test_sample is True

    # Output folder (optional, for saving results/figures)
    # run_folder = datetime.today().strftime('%d%b%y').upper() + '/'
    # os.makedirs(run_folder, exist_ok=True)

    # --- Paths ---
    # Adjust these paths as needed
    projpath = './' # Assumes running from the directory containing my_scs
    datapath = os.path.join(projpath, 'Data/')
    instrpath = os.path.join(datapath, 'instruments/')

    # --- Initialize ---
    if daily:
        freq = 252
        suffix = '_d'
        # date_fmt = '%m/%d/%Y' # Example format if needed by loaders
    else:
        freq = 12
        suffix = ''
        # date_fmt = '%m/%Y'

    # Fix random number generation (less critical in Python unless stochastic parts added)
    np.random.seed(0) # Example seed

    # --- Default Estimation Parameters ---
    default_params = {
        'gridsize': 100,
        'contour_levelstep': 0.01,
        'objective': 'CSR2',
        'rotate_PC': rotate_PC, # Pass main script option
        'devol_unconditionally': False, # Default to False, override later based on data
        'kfold': 3,
        'plot_dof': False, # Plotting disabled in translation
        'plot_coefpaths': False,
        'plot_objective': False,
        'table_coefs': False, # Table output disabled
        # Figure options omitted
        'L1_grid_size': 50, # Default for L1L2 run
    }

    # --- Load FF Factors (if needed globally, though better to pass explicitly) ---
    # print("Loading FF factors...")
    # try:
    #     # Assuming load_ff_anomalies returns dates, factor_returns, market_return
    #     ff_factors_dates, ff_factors, _ = load_ff_anomalies(datapath, daily, t0_str, tN_str)
    #     print("FF factors loaded.")
    # except Exception as e:
    #     print(f"Warning: Could not load FF factors: {e}")
    #     ff_factors_dates, ff_factors = None, None
    # Note: Global variables avoided. Pass factors if needed by estimation functions.

    # --- Set Estimation Parameters ---
    p = default_params.copy() # Start with defaults
    p['freq'] = freq # Set frequency

    if interactions:
        p['kfold'] = 2 # Use 2-fold CV for interactions
    # else: p['gridsize'] = 100 # Already default

    if withhold_test_sample:
        p['oos_test_date'] = oos_test_date_str
    else:
        # If not withholding, set oos_test_date to None or last date
        # SCS_L*est handles None by using the last date of input 'dates'
        p['oos_test_date'] = None

    # --- Process Data and Run Estimation ---
    estimates = None
    print(f"Processing data provider: {dataprovider}")

    if dataprovider == 'ff25':
        if interactions:
            print("Warning: Interactions are typically not used with FF25. Skipping run.")
        else:
            print("Loading FF25 data...")
            try:
                # Assuming loader returns: dates, returns, market, other_data_dict, labels
                dd, re, mkt, DATA, labels = load_ff25(datapath, daily, t0_str, tN_str)
                print("FF25 data loaded.")

                # Specific params for FF25 run
                p_ff25 = p.copy()
                # p_ff25['smb'] = DATA.get('SMB') # Pass factors if needed
                # p_ff25['hml'] = DATA.get('HML')
                p_ff25['L2_table_rows'] = 10 # Example specific param
                p_ff25['table_L2coefs_posneg_sort'] = not rotate_PC
                p_ff25['table_L2coefs_extra_space'] = rotate_PC
                p_ff25['L2_sort_loc'] = 'OLS'
                p_ff25['devol_unconditionally'] = True # Override default

                print("Running SCS_L2est for FF25...")
                estimates = SCS_L2est(dd, re, mkt, freq, labels, p_ff25, verbose=True)
                print("SCS_L2est for FF25 finished.")

            except FileNotFoundError:
                print(f"Error: FF25 data file not found in {datapath}")
            except Exception as e:
                print(f"Error processing FF25 data: {e}")

    elif dataprovider == 'anom': # Assuming 'anom' means managed portfolios
        # Find the appropriate managed portfolio file
        fmask = os.path.join(instrpath, f'managed_portfolios_{dataprovider}{suffix}_*.csv')
        flist = glob.glob(fmask)
        if not flist:
            print(f"Error: No managed portfolio file found matching mask: {fmask}")
        else:
            filename = flist[0] # Use the first file found
            print(f"Loading managed portfolio data from: {filename}")
            try:
                p_anom = p.copy()
                # p_anom['L1_truncPath'] = True # Example specific param if needed by L1L2

                if interactions:
                    print("Loading managed portfolios WITH interactions...")
                    # Load data allowing interactions (empty exclude list)
                    dd, re, mkt, anomalies = load_managed_portfolios(filename, daily, 0.2, exclude_prefixes={})
                    print("Data loaded. Running SCS_L2est (interactions)...")
                    # Typically L2 is used when interactions are present? Check paper/original logic.
                    estimates = SCS_L2est(dd, re, mkt, freq, anomalies, p_anom, verbose=True)
                    print("SCS_L2est (interactions) finished.")
                else:
                    print("Loading managed portfolios WITHOUT interactions...")
                    # Load data excluding interactions/derived terms
                    exclude = {'rX_', 'r2_', 'r3_'}
                    dd, re, mkt, anomalies = load_managed_portfolios(filename, daily, 0.2, exclude_prefixes=exclude)
                    print("Data loaded. Running SCS_L1L2est (no interactions)...")
                    # Run L1+L2 estimation
                    estimates = SCS_L1L2est(dd, re, mkt, freq, anomalies, p_anom, verbose=True)
                    print("SCS_L1L2est (no interactions) finished.")

            except FileNotFoundError:
                print(f"Error: Managed portfolio file not found: {filename}")
            except Exception as e:
                print(f"Error processing managed portfolio data: {e}")
    else:
        print(f"Error: Unknown dataprovider '{dataprovider}'")

    # --- Display Results ---
    if estimates is not None:
        print("\n--- Estimation Results ---")
        if 'optimal_model_L2' in estimates:
            print("\nOptimal L2 Model:")
            opt_l2 = estimates['optimal_model_L2']
            print(f"  Objective ({p['objective']}): {opt_l2.get('objective'):.4f}")
            print(f"  Kappa: {opt_l2.get('kappa'):.4f}")
            print(f"  DoF: {opt_l2.get('dof'):.2f}")
            print(f"  SR: {opt_l2.get('SR'):.4f}")
            # print("  Coefficients (Top 5 absolute):")
            # coefs = opt_l2.get('coefficients', [])
            # if len(coefs) > 0:
            #     idx = np.argsort(np.abs(coefs))[::-1]
            #     for i in idx[:5]:
            #         print(f"    {anomalies[i]}: {coefs[i]:.4f}")

        if 'optimal_model_L1L2' in estimates:
             print("\nOptimal L1+L2 Model:")
             opt_l1l2 = estimates['optimal_model_L1L2']
             print(f"  Objective ({p['objective']}): {opt_l1l2.get('objective'):.4f}")
             print(f"  Kappa (L2): {opt_l1l2.get('kappa'):.4f}")
             print(f"  Num Nonzero (L1): {opt_l1l2.get('num_nonzero')}")
             print(f"  R2 OOS Bias: {opt_l1l2.get('R2oos_bias', np.nan):.4f} +/- {opt_l1l2.get('R2oos_bias_se', np.nan):.4f}")
             # print("  Coefficients (Non-zero):")
             # coefs = opt_l1l2.get('coefficients', [])
             # if len(coefs) > 0:
             #     idx_nonzero = np.where(np.abs(coefs) > 1e-9)[0]
             #     for i in idx_nonzero:
             #          print(f"    {anomalies[i]}: {coefs[i]:.4f}")

    else:
        print("\nNo estimates were generated.")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
