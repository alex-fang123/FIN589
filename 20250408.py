import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_FILE = 'data/PredictorData2023.xlsx'
# Specify the sheet name if needed, otherwise pandas reads the first sheet
SHEET_NAME = 'Monthly' # Use 'Monthly' sheet for predictability analysis

# --- Load Data ---
try:
    # Try reading with the default sheet (first one)
    df_raw = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    print("Successfully loaded data.")
    print("\nFirst 5 rows of the data:")
    print(df_raw.head())
    print("\nColumn names:")
    print(df_raw.columns.tolist())
    print("\nData Info:")
    df_raw.info()

except Exception as e:
    print(f"Error loading data from {DATA_FILE}: {e}")
    df_raw = pd.DataFrame() # Create an empty DataFrame if loading fails


# --- Data Preprocessing ---
def preprocess_data(df):
    """Preprocesses the raw data."""
    print("\n--- Starting Data Preprocessing ---")
    df = df.copy()

    # 1. Handle Dates
    df['Date'] = pd.to_datetime(df['yyyymm'], format='%Y%m')
    df = df.set_index('Date')
    print("Converted 'yyyymm' to datetime index.")

    # 2. Rename Columns (using paper notation where possible)
    df = df.rename(columns={
        'Index': 'sp500',
        'D12': 'd12',
        'E12': 'e12',
        'b/m': 'bm',
        'tbl': 'tbl',
        'AAA': 'aaa',
        'BAA': 'baa',
        'lty': 'lty',
        'ntis': 'ntis',
        'Rfree': 'rf',
        'infl': 'infl',
        'ltr': 'ltr',
        'corpr': 'corpr',
        'svar': 'svar',
        'csp': 'csp',
        'CRSP_SPvw': 'vw',
        'CRSP_SPvwx': 'vwx'
    })
    print("Renamed columns.")

    # 3. Calculate Equity Premium (Log version as often used in paper)
    # Ensure returns are not negative before log. Add small constant if necessary, or check data source convention.
    # Let's calculate simple return premium first for robustness check
    df['ep_simple'] = df['vw'] - df['rf']
    # Log equity premium: log(1+MarketReturn) - log(1+RiskFreeRate)
    df['ep_log'] = np.log(1 + df['vw']) - np.log(1 + df['rf'])
    print("Calculated Equity Premium (simple and log).")


    # 4. Calculate Predictor Variables
    # Dividend Price Ratio (d/p) = log(D12) - log(P)
    df['dp'] = np.log(df['d12']) - np.log(df['sp500'])
    # Dividend Yield (d/y) = log(D12) - log(P_{t-1})
    df['dy'] = np.log(df['d12']) - np.log(df['sp500'].shift(1))
    # Earnings Price Ratio (e/p) = log(E12) - log(P)
    df['ep'] = np.log(df['e12']) - np.log(df['sp500'])
     # Handle potential division by zero or log(0) if E12 is zero or negative
    df.loc[df['e12'] <= 0, 'ep'] = np.nan # Set ep to NaN if E12 is non-positive
    # Dividend Payout Ratio (d/e) = log(D12) - log(E12)
    df['de'] = np.log(df['d12']) - np.log(df['e12'])
    df.loc[df['e12'] <= 0, 'de'] = np.nan # Set de to NaN if E12 is non-positive

    # Term Spread (tms) = Long Term Yield - Treasury Bill Rate
    df['tms'] = df['lty'] - df['tbl']
    # Default Yield Spread (dfy) = BAA Yield - AAA Yield
    df['dfy'] = df['baa'] - df['aaa']
    # Default Return Spread (dfr) = Corporate Bond Return - Long Term Gov Bond Return
    df['dfr'] = df['corpr'] - df['ltr']
    print("Calculated predictor variables (dp, dy, ep, de, tms, dfy, dfr).")

    # 5. Filter Data (Paper often uses 1927 onwards for monthly)
    start_date = '1927-01-01'
    df = df[df.index >= start_date]
    print(f"Filtered data to start from {start_date}.")

    # 6. Lag Predictors (Predict equity premium at t with variables from t-1)
    predictors = ['dp', 'dy', 'ep', 'de', 'svar', 'bm', 'ntis', 'tbl', 'lty', 'ltr', 'tms', 'dfy', 'dfr', 'infl'] # Add others as needed
    for var in predictors:
        if var in df.columns:
            df[f'{var}_lag'] = df[var].shift(1)
        else:
            print(f"Warning: Predictor '{var}' not found in DataFrame columns.")

    print("Lagged predictor variables.")

    # Drop initial rows with NaNs created by lagging/shifting
    # Ensure target and *all* lagged predictors used in the loop later are not NaN initially
    lagged_predictors_in_df = [f'{p}_lag' for p in predictors if f'{p}_lag' in df.columns]
    df = df.dropna(subset=['ep_log'] + lagged_predictors_in_df)
    print("Dropped rows with NaNs after lagging.")
    print("--- Data Preprocessing Complete ---")
    return df


# --- Analysis Functions ---
def run_in_sample_regression(y, x_name, df):
    """
    Runs an in-sample OLS regression of y on the predictor x_name.

    Args:
        y (pd.Series): The dependent variable series (e.g., equity premium).
        x_name (str): The name of the predictor column in the DataFrame.
        df (pd.DataFrame): The DataFrame containing y and x.

    Returns:
        dict: A dictionary containing regression results (R-squared, coefficient, p-value).
              Returns None if regression fails.
    """
    if x_name not in df.columns:
        print(f"Error: Predictor '{x_name}' not found in DataFrame.")
        return None
    if y.name not in df.columns:
         print(f"Error: Dependent variable '{y.name}' not found in DataFrame.")
         return None

    X = df[x_name].copy()
    Y = df[y.name].copy()

    # Drop NaNs for this specific regression pair
    valid_idx = Y.notna() & X.notna()
    Y = Y[valid_idx]
    X = X[valid_idx]

    if len(X) < 2: # Need at least 2 data points for regression
        print(f"Warning: Not enough non-NaN data points for predictor '{x_name}'. Skipping.")
        return None

    X = sm.add_constant(X) # Add intercept term

    try:
        model = sm.OLS(Y, X)
        results = model.fit()
        # Consider using HAC standard errors as in the paper for more robust p-values
        # results_hac = model.fit(cov_type='HAC', cov_kwds={'maxlags': 12}) # Example: Newey-West with 12 lags

        return {
            'predictor': x_name,
            'r_squared': results.rsquared,
            'adj_r_squared': results.rsquared_adj,
            'coefficient': results.params[x_name],
            'p_value': results.pvalues[x_name],
            'n_obs': results.nobs
            # 'p_value_hac': results_hac.pvalues[x_name] # If using HAC
        }
    except Exception as e:
        print(f"Error running OLS for predictor '{x_name}': {e}")
        return None

def run_out_of_sample_forecast(y, x_name, df, start_oos_idx):
    """
    Runs a rolling out-of-sample forecast evaluation.

    Args:
        y (pd.Series): The dependent variable series.
        x_name (str): The name of the predictor column.
        df (pd.DataFrame): The DataFrame containing y and x.
        start_oos_idx (int): The index position where the OOS period begins.

    Returns:
        dict: Dictionary with OOS results (OOS R-squared, MSE-F components, errors, dates).
              Returns None if OOS fails.
    """
    print(f"--- Starting OOS for {x_name} ---")
    if x_name not in df.columns or y.name not in df.columns:
        print(f"Error: Column missing for OOS: {x_name} or {y.name}")
        return None

    y_actual = df[y.name].values
    x_predictor = df[x_name].values
    n_total = len(y_actual)
    forecast_errors_model = []
    forecast_errors_mean = []
    actual_values_oos = [] # Initialize the list here
    predictions_model = []
    predictions_mean = []

    # Check if start_oos_idx is valid
    if start_oos_idx >= n_total or start_oos_idx < 2: # Need at least 2 points for initial regression
         print(f"Warning: Invalid start_oos_idx ({start_oos_idx}) for predictor '{x_name}'. Total obs: {n_total}. Skipping OOS.")
         return None

    # Rolling window forecast
    for t in range(start_oos_idx, n_total):
        # Data available up to time t-1
        y_train = y_actual[:t]
        x_train_raw = x_predictor[:t]

        # Handle NaNs within the training window for this iteration
        valid_train_idx = ~np.isnan(y_train) & ~np.isnan(x_train_raw)
        y_train_clean = y_train[valid_train_idx]
        x_train_clean = x_train_raw[valid_train_idx]

        if len(y_train_clean) < 2: # Need enough points for regression
            # Cannot make a prediction, maybe predict with historical mean?
            # For simplicity, we might skip this period or handle it differently.
            # Let's predict historical mean for both if model fails.
            forecast_model = np.mean(y_train_clean) if len(y_train_clean) > 0 else 0
            forecast_mean = forecast_model
        else:
            # Model forecast
            X_train_reg = sm.add_constant(x_train_clean)
            try:
                model = sm.OLS(y_train_clean, X_train_reg)
                results = model.fit()
                # Predict for time t using data from t-1
                x_pred_input = np.array([1, x_predictor[t-1]]) # Constant and predictor at t-1
                if np.isnan(x_pred_input[1]): # Handle NaN in predictor used for forecasting
                     forecast_model = np.mean(y_train_clean) # Fallback to mean
                else:
                    forecast_model = results.predict(x_pred_input)[0]

            except Exception as e:
                 print(f"Warning: OOS Regression failed at step t={t} for {x_name}. Using mean forecast. Error: {e}")
                 forecast_model = np.mean(y_train_clean) if len(y_train_clean) > 0 else 0


            # Historical mean forecast (expanding window mean)
            forecast_mean = np.mean(y_train_clean)

        # Store forecasts and actual values for time t
        actual_val = y_actual[t]
        if not np.isnan(actual_val) and not np.isnan(forecast_model) and not np.isnan(forecast_mean):
            forecast_errors_model.append(actual_val - forecast_model)
            forecast_errors_mean.append(actual_val - forecast_mean)
            actual_values_oos.append(actual_val) # Now this list exists
            predictions_model.append(forecast_model)
            predictions_mean.append(forecast_mean)
        # else: # Handle cases where prediction or actual is NaN if needed
            # print(f"Skipping OOS error calculation at step t={t} due to NaN.")


    # Calculate OOS R-squared
    sse_model = np.sum(np.array(forecast_errors_model)**2)
    sse_mean = np.sum(np.array(forecast_errors_mean)**2)

    if sse_mean == 0: # Avoid division by zero
        oos_r2 = -np.inf
    else:
        oos_r2 = 1 - (sse_model / sse_mean)

    # Calculate MSE-F components (MSE_N = sse_mean / N_oos, MSE_A = sse_model / N_oos)
    n_oos = len(forecast_errors_model)
    mse_n = sse_mean / n_oos if n_oos > 0 else np.nan
    mse_a = sse_model / n_oos if n_oos > 0 else np.nan

    print(f"--- OOS Complete for {x_name}. OOS R2: {oos_r2:.4f} ---")

    # Ensure oos_dates calculation uses n_oos which reflects actual number of OOS predictions made
    oos_dates_calculated = df.index[start_oos_idx : start_oos_idx + n_oos]

    return {
        'predictor': x_name,
        'oos_r_squared': oos_r2,
        'n_oos': n_oos,
        'mse_null': mse_n, # MSE of historical mean model (benchmark)
        'mse_alternative': mse_a, # MSE of the predictive model
        'rmse_diff': np.sqrt(mse_n) - np.sqrt(mse_a) if mse_n >= 0 and mse_a >= 0 else np.nan, # RMSE difference (sqrt(MSE_N) - sqrt(MSE_A))
        'oos_dates': oos_dates_calculated, # Use the calculated dates
        'errors_model': np.array(forecast_errors_model),
        'errors_mean': np.array(forecast_errors_mean)
    }

def plot_cumulative_performance(oos_results_list, predictors_to_plot=None, filename_prefix="oos_performance"):
    """
    Plots the cumulative difference in squared prediction errors (SSE_mean - SSE_model).

    Args:
        oos_results_list (list): List of dictionaries from run_out_of_sample_forecast.
        predictors_to_plot (list, optional): List of predictor names (e.g., 'dp_lag') to plot.
                                             If None, plots all available.
        filename_prefix (str): Prefix for the saved plot filenames.
    """
    print("\n--- Generating OOS Performance Plots ---")
    if predictors_to_plot is None:
        predictors_to_plot = [res['predictor'] for res in oos_results_list if res]

    # Filter results to only include those requested for plotting
    results_to_plot = [res for res in oos_results_list if res and res['predictor'] in predictors_to_plot]

    num_plots = len(results_to_plot)
    if num_plots == 0:
        print("No valid predictors found for plotting.")
        return

    # Determine grid size (e.g., 3 columns)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
    axes = axes.flatten() # Flatten to 1D array for easy iteration

    plot_count = 0
    for res in results_to_plot:
        predictor = res['predictor']
        if 'errors_model' in res and 'errors_mean' in res and 'oos_dates' in res:
            # Check for sufficient data and matching lengths
            if len(res['errors_model']) > 0 and len(res['errors_mean']) > 0 and len(res['oos_dates']) == len(res['errors_model']):
                ax = axes[plot_count]
                dates = res['oos_dates']
                # Calculate cumulative SSE difference: cumsum(error_mean^2) - cumsum(error_model^2)
                sse_diff_cumulative = np.cumsum(res['errors_mean']**2) - np.cumsum(res['errors_model']**2)

                ax.plot(dates, sse_diff_cumulative, label=f'{predictor} vs Mean')
                ax.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Add zero line
                ax.set_title(f'OOS Performance: {predictor}')
                ax.set_ylabel('Cumulative SSE Diff (Mean - Model)')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, linestyle=':', alpha=0.6)

                # Mark Oil Shock (approximate months)
                oil_shock_start = pd.to_datetime('1973-11-01')
                oil_shock_end = pd.to_datetime('1975-03-01')
                # Only draw the span if it overlaps with the plot's date range
                if dates.min() <= oil_shock_end and dates.max() >= oil_shock_start:
                    ax.axvspan(max(dates.min(), oil_shock_start), min(dates.max(), oil_shock_end),
                               color='red', alpha=0.15, label='Oil Shock 73-75')

                ax.legend(fontsize='small')
                plot_count += 1
            else:
                print(f"Skipping plot for {predictor}: Not enough OOS error data or date mismatch (Errors: {len(res['errors_model'])}, Dates: {len(res['oos_dates'])}).")
        else:
            print(f"Skipping plot for {predictor}: Missing required data keys in results dictionary.")

    # Hide unused subplots
    for j in range(plot_count, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plot_filename = f"{filename_prefix}_cumulative_sse_diff.png"
    # try:
    #     plt.savefig(plot_filename)
    #     print(f"Saved OOS performance plot to {plot_filename}")
    # except Exception as e:
    #     print(f"Error saving plot: {e}")
    plt.show() # Display the plot interactively

# --- Main Execution ---
if __name__ == "__main__" and not df_raw.empty:
    # 1. Preprocess Data
    df_processed = preprocess_data(df_raw)

    if not df_processed.empty:
        print("\nProcessed Data Head:")
        print(df_processed.head())
        print("\nProcessed Data Info:")
        df_processed.info()
        print("\nStarting analysis...")

        # 2. Define Predictors from Paper to test (using lagged versions)
        # Based on paper and available columns after preprocessing
        predictors_to_test = [
            'dp_lag', 'dy_lag', 'ep_lag', 'de_lag', 'svar_lag', 'bm_lag', 'ntis_lag',
            'tbl_lag', 'lty_lag', 'ltr_lag', 'tms_lag', 'dfy_lag', 'dfr_lag', 'infl_lag'
            # Add 'csp_lag', 'ik_lag' if available and calculated
        ]
        # Filter list to only include predictors actually present in the dataframe
        predictors_to_test = [p for p in predictors_to_test if p in df_processed.columns]
        print(f"\nPredictors to be tested: {predictors_to_test}")


        # 3. Loop through predictors and run In-Sample analysis
        print("\n--- Running In-Sample Regressions ---")
        is_results_list = []
        for predictor in predictors_to_test:
            print(f"Running IS regression for: {predictor}")
            results = run_in_sample_regression(df_processed['ep_log'], predictor, df_processed)
            if results:
                is_results_list.append(results)
                print(f"  R-squared: {results['r_squared']:.4f}, Coef: {results['coefficient']:.4f}, P-value: {results['p_value']:.4f}, N: {results['n_obs']}")
            else:
                 print(f"  IS Regression failed for {predictor}")


        # 4. Loop through predictors and run Out-of-Sample analysis
        print("\n--- Running Out-of-Sample Forecasts ---")
        # Define OOS start point. Paper uses various starts, e.g., 20 years after data begins.
        # Data starts ~1927-02. 20 years -> 1947-02. Let's find the index position.
        try:
            # Find the index corresponding to the start date (or the first date after)
            oos_start_date_str = '1947-02-01'
            oos_start_date = pd.to_datetime(oos_start_date_str)
            # Get the index location; use searchsorted for efficiency
            oos_start_idx = df_processed.index.searchsorted(oos_start_date)
            # Ensure we don't start exactly on the date if it doesn't exist, pick the next available
            if oos_start_idx < len(df_processed) and df_processed.index[oos_start_idx] < oos_start_date:
                 oos_start_idx += 1
            # Handle case where start date is beyond the data range
            if oos_start_idx >= len(df_processed):
                 print(f"Warning: OOS start date {oos_start_date_str} is after the last data point. Adjusting OOS start.")
                 oos_start_idx = len(df_processed) // 3 # Fallback
            print(f"OOS period starts at index {oos_start_idx}, corresponding to date {df_processed.index[oos_start_idx].date()}")

        except Exception as e:
             print(f"Error determining OOS start index: {e}. Defaulting to roughly 20 years.")
             oos_start_idx = 20 * 12 # Approximate start index (20 years * 12 months)
             if oos_start_idx >= len(df_processed):
                 oos_start_idx = len(df_processed) // 3 # Fallback if data is too short


        oos_results_list = []
        for predictor in predictors_to_test:
             results = run_out_of_sample_forecast(df_processed['ep_log'], predictor, df_processed, oos_start_idx)
             if results:
                 oos_results_list.append(results)
             else:
                 print(f"  OOS Forecast failed for {predictor}")


        # 5. Summarize and report results (Formatted print)
        print("\n--- In-Sample Results Summary (Formatted) ---")
        is_summary = pd.DataFrame(is_results_list).set_index('predictor')
        # Format for better readability
        is_summary['r_squared'] = is_summary['r_squared'].map('{:.4f}'.format)
        is_summary['adj_r_squared'] = is_summary['adj_r_squared'].map('{:.4f}'.format)
        is_summary['coefficient'] = is_summary['coefficient'].map('{:.4f}'.format)
        is_summary['p_value'] = is_summary['p_value'].map('{:.4f}'.format)
        print(is_summary[['r_squared', 'adj_r_squared', 'coefficient', 'p_value', 'n_obs']])


        print("\n--- Out-of-Sample Results Summary (Formatted) ---")
        oos_summary = pd.DataFrame(oos_results_list).set_index('predictor')
        # Format for better readability
        oos_summary['oos_r_squared'] = oos_summary['oos_r_squared'].map('{:.4f}'.format)
        oos_summary['rmse_diff'] = oos_summary['rmse_diff'].map('{:.6f}'.format) # Format RMSE difference
        oos_summary['mse_null'] = oos_summary['mse_null'].map('{:.6f}'.format)
        oos_summary['mse_alternative'] = oos_summary['mse_alternative'].map('{:.6f}'.format)
        print(oos_summary[['oos_r_squared', 'rmse_diff', 'n_oos', 'mse_null', 'mse_alternative']])

        # 6. Generate Plots
        # Select a subset of key predictors for clarity, or plot all
        key_predictors = ['dp_lag', 'dy_lag', 'ep_lag', 'bm_lag', 'ntis_lag', 'tbl_lag', 'tms_lag', 'infl_lag']
        plot_cumulative_performance(oos_results_list, predictors_to_plot=key_predictors)


        print("\nAnalysis complete.")
    else:
        print("\nData processing resulted in an empty DataFrame. Check preprocessing steps and data filtering.")


elif df_raw.empty:
    print("\nCould not load data. Analysis cannot proceed.")