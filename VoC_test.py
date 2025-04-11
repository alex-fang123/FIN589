import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import joblib
import os
import time
from tqdm import tqdm
from sklearn.metrics import r2_score, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from scipy import stats

# --- Configuration ---
output_dir = "output" # Directory to save plots and results
nr_features = 6000 # Number of base features for RFF
target_z_for_plots = 1000 # Specific z value for detailed plotting

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# --- Data Loading and Initial Processing ---
print("Loading and processing predictor data...")
# Load Excel data from the 'data' subfolder
excel_path = os.path.join("data", "PredictorData2023.xlsx")
try:
    data_raw = pd.read_excel(excel_path, sheet_name="Monthly")
except FileNotFoundError:
    print(f"Error: Predictor data file not found at {excel_path}")
    exit() # Exit if main data file is missing

data_raw["yyyymm"] = pd.to_datetime(data_raw["yyyymm"], format='%Y%m', errors='coerce')
# Handle potential commas in 'Index' column before converting to float
data_raw["Index"] = data_raw["Index"].apply(lambda x: str(x).replace(",", "") if pd.notnull(x) else x)
data_raw = data_raw.set_index("yyyymm")

# Convert columns to float, handling potential errors by coercing them
for col in data_raw.columns:
    data_raw[col] = pd.to_numeric(data_raw[col], errors='coerce')

data_raw = data_raw.rename({"Index": "prices"}, axis=1)

# --- Feature Calculation ---
print("Calculating features...")
columns = ["b/m", "de", "dfr", "dfy", "dp", "dy", "ep", "infl", "ltr", "lty", "ntis", "svar", "tbl", "tms", "lag_returns"]
# Calculate missing columns according to the explanation in Welch and Goyal (2008)

# Ensure required columns exist before calculation
required_cols_for_calc = ["BAA", "AAA", "lty", "tbl", "D12", "E12", "corpr", "ltr", "prices"]
missing_required = [col for col in required_cols_for_calc if col not in data_raw.columns]
if missing_required:
    # Allow script to continue but warn, some calculations might fail
    print(f"Warning: Missing required columns for calculations: {missing_required}. Subsequent calculations might fail.")
    # Or raise ValueError if these are critical:
    # raise ValueError(f"Missing required columns for calculations: {missing_required}")

# Perform calculations safely, checking if columns exist
if "BAA" in data_raw.columns and "AAA" in data_raw.columns:
    data_raw["dfy"] = data_raw["BAA"] - data_raw["AAA"]
if "lty" in data_raw.columns and "tbl" in data_raw.columns:
    data_raw["tms"] = data_raw["lty"] - data_raw["tbl"]
if "D12" in data_raw.columns and "E12" in data_raw.columns:
    data_raw["de"] = np.log(data_raw["D12"]) - np.log(data_raw["E12"])
if "corpr" in data_raw.columns and "ltr" in data_raw.columns:
    data_raw["dfr"] = data_raw["corpr"] - data_raw["ltr"]
if "prices" in data_raw.columns:
    data_raw["lag_price"] = data_raw["prices"].shift()
    if "D12" in data_raw.columns:
        data_raw["dp"] = np.log(data_raw["D12"]) - np.log(data_raw["prices"])
        data_raw["dy"] = np.log(data_raw["D12"]) - np.log(data_raw["lag_price"])
    if "E12" in data_raw.columns:
        data_raw["ep"] = np.log(data_raw["E12"]) - np.log(data_raw["prices"])
    data_raw["returns"] = data_raw["prices"].pct_change()
    data_raw["lag_returns"] = data_raw["returns"].shift()

# Ensure 'returns' and 'prices' are defined even if calculations failed
returns = data_raw["returns"].copy() if "returns" in data_raw.columns else pd.Series(dtype=float)
prices = data_raw["prices"].copy() if "prices" in data_raw.columns else pd.Series(dtype=float)

# Visualize missing data pattern for selected columns
print("Visualizing missing data pattern...")
plt.figure(figsize=(10, 5)) # Create a new figure
msno.matrix(data_raw[columns], figsize=(10, 5))
plt.title("Missings by column")
plt.savefig(os.path.join(output_dir, "missing_pattern.jpg"))
plt.close() # Close the figure

# Select final columns and drop rows with missing values
# Filter columns list to only include those that actually exist in data_raw
existing_columns = [col for col in columns if col in data_raw.columns]
if len(existing_columns) < len(columns):
    print(f"Warning: Some columns specified in 'columns' list do not exist: {set(columns) - set(existing_columns)}")

if not existing_columns:
     print("Error: No valid columns selected for analysis. Exiting.")
     exit()

data = data_raw[existing_columns].dropna()
if data.empty:
    print("Error: Data is empty after dropping NaNs for selected columns. Exiting.")
    exit()

returns = returns[returns.index.isin(data.index)]

# --- Standardization ---
# Standardize predictors using expanding window of 36 months
print("Standardizing predictors...")
for col in tqdm(data.columns): # Iterate over existing columns in data
    rolling_mean = data[col].expanding(36).mean()
    rolling_std = data[col].expanding(36).std()
    # Avoid division by zero or NaNs
    data[col] = (data[col] - rolling_mean) / rolling_std.replace(0, np.nan)

# Standardize returns by their past 12-month rolling standard deviation
print("Standardizing returns...")
returns_std = returns.rolling(12).std().shift()
# Avoid division by zero or NaNs
returns = returns / returns_std.replace(0, np.nan)

# Drop first 36 months (burn-in for expanding stats) and handle potential NaNs from standardization
data = data[36:].dropna()
if data.empty:
    print("Error: Data is empty after burn-in period and dropping NaNs post-standardization. Exiting.")
    exit()

returns = returns.loc[data.index].dropna() # Align indices and drop NaNs
returns_std = returns_std.loc[data.index] # Align returns_std as well

# Check if returns became empty after alignment and dropna
if returns.empty:
    print("Error: Returns data is empty after alignment and dropping NaNs post-standardization. Exiting.")
    exit()
# Ensure data and returns still have aligned indices
common_index = data.index.intersection(returns.index)
data = data.loc[common_index]
returns = returns.loc[common_index]
returns_std = returns_std.loc[common_index]

if data.empty or returns.empty:
     print("Error: Data or returns became empty after final alignment. Exiting.")
     exit()

# --- Covariance Matrix Plot ---
print("Plotting covariance matrix...")
plt.figure(figsize=(10, 10)) # Set figure size before plotting
sns.heatmap(data.cov(), center=0, vmin=-1, vmax=1, cmap=sns.color_palette("coolwarm_r", as_cmap=True))
plt.title("Covariance Matrix")
plt.tight_layout() # Adjust layout
plt.savefig(os.path.join(output_dir, "covariance_matrix.jpg"))
plt.close() # Close the figure

# --- Random Fourier Features (RFF) Generation ---
rff_names = []
rff_features = []
omegas = []

print(f"Generating {2 * nr_features} Random Fourier Features...")
# Generate omegas and apply projections
for i in tqdm(range(nr_features)):
    # Omega samples from N(0, scale^2), scale=2 -> variance=4
    omega = np.random.normal(loc=0.0, scale=2.0, size=data.shape[1]) # shape: (n_original_features,)
    projection = data.values @ omega # shape: (n_obs,)
    rff_features.append(np.sin(projection))
    rff_features.append(np.cos(projection))
    rff_names.append(f"sin_{i}")
    rff_names.append(f"cos_{i}")
    omegas.append(omega)

# Stack features into an array: (2*nr_features, n_obs) -> transpose to (n_obs, 2*nr_features)
rff_array = np.vstack(rff_features).T
rff_df = pd.DataFrame(rff_array, columns=rff_names, index=data.index)

# Combine original and RFF features
# Ensure indices align perfectly before concatenation
data_full = pd.concat([data, rff_df.loc[data.index]], axis=1)
print("Shape of data after RFF transformation:", data_full.shape)

# --- Backtesting Setup ---
regression_data = data_full[rff_names]

# Regularization parameters (Ridge alpha)
z_values = [10**-3, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9]
# Time indices for rolling window (needs at least 12 months history)
# data.shape[0] is the number of observations after initial processing
t_values = list(range(12, data.shape[0])) # t represents the end of the training period

# Store backtest results
backtest_results = []
print("Running backtest...")
start_time = time.time()

# --- Backtesting Loop ---
# Loop through time, excluding the last period as there's no subsequent return to predict
for t in tqdm(t_values[:-1]):
    for z in z_values:
        try:
            # Training data: Features (S) and Returns (R) from t-11 to t (12 months)
            train_start_idx = t - 11 # Index for the start of the 12-month window
            train_end_idx = t      # Index for the end of the 12-month window

            # Use .iloc for position-based slicing corresponding to t_values range
            R = returns.iloc[train_start_idx : train_end_idx + 1].values # Target variable (t-11 to t)
            S = regression_data.iloc[train_start_idx : train_end_idx + 1].values # Features (t-11 to t)

            # Test data: Features (S_t) for period t+1 to predict return R_s at t+1
            test_idx = t + 1
            R_s = returns.iloc[test_idx : test_idx + 1] # Actual return for t+1
            S_t = regression_data.iloc[test_idx : test_idx + 1].values # Features for t+1

            # Check for NaNs in the slices or empty slices
            if R_s.empty or S_t.shape[0] == 0 or np.any(np.isnan(S)) or np.any(np.isnan(R)) or np.any(np.isnan(S_t)) or R_s.isnull().any().any():
                 # print(f"Skipping t={t}, z={z} due to NaNs or empty slice.")
                 continue # Skip if any NaN or empty slice

            # Fit ridge regression: Predict R using S
            # Ensure R is a 1D array if Ridge expects that
            if R.ndim > 1 and R.shape[1] == 1:
                R = R.flatten()

            ridge_model = Ridge(alpha=z, fit_intercept=False, solver='auto') # Use default solver
            ridge_model.fit(S, R)
            beta = ridge_model.coef_
            beta_norm = np.sqrt(np.sum(beta**2))

            # Forecast & strategy return for period t+1
            forecast = (S_t @ beta).item() # Predicted return for t+1
            actual_return_t_plus_1 = R_s.iloc[0] # Get scalar value from the single-element Series
            timing_strategy_return = forecast * actual_return_t_plus_1 # Strategy return

            backtest_results.append({
                "z": z,
                "t_index": t, # Store the integer index t
                "index": R_s.index[0], # Store the actual DateTimeIndex for t+1
                "beta_norm": beta_norm,
                "forecast": forecast,
                "timing_strategy": timing_strategy_return,
                "return": actual_return_t_plus_1
            })

        except IndexError:
             print(f"IndexError at t={t} (Date: {returns.index[t] if t < len(returns.index) else 'OOB'}), z={z}. Likely out of bounds for iloc.")
             continue
        except Exception as e:
            print(f"Error at t={t} (Date: {returns.index[t] if t < len(returns.index) else 'OOB'}), z={z}: {e}")
            # Consider logging more details or breaking if errors are critical
            continue

# Convert results to DataFrame
backtest_df = pd.DataFrame(backtest_results)
if not backtest_df.empty:
     backtest_df = backtest_df.set_index("index") # Use DateTimeIndex

     # Add denormalized returns (ensure returns_std is aligned with backtest_df index)
     aligned_returns_std = returns_std.loc[backtest_df.index]
     backtest_df["return_denorm"] = backtest_df["return"] * aligned_returns_std
     backtest_df["forecast_denorm"] = backtest_df["forecast"] * aligned_returns_std
else:
    print("Warning: Backtest results are empty. Evaluation and plotting will be skipped.")

print(f"Backtest completed in {round(time.time() - start_time, 2)} seconds.")


# --- Performance Evaluation ---
evaluation_results = []
time_factor = 12 # Annualization factor

print("Evaluating performance for each z...")
if not backtest_df.empty:
    for z in z_values:
        df_z = backtest_df[backtest_df["z"] == z].dropna()

        if df_z.empty or len(df_z) < 2: # Need at least 2 points for std dev and regression
            print(f"Skipping z={z} due to insufficient data points after dropna.")
            continue

        # --- Regression-based Metrics ---
        # Regress strategy returns on actual market returns to get alpha and beta
        X_reg = df_z[["return"]].values # Market return as independent variable
        y_reg = df_z["timing_strategy"].values # Strategy return as dependent variable

        beta_strat_on_mkt = np.nan
        alpha_strat_on_mkt = np.nan
        r_squared = np.nan
        try:
            market_reg = LinearRegression().fit(X_reg, y_reg)
            beta_strat_on_mkt = market_reg.coef_[0] # Strategy's Beta relative to market
            # Annualize intercept for Alpha
            alpha_strat_on_mkt = market_reg.intercept_ * time_factor
            # R2 of the regression of strategy return on market return
            r_squared = market_reg.score(X_reg, y_reg)
        except Exception as e:
             print(f"Regression failed for z={z}: {e}")


        # --- Strategy Performance Metrics ---
        strat_mean_annual = df_z["timing_strategy"].mean() * time_factor
        strat_std_annual = df_z["timing_strategy"].std() * np.sqrt(time_factor)
        strat_sharpe = strat_mean_annual / strat_std_annual if strat_std_annual != 0 else np.nan

        # --- Market Performance Metrics (for comparison) ---
        mkt_mean_annual = df_z["return"].mean() * time_factor
        mkt_std_annual = df_z["return"].std() * np.sqrt(time_factor)
        mkt_sharpe = mkt_mean_annual / mkt_std_annual if mkt_std_annual != 0 else np.nan

        # --- Information Ratio ---
        active_return = df_z["timing_strategy"] - df_z["return"]
        tracking_error_annual = active_return.std() * np.sqrt(time_factor)
        information_ratio = (strat_mean_annual - mkt_mean_annual) / tracking_error_annual if tracking_error_annual != 0 else np.nan

        # --- Classification Metrics (Forecast Sign Accuracy) ---
        actual_up = (df_z["return"] > 0).astype(int)
        forecast_up = (df_z["forecast"] > 0).astype(int)

        precision = precision_score(actual_up, forecast_up, zero_division=0)
        recall = recall_score(actual_up, forecast_up, zero_division=0)
        accuracy = accuracy_score(actual_up, forecast_up)

        evaluation_results.append({
            "log10(z)": np.log10(z),
            "beta_norm_mean": df_z["beta_norm"].mean(), # Avg L2 norm of Ridge coefficients
            "Strategy Sharpe": strat_sharpe,
            "Market Sharpe": mkt_sharpe,
            "Information Ratio": information_ratio,
            "Annualized Alpha (vs Market)": alpha_strat_on_mkt,
            "Strategy Beta (vs Market)": beta_strat_on_mkt,
            "R-squared (Strat vs Mkt)": r_squared,
            "Strategy Annual Return": strat_mean_annual,
            "Strategy Annual Vol": strat_std_annual,
            "Market Annual Return": mkt_mean_annual,
            "Market Annual Vol": mkt_std_annual,
            "Precision (Forecast Sign)": precision,
            "Recall (Forecast Sign)": recall,
            "Accuracy (Forecast Sign)": accuracy,
        })
    evaluation_df = pd.DataFrame(evaluation_results)
    print("Performance Evaluation Results:")
    print(evaluation_df.round(5))

    # Save results to Excel
    results_path = os.path.join(output_dir, "backtest_evaluation_results.xlsx")
    try:
        evaluation_df.to_excel(results_path, index=False)
        print(f"Evaluation results saved to {results_path}")
    except Exception as e:
        print(f"Error saving evaluation results to Excel: {e}")

else:
    print("Skipping evaluation as backtest results are empty.")


# --- Plot Performance Metrics ---
if 'evaluation_df' in locals() and not evaluation_df.empty:
    print("Plotting performance metrics...")
    # Set plot style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Plot key metrics using the corrected names
    metrics_to_plot = ["Strategy Sharpe", "Information Ratio", "Annualized Alpha (vs Market)"]
    for metric in metrics_to_plot:
        if metric in evaluation_df.columns:
            sns.lineplot(data=evaluation_df, x="log10(z)", y=metric, marker="o", label=metric)
        else:
            print(f"Warning: Metric '{metric}' not found in evaluation results for plotting.")


    plt.title("Performance Metrics vs log10(z)")
    plt.xlabel("log10(z)")
    plt.ylabel("Value")
    plt.legend()
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "performance_metrics_vs_z.jpg")
    plt.savefig(plot_path)
    print(f"Performance plot saved to {plot_path}")
    plt.close() # Close the figure
else:
    print("Skipping plotting as evaluation results are empty or not generated.")


# --- Load NBER Recession Dates ---
nber = None # Initialize nber to None
try:
    nber_path = os.path.join("data", "NBER_20210719_cycle_dates_pasted.csv")
    # Assuming standard header is row 0. Adjust if needed (e.g., skiprows=1)
    nber = pd.read_csv(nber_path)
    # Convert date columns
    nber["peak"] = pd.to_datetime(nber["peak"], errors='coerce')
    nber["trough"] = pd.to_datetime(nber["trough"], errors='coerce')
    # Drop rows where conversion failed
    nber = nber.dropna(subset=['peak', 'trough'])
    print("NBER data loaded successfully.")
except FileNotFoundError:
    print(f"Warning: NBER data file not found at {nber_path}. Recession plotting will be skipped.")
except Exception as e:
    print(f"Error loading or processing NBER data: {e}")


# --- Plot Forecast/Strategy with Recessions (Combined) ---
if nber is not None and not nber.empty and 'backtest_df' in locals() and not backtest_df.empty:
    print("Plotting forecast and strategy with NBER recessions...")
    plot_df_combined = backtest_df[backtest_df["z"] == target_z_for_plots][["forecast", "timing_strategy"]].copy()

    if not plot_df_combined.empty:
        # Calculate 6-month moving averages
        plot_df_combined["Forecast 6m MA"] = plot_df_combined["forecast"].rolling(6).mean()
        plot_df_combined["Strategy 6m MA"] = plot_df_combined["timing_strategy"].rolling(6).mean()

        # Create recession indicator
        plot_df_combined["NBER Recession"] = 0
        for _, row in nber.iterrows():
             plot_df_combined.loc[(plot_df_combined.index >= row['peak']) & (plot_df_combined.index <= row['trough']), "NBER Recession"] = 1

        plot_df_combined = plot_df_combined.dropna(subset=["Forecast 6m MA", "Strategy 6m MA"]) # Drop NaNs from MA

        if not plot_df_combined.empty:
            fig, ax = plt.subplots(figsize=(18, 10))

            # Plot moving averages
            plot_df_combined["Forecast 6m MA"].plot(ax=ax, label="Forecast (6m MA)")
            plot_df_combined["Strategy 6m MA"].plot(ax=ax, label="Timing Strategy (6m MA)")

            # Add recession shading (only once)
            min_val, max_val = ax.get_ylim() # Get y-limits after plotting lines
            # Ensure min_val and max_val are finite
            min_val = min_val if np.isfinite(min_val) else -1
            max_val = max_val if np.isfinite(max_val) else 1
            ax.fill_between(plot_df_combined.index, min_val, max_val,
                            where=plot_df_combined["NBER Recession"] == 1,
                            color='grey', alpha=0.3, label="NBER Recession")

            ax.axhline(0, c="black", linestyle="--", linewidth=0.8)
            ax.legend(loc="upper right")
            ax.set_title(f"Forecast vs. Timing Strategy (z={target_z_for_plots}, 6m MA) with NBER Recessions")
            ax.set_ylabel("Value")
            ax.set_xlabel("Date")

            plt.tight_layout()
            plot_path_combined = os.path.join(output_dir, f"forecast_strategy_recessions_z{target_z_for_plots}.jpg")
            plt.savefig(plot_path_combined)
            print(f"Combined forecast/strategy plot saved to {plot_path_combined}")
            plt.close(fig) # Close the figure
        else:
             print(f"No data to plot for combined forecast/strategy after processing.")
    else:
        print(f"No data found for z={target_z_for_plots} to plot combined forecast/strategy.")
elif nber is None or nber.empty:
     print("Skipping combined recession plotting as NBER data is missing or invalid.")
else: # backtest_df is empty or missing
     print("Skipping combined recession plotting as backtest data is missing.")


# --- Plot Forecast/Strategy with Recessions (Separate) ---
if nber is not None and not nber.empty and 'backtest_df' in locals() and not backtest_df.empty:
    print("Plotting forecast and strategy separately with NBER recessions...")

    for col in ["forecast", "timing_strategy"]:
        plot_df_single = backtest_df[backtest_df["z"] == target_z_for_plots][[col]].copy()

        if not plot_df_single.empty:
            plot_df_single["6m MA"] = plot_df_single[col].rolling(6).mean()

            # Create recession indicator
            plot_df_single["NBER Recession"] = 0
            for _, row in nber.iterrows():
                 plot_df_single.loc[(plot_df_single.index >= row['peak']) & (plot_df_single.index <= row['trough']), "NBER Recession"] = 1

            plot_df_single = plot_df_single.dropna() # Drop NaNs from MA and original col if any

            if not plot_df_single.empty:
                fig, ax = plt.subplots(figsize=(18, 10))

                # Plot raw data and moving average
                plot_df_single[col].plot(ax=ax, alpha=0.5, label=f"{col} (Raw)", color='lightblue')
                plot_df_single["6m MA"].plot(ax=ax, label=f"{col} (6m MA)", color='steelblue')

                # Adjust ylim based on MA
                min_ma, max_ma = plot_df_single["6m MA"].min(), plot_df_single["6m MA"].max()
                padding = (max_ma - min_ma) * 0.1 if max_ma > min_ma else 0.1 # Add 10% padding, handle flat lines
                ax.set_ylim(min_ma - padding, max_ma + padding)

                # Add recession shading
                min_val, max_val = ax.get_ylim() # Get y-limits after plotting lines and setting ylim
                min_val = min_val if np.isfinite(min_val) else -1
                max_val = max_val if np.isfinite(max_val) else 1
                ax.fill_between(plot_df_single.index, min_val, max_val,
                                where=plot_df_single["NBER Recession"] == 1,
                                color='grey', alpha=0.3, label="NBER Recession")

                ax.axhline(0, c="black", linestyle="--", linewidth=0.8)
                ax.legend(loc="upper right")
                ax.set_title(f"{col.capitalize()} (z={target_z_for_plots}) with NBER Recessions")
                ax.set_ylabel("Value")
                ax.set_xlabel("Date")

                plt.tight_layout()
                plot_path_single = os.path.join(output_dir, f"{col}_recessions_z{target_z_for_plots}.jpg")
                plt.savefig(plot_path_single)
                print(f"Separate plot for {col} saved to {plot_path_single}")
                plt.close(fig) # Close the figure
            else:
                 print(f"No data to plot for {col} after processing.")
        else:
            print(f"No data found for z={target_z_for_plots} to plot {col}.")
# No plt.show() needed here either if running as a script

print("Script finished.")
