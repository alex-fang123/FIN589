import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from math import sqrt

# Import necessary functions from the library
try:
    from .anomnames import anomnames
    from .anomdescr import anomdescr
except ImportError:
    # Fallback if running directly
    try:
        from anomnames import anomnames
        from anomdescr import anomdescr
    except ImportError:
        # Define dummy functions if imports fail, allowing basic script execution
        warnings.warn("Could not import anomnames/anomdescr. Using identity functions.")
        def anomnames(a): return a
        def anomdescr(a): return a

# Configure matplotlib - Force disable LaTeX rendering, use mathtext
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm" # Use Computer Modern for mathtext for consistency
})
# warnings.warn("LaTeX rendering disabled for plots. Using default mathtext.") # No longer needed


def plot_dof(df, x, p):
    """
    Plots effective degrees of freedom vs. kappa.
    Returns the Matplotlib Figure object.
    """
    fig, ax = plt.subplots()
    ax.plot(x, df, linewidth=p.get('line_width', 1.5))
    if p.get('L2_log_scale', True):
        ax.set_xscale('log')
    if p.get('L1_log_scale', True): # Corresponds to y-axis (DoF) in MATLAB code?
        ax.set_yscale('log')
    ax.set_xlabel(p.get('xlbl', r'Root Expected SR$^2$ (prior), $\kappa$'))
    ax.set_ylabel('Effective degrees of freedom')
    ax.grid(True)
    ax.set_xlim(np.min(x), np.max(x))
    plt.tight_layout()
    return fig # Return figure object


def plot_L2coefpaths(x, phi, iL2opt, anomalies, ylbl, p):
    """
    Plots L2 coefficient paths vs. kappa.
    Returns the Matplotlib Figure object.
    """
    N = phi.shape[0]
    sort_loc = p.get('L2_sort_loc', 'opt')
    max_legends = p.get('L2_max_legends', 20)

    if sort_loc == 'opt':
        iSortLoc = iL2opt
    elif sort_loc == 'OLS':
        iSortLoc = 0
    else:
        warnings.warn(f"Unknown L2_sort_loc '{sort_loc}'. Defaulting to 'opt'.")
        iSortLoc = iL2opt

    if N > max_legends:
        sort_indices = np.argsort(np.abs(phi[:, iSortLoc]))[::-1]
    else:
        sort_indices = np.argsort(phi[:, iSortLoc])[::-1]

    fig, ax = plt.subplots()
    lines = ax.plot(x, phi[sort_indices, :].T, linewidth=p.get('line_width', 1.5))

    if p.get('L2_log_scale', True):
        ax.set_xscale('log')

    ax.grid(True)
    ax.set_xlabel(p.get('xlbl', r'Root Expected SR$^2$ (prior), $\kappa$'))
    ax.set_ylabel(ylbl)

    legend_indices = sort_indices[:min(max_legends, N)]
    try:
        legend_labels = anomnames(np.array(anomalies)[legend_indices])
    except Exception:
        warnings.warn("Failed to format anomaly names using anomnames for legend. Using raw names.")
        legend_labels = np.array(anomalies)[legend_indices]

    legend_loc_map = {'bestoutside': 'best'}
    loc = legend_loc_map.get(p.get('legend_loc', 'best'), p.get('legend_loc', 'best'))

    if len(legend_indices) > 0:
        ax.legend([lines[i] for i in range(len(legend_indices))], legend_labels,
                  loc=loc, fontsize=p.get('font_size', 10))

    phi_min = np.nanmin(phi)
    phi_max = np.nanmax(phi)
    if not (np.isnan(phi_min) or np.isnan(phi_max)):
        ax.plot([x[iL2opt], x[iL2opt]], [phi_min, phi_max], '--k', linewidth=1)
        ax.set_ylim(phi_min, phi_max)

    ax.set_xlim(np.min(x), np.max(x))
    plt.tight_layout()
    return fig # Return figure object


def plot_L2cv(x, objL2, p):
    """
    Plots In-sample and Out-of-sample objective function values vs. kappa.
    Returns the Matplotlib Figure object.
    """
    fig, ax = plt.subplots()
    line_width = p.get('line_width', 1.5)
    objective_name = p.get('sObjective', 'Objective')
    cv_method = p.get('method', 'CV')

    ax.plot(x, objL2[:, 0], '--', linewidth=line_width, label='In-sample')
    ax.plot(x, objL2[:, 1], '-', linewidth=line_width, label=f'OOS {cv_method}')

    oos_mean = objL2[:, 1]
    oos_se = objL2[:, 3]
    ax.plot(x, oos_mean + oos_se, ':', color=ax.lines[1].get_color(), linewidth=1)
    ax.plot(x, oos_mean - oos_se, ':', color=ax.lines[1].get_color(), linewidth=1, label=f'OOS {cv_method} $\pm$ 1 s.e.')

    if p.get('L2_log_scale', True):
        ax.set_xscale('log')

    ax.set_xlabel(p.get('xlbl', r'Root Expected SR$^2$ (prior), $\kappa$'))
    ax.set_ylabel(f'IS/OOS {objective_name}')
    ax.grid(True)

    valid_oos = oos_mean[~np.isnan(oos_mean)]
    valid_oos_se = oos_se[~np.isnan(oos_se)]
    if len(valid_oos) > 0:
        y_max_limit = max(0.1, np.nanmin([10, 2 * np.nanmax(valid_oos)]))
        y_min_limit = np.nanmin([0, np.nanmin(valid_oos - valid_oos_se)]) if len(valid_oos_se)>0 else 0
        ax.set_ylim(y_min_limit, y_max_limit)

    ax.set_xlim(np.min(x), np.max(x))

    legend_loc_map = {'NorthWest': 'upper left'}
    loc = legend_loc_map.get(p.get('legend_loc', 'best'), p.get('legend_loc', 'best'))
    ax.legend(loc=loc)
    plt.tight_layout()
    return fig # Return figure object


def plot_L1L2map(x, L1range, cv_test, figname, p):
    """
    Plots a contour map of the OOS objective function over L1 and L2 grids.
    Returns the Matplotlib Figure object.
    """
    fig, ax = plt.subplots()
    data_to_plot = cv_test.T
    min_val = np.nanmin(data_to_plot)
    max_val = np.nanmax(data_to_plot)
    level_step = p.get('contour_levelstep', 0.01)

    if p.get('objective') in {'CSR2', 'GLSR2', 'SRexpl', 'MVU'}:
        min_display_val = -0.1
        data_to_plot = np.maximum(min_display_val, data_to_plot)
        levels = np.arange(min_display_val, max_val + level_step, level_step)
    else:
        limit_val = data_to_plot[-1, 0] + 3 if not np.isnan(data_to_plot[-1, 0]) else max_val + 3
        data_to_plot = np.minimum(data_to_plot, limit_val)
        min_val = np.nanmin(data_to_plot)
        levels = np.arange(min_val, limit_val + level_step, level_step)

    if len(levels) < 2:
        levels = np.linspace(min_val, max_val, 10) if max_val > min_val else [min_val, min_val + 1e-6]

    contour_filled = ax.contourf(x, L1range, data_to_plot, levels=levels, cmap='viridis')
    fig.colorbar(contour_filled, ax=ax, label=p.get('sObjective', 'Objective Value'))

    if p.get('L1_log_scale', True):
        ax.set_yscale('log')
    if p.get('L2_log_scale', True):
        ax.set_xscale('log')

    ax.set_xlabel(p.get('xlbl', r'Root Expected SR$^2$ (prior), $\kappa$'))
    ax.set_ylabel('Number of nonzero coefficients')
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(L1range), np.max(L1range))
    plt.tight_layout()
    return fig # Return figure object


def table_L2coefs(phi, se, anomalies, p):
    """
    Prints a table of the largest coefficients and t-stats.
    """
    nrows = p.get('L2_table_rows', 10)

    with np.errstate(divide='ignore', invalid='ignore'):
        tstats = phi / se
    tstats[np.isnan(tstats)] = 0
    tstats[np.isinf(tstats)] = np.sign(phi[np.isinf(tstats)]) * 1e9 if np.isinf(tstats).any() else 0

    sort_indices = np.argsort(np.abs(tstats))[::-1]
    idx_to_show = sort_indices[:min(nrows, len(phi))]

    portfolio_names = []
    if len(idx_to_show) > 0:
        try:
            anomalies_to_describe = np.array(anomalies)[idx_to_show]
            if anomalies_to_describe.size > 0:
                 portfolio_names = anomdescr(list(anomalies_to_describe))
            else:
                 warnings.warn("Anomaly indices for table are empty. Using raw indices.")
                 portfolio_names = [f"Index {i}" for i in idx_to_show]
        except Exception as e:
            warnings.warn(f"Failed to get descriptions using anomdescr: {e}. Using raw anomaly names.")
            try:
                 portfolio_names = np.array(anomalies)[idx_to_show]
            except IndexError:
                 warnings.warn("Could not retrieve anomaly names for table.")
                 portfolio_names = [f"Index {i}" for i in idx_to_show]
    else:
         warnings.warn("No coefficients selected for table.")

    if len(portfolio_names) == len(idx_to_show):
        tbl_data = {
            'Portfolio': portfolio_names,
            'b': phi[idx_to_show],
            '|t-stat|': np.abs(tstats[idx_to_show])
        }
    else:
         tbl_data = {
            'Index': idx_to_show,
            'b': phi[idx_to_show],
            '|t-stat|': np.abs(tstats[idx_to_show])
         }
    tbl = pd.DataFrame(tbl_data)

    print("\n--- Top Coefficients (Sorted by |t-stat|) ---")
    print(tbl.to_string(index=False, float_format="%.4f"))
    print("--------------------------------------------")
