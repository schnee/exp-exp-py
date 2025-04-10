# app.py
# A/B Test Design and Analysis Tool
# Current time: Thursday, April 10, 2025 at 9:15:41 AM CDT (Austin, Texas)

import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
import warnings # To suppress specific warnings if needed

# --- Statistical Helper Functions ---

def calculate_significance_error(p_control, p_threshold, n_samples):
    """
    Calculates the probability of a Type I error (alpha).
    Probability that the control group (rate p_control) measures
    at or above p_threshold in n_samples.
    Equivalent to R's pbinom(ceiling(q*n)-1, prob=p, size=n, lower.tail=FALSE)
    """
    n_samples = float(n_samples) # Ensure n is float for calculations
    if n_samples <= 0: return 1.0
    if p_threshold <= p_control: return 1.0 # Error is certain or threshold not meaningful
    if p_control <= 0: return 0.0 # Edge case: If control is 0%, can never reach positive threshold > 0%

    # Threshold count (minimum successes needed to meet threshold)
    # Use max(0, ...) to handle potential floating point inaccuracies near 0
    threshold_count = math.ceil(max(0.0, p_threshold * n_samples))

    # We want P(X >= threshold_count) where X ~ Binom(n_samples, p_control)
    # This is the survival function sf(k) = P(X > k) = 1 - cdf(k)
    # P(X >= k) = P(X > k-1) = sf(k-1)
    # Clamp k-1 at -1 if threshold_count is 0, sf(-1) correctly gives 1.
    k_minus_1 = max(-1.0, threshold_count - 1.0)

    try:
        # Ensure n is integer for binom function
        n_int = int(round(n_samples))
        if n_int <= 0: return 1.0
        # Use np.clip for p_control safety
        p_safe = np.clip(p_control, 0.0, 1.0)
        return stats.binom.sf(k_minus_1, n_int, p_safe)
    except (ValueError, OverflowError) as e:
        # Handle potential numerical issues gracefully
        st.warning(f"Binomial calculation error (Significance): n={n_samples:.1f}, k-1={k_minus_1:.1f}, p={p_control:.4f}. Error: {e}")
        # Try with higher precision if available or return edge value
        # Returning NaN might be better to signal failure clearly
        return np.nan


def calculate_power_error(p_threshold, p_variant, n_samples):
    """
    Calculates the probability of a Type II error (beta).
    Probability that the variant group (rate p_variant) measures
    at or below p_threshold in n_samples.
    Equivalent to R's pbinom(floor(p*n), prob=q, size=n, lower.tail=TRUE)
    """
    n_samples = float(n_samples) # Ensure n is float
    if n_samples <= 0: return 1.0
    if p_variant <= p_threshold: return 1.0 # Error is certain or variant not better than threshold
    if p_variant >= 1: return 0.0 if p_threshold < 1.0 else 1.0 # Edge case: If variant is 100%, can never be below threshold < 100%

    # Threshold count (maximum successes allowed to be below threshold)
    # Use max(0, ...) for safety.
    threshold_count = math.floor(max(0.0, p_threshold * n_samples))

    # We want P(X <= threshold_count) where X ~ Binom(n_samples, p_variant)
    # This is the cumulative distribution function cdf(k) = P(X <= k)
    try:
        # Ensure n is integer
        n_int = int(round(n_samples))
        if n_int <= 0: return 1.0
        # Use np.clip for p_variant safety
        p_safe = np.clip(p_variant, 0.0, 1.0)
        return stats.binom.cdf(threshold_count, n_int, p_safe)
    except (ValueError, OverflowError) as e:
        st.warning(f"Binomial calculation error (Power): n={n_samples:.1f}, k={threshold_count:.1f}, p={p_variant:.4f}. Error: {e}")
        return np.nan


# --- Experiment Design Function ---

@st.cache_data(ttl=3600) # Cache for 1 hour
def design_experiment(pA, pB, pError, pAUpper=None, pBLower=None, search_range=(100, 2_000_000), min_samples_for_calc=10):
    """
    Designs the A/B test sample sizes and generates plots.
    Finds the smallest integer N satisfying the error constraints using root finding.
    """
    if pAUpper is None: pAUpper = pB
    if pBLower is None: pBLower = pA

    nA = np.nan
    nB = np.nan
    final_sig_error = np.nan
    final_pow_error = np.nan

    # --- Find sample size nA (Control Significance) ---
    try:
        # Define the function whose root we want to find: error_func - target_error = 0
        # We want error <= pError, so find root of error - (pError - epsilon) = 0
        epsilon = 1e-9
        def objectiveA(k):
            # Ensure k is positive before calculating
            k = max(1.0, k) 
            err = calculate_significance_error(pA, pAUpper, k)
            if np.isnan(err): return np.inf # Return large value if calculation fails
            return err - (pError - epsilon)

        # Check if target error is met at the start of the range
        f_low_a = objectiveA(search_range[0])
        if np.isnan(f_low_a): f_low_a = np.inf # Assume failure if NaN

        if f_low_a <= 0:
             # Error target met at lower bound. Check if even smaller N works.
             # Search downwards from search_range[0] to min_samples_for_calc
             res = optimize.root_scalar(objectiveA, bracket=[min_samples_for_calc, search_range[0]], method='brentq', xtol=0.5)
             if res.converged:
                  nA_float = res.root
             else: # If downward search fails, use lower bound
                  nA_float = search_range[0]
                  if objectiveA(min_samples_for_calc) <= 0: # Double check minimum practical N
                       nA_float = min_samples_for_calc
             st.info(f"Significance target met near N={search_range[0]}. Smallest N found ≈ {nA_float:.0f}.")
        else:
             # Error target not met at lower bound, search within the main range
             f_high_a = objectiveA(search_range[1])
             if np.isnan(f_high_a): f_high_a = np.inf

             if f_high_a > 0: # Error target potentially not met even at upper bound
                  st.warning(f"Significance target {pError:.2E} may not be met in N range [{search_range[0]:,}, {search_range[1]:,}]. Error at N={search_range[1]:,} is approx {f_high_a + pError:.2E}.")
                  # Attempt root finding anyway; brentq might extrapolate or find root if function dips
             
             # Ensure the bracket has opposite signs for brentq, otherwise signal potential issue
             if f_low_a * f_high_a >= 0 and f_low_a > 0:
                  st.warning(f"Objective function for nA does not appear to cross zero in the search range [{search_range[0]:,}, {search_range[1]:,}]. Result might be inaccurate.")
                  # Fallback to max range if no sign change detected
                  nA_float = search_range[1] 
             else:
                  try:
                       # Brentq requires opposite signs if function is continuous and crosses zero
                       res = optimize.root_scalar(objectiveA, bracket=[search_range[0], search_range[1]], method='brentq', xtol=0.5)
                       if res.converged:
                           nA_float = res.root
                       else:
                            st.error("Root finding for nA did not converge.")
                            nA_float = np.nan
                  except ValueError as e_brentq: # Handle cases where signs are same despite checks
                        st.error(f"Brentq error for nA (likely same signs at bounds): {e_brentq}")
                        nA_float = np.nan # Indicate failure


        # We need the smallest integer n where error <= pError. Root is where error ≈ pError.
        # Ceiling ensures we meet the condition. Check result is valid.
        if not pd.isna(nA_float):
            nA = math.ceil(max(min_samples_for_calc, nA_float)) # Ensure at least min practical N
            final_sig_error = calculate_significance_error(pA, pAUpper, nA)
        else:
            nA = np.nan # Propagate failure

    except Exception as e:
        st.error(f"An unexpected error occurred finding nA: {e}")
        nA = np.nan


    # --- Find sample size nB (Variant Power) ---
    # Similar logic as for nA
    try:
        def objectiveB(k):
            k = max(1.0, k)
            err = calculate_power_error(pBLower, pB, k)
            if np.isnan(err): return np.inf
            return err - (pError - epsilon)

        f_low_b = objectiveB(search_range[0])
        if np.isnan(f_low_b): f_low_b = np.inf

        if f_low_b <= 0:
             res = optimize.root_scalar(objectiveB, bracket=[min_samples_for_calc, search_range[0]], method='brentq', xtol=0.5)
             if res.converged:
                  nB_float = res.root
             else:
                  nB_float = search_range[0]
                  if objectiveB(min_samples_for_calc) <= 0:
                       nB_float = min_samples_for_calc
             st.info(f"Power target met near N={search_range[0]}. Smallest N found ≈ {nB_float:.0f}.")
        else:
             f_high_b = objectiveB(search_range[1])
             if np.isnan(f_high_b): f_high_b = np.inf

             if f_high_b > 0:
                 st.warning(f"Power target {pError:.2E} may not be met in N range [{search_range[0]:,}, {search_range[1]:,}]. Error at N={search_range[1]:,} is approx {f_high_b + pError:.2E}.")

             if f_low_b * f_high_b >= 0 and f_low_b > 0:
                  st.warning(f"Objective function for nB does not appear to cross zero in the search range [{search_range[0]:,}, {search_range[1]:,}]. Result might be inaccurate.")
                  nB_float = search_range[1]
             else:
                  try:
                        res = optimize.root_scalar(objectiveB, bracket=[search_range[0], search_range[1]], method='brentq', xtol=0.5)
                        if res.converged:
                            nB_float = res.root
                        else:
                            st.error("Root finding for nB did not converge.")
                            nB_float = np.nan
                  except ValueError as e_brentq:
                        st.error(f"Brentq error for nB (likely same signs at bounds): {e_brentq}")
                        nB_float = np.nan


        if not pd.isna(nB_float):
            nB = math.ceil(max(min_samples_for_calc, nB_float))
            final_pow_error = calculate_power_error(pBLower, pB, nB)
        else:
            nB = np.nan

    except Exception as e:
        st.error(f"An unexpected error occurred finding nB: {e}")
        nB = np.nan

    # --- Generate Plot Data & Plot (only if N calculated) ---
    plot_fig = None
    if not (pd.isna(nA) or pd.isna(nB)):
        try:
            # Determine relevant range for counts based on expected values and std deviations
            stdA = math.sqrt(nA * pA * (1 - pA)) if nA > 0 and 0 < pA < 1 else 0
            stdB = math.sqrt(nB * pB * (1 - pB)) if nB > 0 and 0 < pB < 1 else 0
            buffer = 4.0 # Number of std devs for range

            meanA_count = nA * pA
            meanB_count = nB * pB

            low_count = math.floor(min(meanA_count - buffer * stdA, meanB_count - buffer * stdB))
            high_count = math.ceil(max(meanA_count + buffer * stdA, meanB_count + buffer * stdB))
            
            # Ensure range is sensible and within bounds [0, max(nA, nB)]
            low_count = max(0, low_count)
            max_n = max(nA, nB)
            high_count = min(max_n, max(low_count + 1, high_count)) # Ensure at least 2 points, don't exceed N

            count_range = np.arange(low_count, high_count + 1)

            # Create DataFrame for plotting A
            dfA = pd.DataFrame({'count': count_range[count_range <= nA]}) # Filter counts applicable to nA
            if not dfA.empty:
                 dfA['group'] = f'A (Control): N={nA:,}'
                 dfA['density'] = stats.binom.pmf(dfA['count'], nA, pA)
                 dfA['rate'] = dfA['count'] / nA
                 dfA['error'] = dfA['rate'] >= pAUpper # Significance error region
            else: dfA=None # Handle empty case

            # Create DataFrame for plotting B
            dfB = pd.DataFrame({'count': count_range[count_range <= nB]}) # Filter counts applicable to nB
            if not dfB.empty:
                 dfB['group'] = f'B (Variant): N={nB:,}'
                 dfB['density'] = stats.binom.pmf(dfB['count'], nB, pB)
                 dfB['rate'] = dfB['count'] / nB
                 dfB['error'] = dfB['rate'] <= pBLower # Power error region
            else: dfB = None # Handle empty case

            # Combine if both exist
            if dfA is not None and dfB is not None:
                 df_plot = pd.concat([dfA, dfB], ignore_index=True)
            elif dfA is not None: df_plot = dfA
            elif dfB is not None: df_plot = dfB
            else: df_plot = None

            # --- Generate Plot ---
            if df_plot is not None:
                groups = df_plot['group'].unique()
                num_groups = len(groups)
                plot_fig, axes = plt.subplots(num_groups, 1, figsize=(8, 4 * num_groups), sharex=True)
                if num_groups == 1: axes = [axes] # Make axes iterable if only one plot

                for i, group_name in enumerate(groups):
                    ax = axes[i]
                    df_group = df_plot[df_plot['group'] == group_name].copy() # Work on a copy
                    df_group.sort_values('rate', inplace=True) # Ensure rates are monotonic for filling

                    # Line plot for density
                    sns.lineplot(data=df_group, x='rate', y='density', ax=ax, color='black', linewidth=1)

                    # Fill error region
                    df_error = df_group[df_group['error']]
                    if not df_error.empty:
                        ax.fill_between(df_error['rate'], 0, df_error['density'], color='red', alpha=0.4, step='mid') # Use step for discrete

                    # Vertical lines for thresholds
                    ax.axvline(pAUpper, color='grey', linestyle='--', label=f'Upper Threshold ({pAUpper:.3f})')
                    ax.axvline(pBLower, color='darkgrey', linestyle=':', label=f'Lower Threshold ({pBLower:.3f})')

                    ax.set_title(group_name)
                    ax.set_ylabel('Probability Mass')
                    ax.tick_params(axis='y', labelsize=9)
                    ax.legend(fontsize='small', loc='upper right')
                    ax.grid(axis='y', linestyle=':', alpha=0.6)

                # Set labels only on the bottom plot if sharing x-axis
                axes[-1].set_xlabel('Observed Conversion Rate')
                # Format x-axis as percentage
                axes[-1].xaxis.set_major_formatter(plt.FuncFormatter('{:.1%}'.format))
                plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")


                # Remove x-labels from upper plots if multiple groups
                if num_groups > 1:
                    for i in range(num_groups - 1):
                        axes[i].set_xlabel('')
                        axes[i].tick_params(axis='x', labelbottom=False)

                plot_fig.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout slightly
                plot_fig.suptitle('Binomial Distributions at Calculated Sample Sizes', y=0.99)


        except Exception as plot_e:
            st.error(f"Error generating design plot: {plot_e}")
            plot_fig = None

    return {"nA": nA, "nB": nB, "plot": plot_fig, "final_sig_error": final_sig_error, "final_pow_error": final_pow_error}


# --- Bayesian A/B Test Functions ---

@st.cache_data(ttl=3600)
def calculate_prob_b_beats_a(success_a, visits_a, success_b, visits_b, num_samples=25000):
    """Calculates P(B > A) using Beta distribution posteriors and sampling."""
    if not (visits_a > 0 and visits_b > 0 and 0 <= success_a <= visits_a and 0 <= success_b <= visits_b):
        st.warning("Invalid inputs for Bayesian P(B>A) calculation.")
        return np.nan

    # Beta distribution parameters (alpha = successes + 1, beta = failures + 1)
    alpha_a = success_a + 1
    beta_a = visits_a - success_a + 1
    alpha_b = success_b + 1
    beta_b = visits_b - success_b + 1

    # Sample from the posterior distributions
    # Add error handling for potential issues in rvs (e.g., large alpha/beta)
    try:
        with warnings.catch_warnings(): # Suppress potential integration warnings in beta.rvs
            warnings.simplefilter("ignore", category=RuntimeWarning)
            samples_a = stats.beta.rvs(alpha_a, beta_a, size=num_samples)
            samples_b = stats.beta.rvs(alpha_b, beta_b, size=num_samples)
    except Exception as rvs_e:
        st.error(f"Error sampling from Beta distributions: {rvs_e}")
        return np.nan

    # Calculate the probability
    prob = np.mean(samples_b > samples_a)
    return prob

@st.cache_data(ttl=3600)
def get_beta_plot_data(success_a, visits_a, success_b, visits_b, num_points=500):
    """Prepares data for plotting individual Beta distributions."""
    if not (visits_a > 0 and visits_b > 0 and 0 <= success_a <= visits_a and 0 <= success_b <= visits_b):
        st.warning("Invalid inputs for Beta plot data generation.")
        return None, None

    alpha_a = success_a + 1
    beta_a = visits_a - success_a + 1
    alpha_b = success_b + 1
    beta_b = visits_b - success_b + 1

    # Determine a sensible range for x-axis based on both distributions
    mean_a = alpha_a / (alpha_a + beta_a)
    mean_b = alpha_b / (alpha_b + beta_b)
    # Use ppf (percent point function) for robust range finding, e.g., 0.1% to 99.9% quantiles
    q_low = 0.0001
    q_high = 0.9999
    min_x = max(0.0, min(stats.beta.ppf(q_low, alpha_a, beta_a), stats.beta.ppf(q_low, alpha_b, beta_b)))
    max_x = min(1.0, max(stats.beta.ppf(q_high, alpha_a, beta_a), stats.beta.ppf(q_high, alpha_b, beta_b)))

    # Ensure range has some width, especially if distributions are very narrow or identical
    if max_x - min_x < 1e-6:
        width = max(0.01, (mean_a + mean_b) * 0.05) # Heuristic width
        min_x = max(0.0, (mean_a + mean_b)/2 - width/2)
        max_x = min(1.0, (mean_a + mean_b)/2 + width/2)
    if max_x == min_x: max_x = min(1.0, min_x + 0.001) # Final check

    x = np.linspace(min_x, max_x, num_points)

    pdf_a = stats.beta.pdf(x, alpha_a, beta_a)
    pdf_b = stats.beta.pdf(x, alpha_b, beta_b)

    df = pd.DataFrame({'rate': x, 'pdf_a': pdf_a, 'pdf_b': pdf_b})
    df_melted = df.melt(id_vars='rate', var_name='group', value_name='density')
    df_melted['group'] = df_melted['group'].map({'pdf_a': f'Control (A): {success_a}/{visits_a}',
                                                 'pdf_b': f'Variant (B): {success_b}/{visits_b}'})
    limits = {'min_x': min_x, 'max_x': max_x}
    return df_melted, limits

@st.cache_data(ttl=3600)
def plot_beta_distributions(df_melted, limits):
    """Plots the individual Beta distributions."""
    if df_melted is None or limits is None:
        st.warning("Cannot plot individual Beta distributions due to invalid input data.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df_melted, x='rate', y='density', hue='group', ax=ax)
    ax.set_title('Posterior Distributions of Conversion Rates')
    ax.set_xlabel('Conversion Rate')
    ax.set_ylabel('Probability Density')
    ax.legend(title='Group')
    ax.set_xlim(limits['min_x'], limits['max_x'])
    ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.1%}'.format)) # Format x-axis as percentage
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    fig.tight_layout()
    return fig

@st.cache_data(ttl=3600)
def plot_joint_beta_distribution(success_a, visits_a, success_b, visits_b, limits=None, grid_size=100):
    """Plots the joint probability density of the two Beta distributions."""
    if not (visits_a > 0 and visits_b > 0 and 0 <= success_a <= visits_a and 0 <= success_b <= visits_b):
        st.warning("Invalid inputs for joint Beta plot generation.")
        return None

    alpha_a = success_a + 1
    beta_a = visits_a - success_a + 1
    alpha_b = success_b + 1
    beta_b = visits_b - success_b + 1

    # Determine range if not provided
    if limits:
        min_rate = limits['min_x']
        max_rate = limits['max_x']
    else:
        # Fallback range calculation
        q_low = 0.0001
        q_high = 0.9999
        min_rate = max(0.0, min(stats.beta.ppf(q_low, alpha_a, beta_a), stats.beta.ppf(q_low, alpha_b, beta_b)))
        max_rate = min(1.0, max(stats.beta.ppf(q_high, alpha_a, beta_a), stats.beta.ppf(q_high, alpha_b, beta_b)))
        if max_rate - min_rate < 1e-6:
            mean_a = alpha_a / (alpha_a + beta_a)
            mean_b = alpha_b / (alpha_b + beta_b)
            width = max(0.01, (mean_a + mean_b) * 0.05)
            min_rate = max(0.0, (mean_a + mean_b)/2 - width/2)
            max_rate = min(1.0, (mean_a + mean_b)/2 + width/2)
        if max_rate == min_rate: max_rate = min(1.0, min_rate + 0.001)

    # Create grid
    rate_a_grid_vals = np.linspace(min_rate, max_rate, grid_size)
    rate_b_grid_vals = np.linspace(min_rate, max_rate, grid_size)
    grid_a, grid_b = np.meshgrid(rate_a_grid_vals, rate_b_grid_vals)

    # Calculate PDF on grid
    pdf_a = stats.beta.pdf(grid_a, alpha_a, beta_a)
    pdf_b = stats.beta.pdf(grid_b, alpha_b, beta_b)
    joint_pdf = pdf_a * pdf_b # Joint density for independent variables

    # Plotting
    fig, ax = plt.subplots(figsize=(7, 6.5)) # Slightly taller for labels

    # Use contourf for filled contours - often better than imshow for densities
    contour_levels = np.linspace(0, joint_pdf.max(), 10) # Adjust number of levels as needed
    cf = ax.contourf(rate_a_grid_vals, rate_b_grid_vals, joint_pdf, levels=contour_levels, cmap='Greens', alpha=0.9)

    # Add contour lines for definition
    ax.contour(rate_a_grid_vals, rate_b_grid_vals, joint_pdf, levels=contour_levels[1:], colors='black', alpha=0.2, linewidths=0.5)


    # Add diagonal line (Rate A = Rate B)
    ax.plot([min_rate, max_rate], [min_rate, max_rate], color='black', linestyle='-', linewidth=1.0, label='Rate A = Rate B')

    ax.set_xlabel(f'Control Rate (A) - {success_a}/{visits_a}')
    ax.set_ylabel(f'Variant Rate (B) - {success_b}/{visits_b}')
    ax.set_title('Joint Probability Density of Rates')

    ax.set_xlim(min_rate, max_rate)
    ax.set_ylim(min_rate, max_rate)

    # Format axes as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.1%}'.format))
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.1%}'.format))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


    # Add a color bar
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Joint Probability Density')
    # Ensure colorbar ticks are reasonable
    cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))


    fig.tight_layout()
    return fig


# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="A/B Test Assistant")

st.title("📈 A/B Test Design and Analysis Assistant")
st.markdown("""
This tool assists with two key phases of A/B testing:

1.  **Experiment Design (Frequentist):** Calculate required sample sizes (`N`) per group based on baseline rate, desired lift, significance level (`alpha`), and statistical power (`1-beta`).
2.  **Result Analysis (Bayesian):** Estimate the probability that variant B is truly better than control A, given the observed visits and successes. Visualize the uncertainty using posterior distributions.
""")
st.markdown("---")

# --- Sidebar Inputs ---
st.sidebar.header("📊 Experiment Design")
st.sidebar.markdown("Parameters to calculate required sample sizes.")

with st.sidebar.form("design_form"):
    st.markdown("**Input Rates & Error**")
    pA_input = st.number_input("Baseline Conversion Rate (Control, pA)", min_value=0.0, max_value=0.999, value=0.10, step=0.001, format="%.4f", help="The current conversion rate of the control group.")
    # Allow specifying lift instead of absolute pB
    lift_type = st.radio("Specify Variant Target By:", ["Absolute Rate (pB)", "Relative Lift (%)", "Absolute Lift (%)"], horizontal=True, index=0)

    pB_input_val = 0.12 # Default
    if lift_type == "Absolute Rate (pB)":
        pB_input_val = st.number_input("Target Conversion Rate (Variant, pB)", min_value=0.0001, max_value=1.0, value=0.12, step=0.001, format="%.4f", help="The desired conversion rate you want to detect for the variant.")
    elif lift_type == "Relative Lift (%)":
        relative_lift_pct = st.number_input("Minimum Detectable Relative Lift (%)", min_value=0.1, value=20.0, step=0.5, format="%.1f", help="The minimum percentage increase over baseline (pA) you want to detect (e.g., 10% means pB = pA * 1.10).")
        pB_input_val = pA_input * (1 + relative_lift_pct / 100.0)
        st.caption(f"Calculated Target Rate (pB): `{pB_input_val:.5f}`")
    elif lift_type == "Absolute Lift (%)":
        absolute_lift_pct = st.number_input("Minimum Detectable Absolute Lift (% points)", min_value=0.01, value=2.0, step=0.1, format="%.2f", help="The minimum absolute increase in percentage points over baseline (pA) you want to detect (e.g., 2% means pB = pA + 0.02).")
        pB_input_val = pA_input + absolute_lift_pct / 100.0
        st.caption(f"Calculated Target Rate (pB): `{pB_input_val:.5f}`")
        
    # Ensure pB is valid after calculation
    pB_input = np.clip(pB_input_val, 0.00001, 1.0) # Clip pB to valid range after lift calculation


    pError_input = st.select_slider("Acceptable Error Rate (alpha/beta)", options=[0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20], value=0.05, help="Sets both the maximum Type I error (alpha, significance level) and Type II error (beta, 1-power). E.g., 0.05 means 95% significance and 95% power.")

    st.markdown("**Threshold Setting**")
    concurrent_input = st.checkbox("Use Midpoint Thresholds?", value=False, help="Check to use the average of pA and pB as the decision threshold for both error calculations. Default uses pB as threshold for significance (A vs pB) and pA as threshold for power (B vs pA).")
    design_button = st.form_submit_button("Calculate Sample Sizes", type="primary")


st.sidebar.header("🔍 Bayesian Analysis")
st.sidebar.markdown("Observed results to calculate P(B > A).")

with st.sidebar.form("bayes_form"):
    st.markdown("**Control Group (A)**")
    visits_a_input = st.number_input("Visits (A)", min_value=1, value=10000, step=100, help="Total observations for the control group.")
    successes_a_input = st.number_input("Successes (A)", min_value=0, value=1000, step=10, help="Number of successful conversions for the control group.")

    st.markdown("**Variant Group (B)**")
    visits_b_input = st.number_input("Visits (B)", min_value=1, value=10000, step=100, help="Total observations for the variant group.")
    successes_b_input = st.number_input("Successes (B)", min_value=0, value=1200, step=10, help="Number of successful conversions for the variant group.")
    bayes_button = st.form_submit_button("Analyze Results", type="primary")


# --- Main Panel Outputs ---
st.subheader("1. Experiment Design Results")
if design_button:
    # Validation
    input_valid = True
    if pA_input >= pB_input:
        st.error("Target Rate (pB) must be greater than Baseline Rate (pA). Adjust inputs.")
        input_valid = False
    if pA_input <=0 or pB_input <= 0 or pA_input >=1 or pB_input > 1:
         st.warning("Baseline and Target Rates should be between 0 and 1 (exclusive of 0 for pA/pB, exclusive of 1 for pA). Clamping values.")
         pA_input = np.clip(pA_input, 0.00001, 0.99999)
         pB_input = np.clip(pB_input, pA_input + 1e-6, 1.0) # Ensure pB > pA after clipping

    if input_valid:
        pAUpper_calc = pB_input
        pBLower_calc = pA_input
        threshold_desc = f"Thresholds: A vs {pB_input:.4f}, B vs {pA_input:.4f}"
        if concurrent_input:
            mid_rate = (pA_input + pB_input) / 2.0
            pAUpper_calc = mid_rate
            pBLower_calc = mid_rate
            threshold_desc = f"Thresholds: Midpoint = {mid_rate:.4f}"

        st.markdown(f"**Calculating N with:** Baseline pA=`{pA_input:.4f}`, Target pB=`{pB_input:.4f}`, Target Error=`{pError_input:.3f}` ({1-pError_input:.1%} Power/Significance)")
        st.markdown(f"Using {threshold_desc}")

        with st.spinner("⚙️ Calculating required sample sizes... This can take a few seconds."):
             design_results = design_experiment(pA_input, pB_input, pError_input, pAUpper_calc, pBLower_calc)

        if not (pd.isna(design_results['nA']) or pd.isna(design_results['nB'])):
            col1, col2 = st.columns(2)
            with col1:
                 st.metric("Required N (Control A)", value=f"{design_results['nA']:,}")
                 st.caption(f"Est. Sig. Error: {design_results.get('final_sig_error', 'N/A'):.2E}")
            with col2:
                 st.metric("Required N (Variant B)", value=f"{design_results['nB']:,}")
                 st.caption(f"Est. Power Error: {design_results.get('final_pow_error', 'N/A'):.2E}")

            if design_results["plot"] is not None:
                st.pyplot(design_results["plot"])
            else:
                 st.warning("Could not generate experiment design plot.")
        else:
             st.error("Failed to calculate valid sample sizes. Check warnings above or adjust inputs (e.g., make rates further apart, increase error tolerance, or check search range if modified).")
else:
    st.info("Configure parameters in the sidebar and click 'Calculate Sample Sizes' to design an experiment.")

st.markdown("---")
st.subheader("2. Bayesian Analysis Results")

if bayes_button:
    # Validation
    valid_bayes = True
    rate_a_obs = 0
    rate_b_obs = 0
    if visits_a_input <= 0 or visits_b_input <= 0:
        st.error("Visits must be greater than zero.")
        valid_bayes = False
    if successes_a_input > visits_a_input:
        st.error("Control successes cannot exceed control visits.")
        valid_bayes = False
    if successes_b_input > visits_b_input:
        st.error("Variant successes cannot exceed variant visits.")
        valid_bayes = False
    if successes_a_input < 0 or successes_b_input < 0:
        st.error("Successes cannot be negative.")
        valid_bayes = False

    if valid_bayes:
        rate_a_obs = successes_a_input / visits_a_input
        rate_b_obs = successes_b_input / visits_b_input
        st.markdown(f"**Analyzing:** Control (A): `{successes_a_input}/{visits_a_input}` (Rate: `{rate_a_obs:.2%}`), Variant (B): `{successes_b_input}/{visits_b_input}` (Rate: `{rate_b_obs:.2%}`)")

        with st.spinner("🎲 Running Bayesian simulation and plotting..."):
            prob_b_beats_a = calculate_prob_b_beats_a(successes_a_input, visits_a_input, successes_b_input, visits_b_input)
            beta_df, beta_limits = get_beta_plot_data(successes_a_input, visits_a_input, successes_b_input, visits_b_input)
            beta_plot_fig = plot_beta_distributions(beta_df, beta_limits) if beta_df is not None else None
            joint_plot_fig = plot_joint_beta_distribution(successes_a_input, visits_a_input, successes_b_input, visits_b_input, limits=beta_limits) if beta_limits is not None else None

        if not pd.isna(prob_b_beats_a):
             # Display P(B>A) prominently
             st.metric("Probability [Variant (B) > Control (A)]", value=f"{prob_b_beats_a:.2%}",
                       help="Based on sampling from the posterior Beta distributions. This indicates the belief that the underlying rate of B is higher than A, given the data.")

             # Display plots
             plot_col1, plot_col2 = st.columns(2)
             with plot_col1:
                 if beta_plot_fig: st.pyplot(beta_plot_fig)
                 else: st.warning("Could not generate individual posterior plot.")
             with plot_col2:
                 if joint_plot_fig: st.pyplot(joint_plot_fig)
                 else: st.warning("Could not generate joint posterior plot.")
        else:
             st.error("Bayesian calculation failed. Check inputs or potential calculation errors noted above.")
else:
    st.info("Enter observed results in the sidebar and click 'Analyze Results' for Bayesian analysis.")


# Add footer
st.sidebar.markdown("---")
st.sidebar.info("A/B Test Assistant | Uses Streamlit, SciPy, Matplotlib | v1.2")
# Add link to uv or astral
st.sidebar.markdown("[Powered by fast tooling like `uv` from Astral](https://astral.sh/uv)")
