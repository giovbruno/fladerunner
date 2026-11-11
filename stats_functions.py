import numpy as np
from scipy import stats
import copy
import pandas as pd
from pdb import set_trace

def compute_t_value(r, df, n, alternative='two-sided'):
    """
    Parameters
    -----------
    r: correlation coefficient
    df (int): degrees of freedom (n - 2 for correlation).
    n (int): sample size
    alternative (str): 'two-sided', 'less' o 'greater' .
    """

    t_calc = r*np.sqrt((n - 2.)/(1. - r**2))

    if alternative == 'two-sided':
        p_val = 2 * (1 - stats.t.cdf(abs(t_calc), df))
    elif alternative == 'greater':
        p_val = 1 - stats.t.cdf(t_calc, df)
    elif alternative == 'less':
        p_val = stats.t.cdf(t_calc, df)

    return p_val

def weighted_bootstrap_diff_correlation(X1, X2, n_bootstraps=1000, CI=68.):
    '''
    Bootstrap test for the difference in correlation coefficients with weighted
    resampling.

    Parameters
    ----------
    - x1, y1: Arrays for the first correlation.
    - x2, y2: Arrays for the second correlation. If y2 == None, the correlation
            is computed for random pairs of the same array.
    - w1, w2: Arrays of weights for the two datasets. If None, uniform weights
              are used.
    - n_bootstraps: Number of bootstrap iterations.

    Returns
    -------
    - p_value: Two-sided p-value for H0: rho1 = rho2.
    - observed_diff: Observed difference in correlations.
    - null_diffs: Bootstrap distribution of differences under H0.
    '''

    # Convert weights to numpy arrays if not provided
    if 'w' not in X1.keys():
        X1['w'] = np.ones(len(X1))
    if 'w' not in X2.keys():
        X2['w'] = np.ones(len(X2))

    data1 = np.column_stack([X1['z1'], X1['z2']])
    if 'z2' not in X2.keys():
        data2 = np.column_stack([X2['z1'], X2['z1']])
    else:
        data2 = np.column_stack([X2['z1'], X2['z2']])
    weights1 = X1['w']
    weights2 = X2['w']

    null_diffs = []
    rho1, rho2 = [], []
    rng = np.random.default_rng()
    for bi in range(n_bootstraps):
        if bi % 100 == 0:
            print(bi)
        # Resample each dataset independently with their weights
        x1_boot, y1_boot = rng.choice([X1['z1'], X1['z2']], size=len(X1), \
                    replace=True, p=X1['w'], axis=1)
        if 'z2' not in X2.keys():
            zz = X2.groupby('ticname', group_keys=False).apply( \
                pair_with_random).reset_index(drop=True)
            # Remove symmetric rows (x, y) = (y, x)
            # Create a canonical key for each pair
            X2_new = pd.concat([X2, zz], axis=1)
            # Remove pairs (x, y) = (y, x)
            cols_to_check = ['scaled1', 'scaled2']
            sorted_pairs = np.sort(X2_new[cols_to_check].values, axis=1)
            # Keep only the first occurrence of each unique pair
            X2_temp = pd.DataFrame(sorted_pairs, columns=cols_to_check)
            X2_cleaned = X2_new[~X2_temp.duplicated(keep="first")]
            X2_cleaned = X2_cleaned[X2_cleaned['scaled1'] \
                        != X2_cleaned['scaled2']]
            # Time distance constraint
            deltat = abs(X2_cleaned['Peak time [BTJD]'] \
                        - X2_cleaned['Paired Peak time [BTJD]'])*24
            X2_cleaned = X2_cleaned[deltat >= 1.]
            x2_boot, y2_boot = X2_cleaned['scaled1'], X2_cleaned['scaled2']
        else:
            x2_boot, y2_boot = rng.choice([x2, y2], size=len(x2), \
                    replace=True, p=weights2/np.sum(weights2), axis=1)

        # Compute non-weighted correlations for bootstrapped samples
        # (weights are assigned by sampling)
        rho1_boot, _ = stats.spearmanr(x1_boot, y1_boot)
        rho2_boot, _ = stats.spearmanr(x2_boot, y2_boot)
        diff = rho1_boot - rho2_boot
        rho1.append(rho1_boot)
        rho2.append(rho2_boot)
        null_diffs.append(diff)

    # Confidence intervals
    CImin = (100. - CI)/2.
    CImax = CImin + CI
    perc = [CImin, 50., CImax]
    rho1_CI = np.percentile(rho1, perc)
    rho2_CI = np.percentile(rho2, perc)
    null_diffs_CI95 = np.percentile(null_diffs, perc)

    # Return also a realisation for the second sample, for plotting
    return rho1, rho2, null_diffs_CI95, [x2_boot, y2_boot]

def pair_with_random(group, par='log_ED', w='weight', \
            otherpar='Peak time [BTJD]'):
    '''
    Pair each row with a randomly selected row from the same group.

    Parameters
    ----------
    otherpar: any control parameter to associated to the shuffled wors
    '''
    n = len(group)
    paired_values = []
    paired_t = []

    for i in range(n):
        candidates = [j for j in range(n) if (j != i and j != i - 1 \
                    and j != i + 1)]
        ww = np.array(group[w])[candidates]
        paired_idx = np.random.choice(candidates, p=ww/np.sum(ww))
        paired_values.append(group[par].values[paired_idx])
        if len(otherpar) > 0:
            paired_t.append(group[otherpar].values[paired_idx])

    # Compute Z-score to avoid Simpson paradox
    group_mean = group[par].mean()
    group_std = group[par].std()
    paired_mean = np.nanmean(paired_values)
    paired_std = np.nanstd(paired_values)

    if len(otherpar) > 0:
        return pd.DataFrame({
            'scaled1': (group[par].values - group_mean)/group_std,
            'scaled2': (paired_values - paired_mean)/paired_std,
            'Paired ' + otherpar: paired_t})
    else:
        return pd.DataFrame({
            'scaled1': (group[par].values - group_mean)/group_std,
            'scaled2': (paired_values - paired_mean)/paired_std})

def weighted_corr(x, y, w, kind='spearman'):
    '''
    Compute weighted correlation coefficient (Spearman or Pearson).
    '''
    w = np.asarray(w)
    x = np.asarray(x)
    y = np.asarray(y)

    if kind == 'spearman':
        # Compute ranks
        x = stats.rankdata(x)
        y = stats.rankdata(y)

    # Weighted means
    w_sum = np.sum(w)
    mean_x = np.sum(w * x) / w_sum
    mean_y = np.sum(w * y) / w_sum

    # Weighted covariance and variances
    dx = x - mean_x
    dy = y - mean_y

    cov_xy = np.sum(w * dx * dy) / w_sum
    var_x = np.sum(w * dx**2) / w_sum
    var_y = np.sum(w * dy**2) / w_sum

    # Correlation
    corr = cov_xy / np.sqrt(var_x * var_y)
    p_value = compute_t_value(corr, len(x) - 2, len(x))

    # Handle edge cases
    if var_x == 0 or var_y == 0:
         # No variance, return 0 correlation and undefined p-value
        return 0.0, 1.0
    # Return correlation, p-value placeholder
    return corr, p_value

def monte_carlo_permutation(x_obs, y_obs, sigma_y, n_mc=1000, n_perm=1000):
    '''
    Simulate many data sets within uncertainties and for each derive a
    Spearman correlation coefficient with its p-value.
    '''

    rng = np.random.default_rng()
    p_values = []
    print('Monte-Carlo permutation test...')
    for j in range(n_mc):
        if j % 10 == 0:
            print('Iter:', j)
        # Perturb observed data by measurement uncertainties
        y_perturbed = y_obs + rng.normal(0., sigma_y, len(y_obs))
        # Compute observed Spearman correlation on perturbed data
        obs_rho, _ = stats.spearmanr(x_obs, y_perturbed, alternative='less')
        # Build null distribution by permuting y_perturbed
        perm_rhos = []
        for i in range(n_perm):
            y_perm = rng.permutation(y_perturbed)
            rho_perm, _ = stats.spearmanr(x_obs, y_perm)
            perm_rhos.append(rho_perm)
        perm_rhos = np.array(perm_rhos)

        # Two-sided p-value for this MC iteration
        p_val = np.mean(np.abs(perm_rhos) >= np.abs(obs_rho))
        p_values.append(p_val)

    # Aggregate p-values (median)
    final_p_value = np.median(p_values)

    return final_p_value

def permutated_skew(df, par, weight, refvalue, niter=1000):
    '''
    How likely is it that a random selection of ratios shows the same level
    of asymmetry?
    There's no more need of using "shift(1)", as we are not looking for
    consecutive flares anymore
    '''
    l1 = len(df)

    stat_r = []
    for i in range(niter):
        random_ratio = np.log10(np.random.choice(df[par], size=l1) \
            / np.random.choice(df[par], p=weight, size=l1))
        stat_r.append(abs(stats.skew(random_ratio, nan_policy='omit')))
    pvalue = np.nansum(np.array(stat_r) >= refvalue)/niter
    return pvalue

def random_pais_correlation(x, weights, CI=95.):
    '''
    Correlation and p-value for random pais of elements.

    Compares n_iter random pairs of events and then generates random
    distributions to compare.
    '''
    N_pairs = 2000
    n_iter = 1000

    real_corr = []
    for it in range(n_iter):
        indices = np.random.choice(len(x), size=(N_pairs, 2), replace=True, \
                    p=weights)
        r, _ = stats.pearsonr(x[indices[:, 0]], x[indices[:, 1]])
        real_corr.append(r)
    CI_min = (100. - CI)/2.
    CI_max = CI_min + CI
    r_obs_95 = np.percentile(real_corr, [CI_min, CI_max])
    print('Confidence interval within {:.1f}-{:.1f}%: {:.3f}-{:.3f}'.format( \
                CI_min, CI_max, r_obs_95[0], r_obs_95[1]))

    return r_obs_95

def z_score(x, log=True):
    '''
    Normalise a data set with z score. Log-scale data if the data if it's
    ~log-normally distributed.
    '''
    if log:
        x = np.log10(x)

    mm = np.nanmean(x)
    ss = np.nanstd(x)
    return (x - mm)/ss

def z_score_group(df, par, group):
    '''
    Z score applied to dataframe groups.
    '''
    mu = df.groupby(group)[par].transform('mean')
    std = df.groupby(group)[par].transform('std')

    x = df[par]
    for i in range(len(par)):
        x['z' + str(i)] = ((x - mu)/std)[par[i]]
        #xmu = x.sub(mu)
        #xdiv = xmu.div(std)

    return x, mu, std
