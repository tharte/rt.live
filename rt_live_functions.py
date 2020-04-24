import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scipy import stats as sps
from scipy.interpolate import interp1d

FILTERED_REGION_CODES = ['AS', 'GU', 'PR', 'VI', 'MP']

def highest_density_interval(pmf, p=.9, debug=False):
    # If we pass a DataFrame, just call this recursively on the columns
    if (isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame(
            [highest_density_interval(pmf[col], p=p) for col in pmf],
            index=pmf.columns
        )

    cumsum = np.cumsum(pmf.values)

    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]

    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()

    # Find the smallest range (highest density)
    best = (highs - lows).argmin()
    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]

    return pd.Series(
        [low, high],
        index=[f'Low_{p*100:.0f}',
        f'High_{p*100:.0f}']
    )

def prepare_cases(cases, cutoff=25):
    new_cases = cases.diff()

    smoothed = new_cases.rolling(
        7,
        win_type='gaussian',
        min_periods=1,
        center=True
    ).mean(std=2).round()

    idx_start = np.searchsorted(smoothed, cutoff)

    smoothed = smoothed.iloc[idx_start[0]:]
    original = new_cases.loc[smoothed.index]

    return original, smoothed
