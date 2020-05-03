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

def make_plottable_df(series, name):
    date = pd.to_datetime(
        series.index, infer_datetime_format=True, 
        errors='coerce'
    )
    df = pd.DataFrame(
        {name: series.values}
    )
    df.index = date
    return df

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

    if not len(lows) or not len(highs):
        low, high = (np.NaN, np.NaN)

    else:
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

def get_posteriors(sr, sigma=0.15, gamma=1/7, r_t_range=None, R_T_MAX=12):
    if r_t_range is None:
        r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(gamma * (r_t_range[:, None] - 1))

    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:]
    )

    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(
        loc=r_t_range,
        scale=sigma
   ).pdf(r_t_range[:, None]) 

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)

    # (4) Calculate the initial prior
    #prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 = np.ones_like(r_t_range)/len(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )

    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]

        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior

        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)

        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator

        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)

    return posteriors, log_likelihood

def plot_rt(result, ax, state_name, fig):
    ax.set_title(f"{state_name}")
    # Colors
    ABOVE = [1,0,0]
    MIDDLE = [1,1,1]
    BELOW = [0,0,0]
    cmap = ListedColormap(
        np.r_[
            np.linspace(0,1,25),
            np.linspace(1,1,25)
        ]
    )
    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5
    index = result['ML'].index.get_level_values('date')
    values = result['ML'].values
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
        values,
        s=40,
        lw=.5,
        # c=cmap(color_mapped(values)),
        edgecolors='k', zorder=2
    )
    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(date2num(index),
        result['Low_90'].values,
        bounds_error=False,
        fill_value='extrapolate'
    )
    highfn = interp1d(date2num(index),
        result['High_90'].values,
        bounds_error=False,
        fill_value='extrapolate'
    )
    extended = pd.date_range(
        start=pd.Timestamp('2020-03-01'), end=index[-1]+pd.Timedelta(days=1)
    )
    ax.fill_between(
        extended,
        lowfn(date2num(extended)),
        highfn(date2num(extended)),
        color='k',
        alpha=.1,
        lw=0,
        zorder=3
    )
    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);
    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0, 5.0)
    ax.set_xlim(
        pd.Timestamp('2020-03-01'), 
        result.index.get_level_values('date')[-1]+pd.Timedelta(days=1)
    )
    fig.set_facecolor('w')
    
def plot_standings(mr, figsize=None, title='Most Recent $R_t$ by State'):
    if not figsize:
        figsize = ((15.9/50)*len(mr)+.1,2.5)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    err = mr[['Low_90', 'High_90']].sub(mr['ML'], axis=0).abs()
    bars = ax.bar(
        mr.index,
        mr['ML'],
        width=.825,
        color=FULL_COLOR,
        ecolor=ERROR_BAR_COLOR,
        capsize=2,
        error_kw={'alpha':.5, 'lw':1},
        yerr=err.values.T
    )
    for bar, state_name in zip(bars, mr.index):
        if state_name in no_lockdown:
            bar.set_color(NONE_COLOR)
        if state_name in partial_lockdown:
            bar.set_color(PARTIAL_COLOR)
    labels = mr.index.to_series().replace({'District of Columbia':'DC'})
    ax.set_xticklabels(labels, rotation=90, fontsize=11)
    ax.margins(0)
    ax.set_ylim(0,2.)
    ax.axhline(1.0, linestyle=':', color='k', lw=1)
    leg = ax.legend(
        handles=[
            Patch(label='Full', color=FULL_COLOR),
            Patch(label='Partial', color=PARTIAL_COLOR),
            Patch(label='None', color=NONE_COLOR)
        ],
        title='Lockdown',
        ncol=3,
        loc='upper left',
        columnspacing=.75,
        handletextpad=.5,
        handlelength=1
    )
    leg._legend_box.align = "left"
    fig.set_facecolor('w')
    return fig, ax

