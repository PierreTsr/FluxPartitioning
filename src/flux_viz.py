from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


def diag_line(x, y, ax, color='black', xy=(.05, .76)):
    # remove NAN values in x, y series
    x, y = x.to_frame(), y.to_frame()
    x_y = pd.concat([x, y], axis=1)

    if x_y.isnull().values.any():
        x_y = x_y.dropna()

    x, y = x_y.iloc[:, 0], x_y.iloc[:, 1]

    max_value = max(max(x), max(y))
    min_value = min(min(x), min(y))
    mean_value = np.mean(x)

    line = np.linspace(min_value, max_value, 100)
    ax.plot(line, line, '--', color=color)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    RMSE = (((y - x) ** 2).mean()) ** .5

    ax.annotate(
        f'$N$ = {len(x)} \n$R^{2}$ = {np.around(r_value ** 2, decimals=3)} \n$RMSE$ = {np.around(RMSE, decimals=3)}',
        xy=xy, xycoords='axes fraction')


def quad_viz(train_df, test_df, key, reference="canopy", filename=None, colors=None, bayesian=True, **kwargs):
    if bayesian:
        fig, ax = plt.subplots(figsize=(25, 10), nrows=2, ncols=3, gridspec_kw={'width_ratios': [3, 1, 1]})
    else:
        fig, ax = plt.subplots(figsize=(20, 10), nrows=2, ncols=2, gridspec_kw={'width_ratios': [3, 1]})

    s_circle = 4
    cmap = 'RdBu_r'
    alpha = 0.6

    ref = key + "_" + reference
    map = key + "_MAP"
    mean = key + "_mean"
    sigma = key + "_sigma"
    nn = key + "_NN"

    ax[0, 0].scatter(train_df.index, train_df[ref], label=ref, s=s_circle)
    if bayesian:
        ax[0, 0].scatter(train_df.index, train_df[map], label=map, s=s_circle, alpha=0.6)
        ax[0, 0].scatter(train_df.index, train_df[mean], label=mean, s=s_circle, alpha=0.6)
        lower = train_df[mean] - 2 * train_df[sigma]
        upper = train_df[mean] + 2 * train_df[sigma]
        where = [True] * train_df.shape[0]
        where[(train_df.index < datetime(2014, 1, 1)).sum() - 1] = False
        ax[0, 0].fill_between(train_df.index, lower, upper, where=where, facecolor="red", alpha=0.5,
                              label="Two std band")
    else:
        ax[0, 0].scatter(train_df.index, train_df[nn], label=nn, s=s_circle, alpha=0.6)
    ax[0, 0].set_ylabel(key)
    ax[0, 0].set_title('Training Set', fontsize=14, fontweight='bold')

    ax[1, 0].scatter(test_df.index, test_df[ref], label=ref, s=s_circle)
    if bayesian:
        ax[1, 0].scatter(test_df.index, test_df[map], label=map, s=s_circle, alpha=0.6)
        ax[1, 0].scatter(test_df.index, test_df[mean], label=mean, s=s_circle, alpha=0.6)
        lower = test_df[mean] - 2 * test_df[sigma]
        upper = test_df[mean] + 2 * test_df[sigma]
        where = [True] * train_df.shape[0]
        where[(train_df.index < datetime(2014, 1, 1)).sum() - 1] = False
        ax[1, 0].fill_between(test_df.index, lower, upper, facecolor="red", alpha=0.5, label="Two std band")
    else:
        ax[1, 0].scatter(test_df.index, test_df[nn], label=nn, s=s_circle, alpha=0.6)
    ax[1, 0].set_ylabel(key)
    ax[1, 0].set_title('Test Set', fontsize=14, fontweight='bold')

    train_df.plot(map, ref, alpha=alpha, ax=ax[0, 1], kind='scatter', s=15, c=colors, cmap=cmap)
    test_df.plot(map, ref, alpha=alpha, ax=ax[1, 1], kind='scatter', s=15, c=colors, cmap=cmap)
    if bayesian:
        train_df.plot(mean, ref, alpha=alpha, ax=ax[0, 2], kind='scatter', s=15, c=colors, cmap=cmap)
        test_df.plot(mean, ref, alpha=alpha, ax=ax[1, 2], kind='scatter', s=15, c=colors, cmap=cmap)

    if "unit" in kwargs.keys():
        if bayesian:
            ax[0, 1].set_ylabel(ref + "\n" + kwargs["unit"])
            ax[0, 1].set_xlabel(map + "\n" + kwargs["unit"])
            ax[1, 1].set_ylabel(ref + "\n" + kwargs["unit"])
            ax[1, 1].set_xlabel(map + "\n" + kwargs["unit"])
            ax[0, 2].set_ylabel(ref + "\n" + kwargs["unit"])
            ax[0, 2].set_xlabel(mean + "\n" + kwargs["unit"])
            ax[1, 2].set_ylabel(ref + "\n" + kwargs["unit"])
            ax[1, 2].set_xlabel(mean + "\n" + kwargs["unit"])
        else:
            ax[0, 1].set_ylabel(ref + "\n" + kwargs["unit"])
            ax[0, 1].set_xlabel(nn + "\n" + kwargs["unit"])
            ax[1, 1].set_ylabel(ref + "\n" + kwargs["unit"])
            ax[1, 1].set_xlabel(nn + "\n" + kwargs["unit"])
    else:
        if bayesian:
            ax[0, 1].set_ylabel(ref)
            ax[0, 1].set_xlabel(map)
            ax[1, 1].set_ylabel(ref)
            ax[1, 1].set_xlabel(map)
            ax[0, 2].set_ylabel(ref)
            ax[0, 2].set_xlabel(mean)
            ax[1, 2].set_ylabel(ref)
            ax[1, 2].set_xlabel(mean)
        else:
            ax[0, 1].set_ylabel(ref)
            ax[0, 1].set_xlabel(nn)
            ax[1, 1].set_ylabel(ref)
            ax[1, 1].set_xlabel(nn)

    diag_line(train_df[map], train_df[ref], ax=ax[0, 1])
    diag_line(test_df[map], test_df[ref], ax=ax[1, 1])

    if bayesian:
        diag_line(train_df[mean], train_df[ref], ax=ax[0, 2])
        diag_line(test_df[mean], test_df[ref], ax=ax[1, 2])

    ax[0, 1].set_title('Training set', fontsize=14, fontweight='bold')
    ax[1, 1].set_title('Test set', fontsize=14, fontweight='bold')

    for each_ax in ax[:, 0]:
        each_ax.legend(loc='best')

    plt.tight_layout()
    plt.rcParams['savefig.dpi'] = 220
    if filename is not None:
        fig.savefig(filename)
    plt.show()
    return fig, ax
