"""
    Description:
    Visualizations functions for flux partitioning.
    Authors: Weiwei Zhan, Pierre Tessier
 """
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
from brokenaxes import brokenaxes
from pathlib import Path
from scipy import stats


def diag_line(x, y, ax, color='black', xy=(.05, .76)):
    """
    Add a dotted diagonal line to a plot.

    :param x: x_values of the observations
    :type x: pd.Series
    :param y: y_values of the observations
    :type y: pd.Series
    :param ax: axes to plot the line on
    :type ax: plt.Axes
    :param color: color of the line
    :type color: str
    :param xy: annotations position
    :type xy: (float, float)
    """
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
    """
    Create a plot with two dataset results on top of each other (usually training and testing).

    It can work with or without uncertainty visualization, MAP visualization and only plots a single variable (NEE, GPP
    or Reco).

    :param train_df: first dataset to use
    :type train_df: pd.DataFrame
    :param test_df: second dataset to use
    :type test_df: pd.DataFrame
    :param key: name of the variable to observe: "NEE" | "GPP" | "Reco"
    :type key: str
    :param reference: postfix of the target values
    :type reference: str
    :param filename: path to use to save the visuals (not saved if None is provided)
    :type filename: str | Path | None
    :param colors: name of a variable to use to color diagonal plots
    :type colors: str
    :param bayesian: if set to True, it will try to plot uncertainties and MAP
    :type bayesian: bool
    :param kwargs: additional parameters include `unit` (add a unit to axes), `postfix` (add a postfix to titles),
    `data_break` (break the axis of the first data set at provided date)
    :return: figure objects
    :rtype: (plt.Figure, np.ndarray[plt.Axes])
    """
    if bayesian:
        fig = plt.figure(figsize=(25, 10))
        gs = GridSpec(2, 3, width_ratios=[3, 1, 1], figure=fig)
    else:
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 2, width_ratios=[3, 1], figure=fig)

    train_title = "Training Set"
    test_title = "Testing Set"
    if "postfix" in kwargs.keys():
        train_title += " - " + kwargs["postfix"]
        test_title += " - " + kwargs["postfix"]

    ax = np.empty(gs.get_geometry(), dtype=object)
    for i in range(gs.nrows):
        for j in range(gs.ncols):
            if (i, j) == (0, 0):
                if "date_break" in kwargs:
                    lim = kwargs["date_break"]
                    ax[0, 0] = brokenaxes(
                        xlims=(
                            (train_df.index.min(), train_df.loc[train_df.index < lim, :].index.max()),
                            (train_df.loc[train_df.index > lim, :].index.min(), train_df.index.max())
                        ),
                        subplot_spec=gs[0],
                        diag_color="w"
                    )
                else:
                    ax[0, 0] = fig.add_subplot(gs[0])
            else:
                ax[i, j] = fig.add_subplot(gs[i * gs.ncols + j])

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
    ax[0, 0].set_title(train_title, fontsize=14, fontweight='bold')

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
    ax[1, 0].set_title(test_title, fontsize=14, fontweight='bold')

    if bayesian:
        train_df.plot(mean, ref, alpha=alpha, ax=ax[0, 2], kind='scatter', s=15, c=colors, cmap=cmap)
        test_df.plot(mean, ref, alpha=alpha, ax=ax[1, 2], kind='scatter', s=15, c=colors, cmap=cmap)
        train_df.plot(map, ref, alpha=alpha, ax=ax[0, 1], kind='scatter', s=15, c=colors, cmap=cmap)
        test_df.plot(map, ref, alpha=alpha, ax=ax[1, 1], kind='scatter', s=15, c=colors, cmap=cmap)
        diag_line(train_df[mean], train_df[ref], ax=ax[0, 2])
        diag_line(test_df[mean], test_df[ref], ax=ax[1, 2])
        diag_line(train_df[map], train_df[ref], ax=ax[0, 1])
        diag_line(test_df[map], test_df[ref], ax=ax[1, 1])
    else:
        train_df.plot(nn, ref, alpha=alpha, ax=ax[0, 1], kind='scatter', s=15, c=colors, cmap=cmap)
        test_df.plot(nn, ref, alpha=alpha, ax=ax[1, 1], kind='scatter', s=15, c=colors, cmap=cmap)
        diag_line(train_df[nn], train_df[ref], ax=ax[0, 1])
        diag_line(test_df[nn], test_df[ref], ax=ax[1, 1])

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

    ax[0, 1].set_title(train_title, fontsize=14, fontweight='bold')
    ax[1, 1].set_title(test_title, fontsize=14, fontweight='bold')
    if bayesian:
        ax[0, 2].set_title(train_title, fontsize=14, fontweight='bold')
        ax[1, 2].set_title(test_title, fontsize=14, fontweight='bold')

    for each_ax in ax[:, 0]:
        each_ax.legend(loc='best')

    plt.tight_layout()
    plt.rcParams['savefig.dpi'] = 220
    if filename is not None:
        fig.savefig(filename)
    return fig, ax


def dual_viz_val(val_df, key, reference="canopy", filename=None, colors=None, bayesian=True, **kwargs):
    """
    Same as previous function but for a single dataset (validation).

    :param val_df: dataset to use
    :type val_df: pd.DataFrame
    :param key: name of the variable to observe: "NEE" | "GPP" | "Reco"
    :type key: str
    :param reference: postfix of the target values
    :type reference: str
    :param filename: path to use to save the visuals (not saved if None is provided)
    :type filename: str | Path | None
    :param colors: name of a variable to use to color diagonal plots
    :type colors: str | None
    :param bayesian: if set to True, it will try to plot uncertainties and MAP
    :type bayesian: bool
    :param kwargs: additional parameters include `unit` (add a unit to axes), `postfix` (add a postfix to titles),
    `data_break` (break the axis of the first data set at provided date)
    :return: figure objects
    :rtype: (plt.Figure, np.ndarray[plt.Axes])
    """

    if bayesian:
        fig, ax = plt.subplots(figsize=(25, 6), nrows=1, ncols=3, gridspec_kw={'width_ratios': [3, 1, 1]})
    else:
        fig, ax = plt.subplots(figsize=(20, 6), nrows=1, ncols=2, gridspec_kw={'width_ratios': [3, 1]})

    s_circle = 4
    cmap = 'RdBu_r'
    alpha = 0.6

    ref = key + "_" + reference
    map = key + "_MAP"
    mean = key + "_mean"
    sigma = key + "_sigma"
    nn = key + "_NN"

    ax[0].scatter(val_df.index, val_df[ref], label=ref, s=s_circle)
    if bayesian:
        ax[0].scatter(val_df.index, val_df[map], label=map, s=s_circle, alpha=0.6)
        ax[0].scatter(val_df.index, val_df[mean], label=mean, s=s_circle, alpha=0.6)
        lower = val_df[mean] - 2 * val_df[sigma]
        upper = val_df[mean] + 2 * val_df[sigma]
        where = [True] * val_df.shape[0]
        where[(val_df.index < datetime(2014, 1, 1)).sum() - 1] = False
        ax[0].fill_between(val_df.index, lower, upper, where=where, facecolor="red", alpha=0.5,
                           label="Two std band")
    else:
        ax[0].scatter(val_df.index, val_df[nn], label=nn, s=s_circle, alpha=0.6)
    ax[0].set_ylabel(key)
    ax[0].set_title('Validation Set', fontsize=14, fontweight='bold')
    ax[1].set_title('Validation set', fontsize=14, fontweight='bold')
    if bayesian:
        ax[2].set_title('Validation set', fontsize=14, fontweight='bold')

    if bayesian:
        val_df.plot(mean, ref, alpha=alpha, ax=ax[2], kind='scatter', s=15, c=colors, cmap=cmap)
        val_df.plot(map, ref, alpha=alpha, ax=ax[1], kind='scatter', s=15, c=colors, cmap=cmap)
        diag_line(val_df[mean], val_df[ref], ax=ax[2])
        diag_line(val_df[map], val_df[ref], ax=ax[1])
    else:
        val_df.plot(nn, ref, alpha=alpha, ax=ax[1], kind='scatter', s=15, c=colors, cmap=cmap)
        diag_line(val_df[nn], val_df[ref], ax=ax[1])

    if "unit" in kwargs.keys():
        if bayesian:
            ax[1].set_ylabel(ref + "\n" + kwargs["unit"])
            ax[1].set_xlabel(map + "\n" + kwargs["unit"])
            ax[2].set_ylabel(ref + "\n" + kwargs["unit"])
            ax[2].set_xlabel(mean + "\n" + kwargs["unit"])
        else:
            ax[1].set_ylabel(ref + "\n" + kwargs["unit"])
            ax[1].set_xlabel(nn + "\n" + kwargs["unit"])
    else:
        if bayesian:
            ax[1].set_ylabel(ref)
            ax[1].set_xlabel(map)
            ax[2].set_ylabel(ref)
            ax[2].set_xlabel(mean)
        else:
            ax[1].set_ylabel(ref)
            ax[1].set_xlabel(nn)

    ax[0].legend()

    plt.tight_layout()
    plt.rcParams['savefig.dpi'] = 220
    if filename is not None:
        fig.savefig(filename)
    return fig, ax
