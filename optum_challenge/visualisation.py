import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List
from pandas import DataFrame


def keep_levels_greater_than_pct(df: DataFrame, column: str, pct: float = 0.01) -> List:

    """
    Outputs levels of a categorical column where the amount of data a level occupies is greater than param pct

    :param df: DataFrame
    :param column: str, column name
    :param pct: float, percentage required to keep level
    :return keep: list, levels to keep
    """

    keep = df[column].value_counts(normalize=True)

    keep = list(keep[keep >= pct].index)

    return keep


def plot_bar_chart(df: DataFrame, column: str, target_column: str = 'status_group'):

    """
    Plots bar chart

    :param df: DataFrame
    :param column: str
    :param target_column: str, default status_group
    """

    colours = {'functional': 'blue',
               'non functional': 'orange',
               'functional needs repair': 'green'}

    plt.figure(figsize=(20, 10))
    plt.xticks(rotation=30)
    print(sns.countplot(df[column], hue=df[target_column], palette=colours))


def plot_normalised_bar_chart_hue(df, column, target_column='status_group', dropna=True):

    """
    Plots normalised bar chart

    :param df: DataFrame
    :param column: str
    :param target_column: str, default status_group
    :param dropna: bool, drop nulls or not
    """
    norm = (df
            .groupby(column)[target_column]
            .value_counts(normalize=True, dropna=dropna)
            .mul(100)
            .rename('pct')
            .reset_index())

    colours = {'functional': 'blue',
               'non functional': 'orange',
               'functional needs repair': 'green'}

    plot = sns.catplot(x=column, y='pct', hue=target_column, kind='bar', data=norm, palette=colours)
    plot.fig.set_figheight(10)
    plot.fig.set_figwidth(30)
    plt.xticks(rotation=30)

    print(plot)
