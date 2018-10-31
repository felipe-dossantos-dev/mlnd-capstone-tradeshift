import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def categorical(dataframe, column):
    plt.figure()
    fig, ax = plt.subplots(figsize=(20, 10))

    cat_perc = pd.DataFrame({'count': dataframe.groupby(
        [column])[column].size()}).reset_index()
    cat_perc = cat_perc.nlargest(25, 'count')
    cat_perc.sort_values(by='count', ascending=False, inplace=True)
    # Bar plot
    # Order the bars descending on target mean
    sns.barplot(ax=ax, x=column, y='count',
                data=cat_perc, order=cat_perc[column])
    plt.ylabel('how many times', fontsize=18)
    plt.xlabel(column, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()


def correlation_map(dataframe, columns):
    correlations = dataframe[columns].corr()

    # Create color map ranging between two colors
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    fig, ax = plt.subplots(figsize=(40, 40))
    sns.heatmap(correlations,
                cmap=cmap,
                vmax=1.0,
                center=0,
                fmt='.2f',
                square=True,
                linewidths=.5,
                annot=True,
                cbar_kws={"shrink": .75})
    plt.show()
