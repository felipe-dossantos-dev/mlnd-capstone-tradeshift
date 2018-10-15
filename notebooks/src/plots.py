import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def categorical(dataframe, collumn):
    plt.figure()
    fig, ax = plt.subplots(figsize=(20, 10))

    cat_perc = pd.DataFrame({'count': dataframe.groupby(
        [collumn])[collumn].size()}).reset_index()
    cat_perc.sort_values(by='count', ascending=False, inplace=True)
    # Bar plot
    # Order the bars descending on target mean
    sns.barplot(ax=ax, x=collumn, y='count',
                data=cat_perc, order=cat_perc[collumn])
    plt.ylabel('% count', collumnontsize=18)
    plt.xlabel(collumn, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()
