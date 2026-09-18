import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_missing_value(df, return_df=False, feature_alignment='horizontal', figsize=(15, 8)):
    """
    Visualisasi missing value sebagai heatmap.

    == Contoh ==
    from salasml.plot import plot_missing_value
    plot_missing_value(df)
    df_miss = plot_missing_value(df, return_df=True)

    == Parameter ==
    df               : pandas DataFrame
    return_df        : jika True, kembalikan DataFrame ringkasan missing value
    feature_alignment: {'vertical', 'horizontal'} atau {'v', 'h'}
    figsize          : ukuran figure (lebar, tinggi)
    """
    plt.figure(figsize=figsize)
    if feature_alignment in ['vertical', 'v', 'column', 'c']:
        sns.heatmap(~df.isna(), cbar=False, cmap="Blues")
    elif feature_alignment in ['horizontal', 'h', 'row', 'r']:
        sns.heatmap(~df.isna().T, cbar=False, cmap="Blues")
    else:
        raise Exception("Alignment yang didukung: {'vertical', 'horizontal'} atau {'v', 'h'}")
    plt.xticks(rotation=45, horizontalalignment='right')
    if return_df:
        df_miss = pd.DataFrame(df.isna().sum(), columns=['missing_value'])
        df_miss["%"] = round(df_miss / len(df) * 100, 2)
        return df_miss
