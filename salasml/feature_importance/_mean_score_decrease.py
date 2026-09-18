import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def mean_score_decrease(X_train, y_train, model, plot=False, topk=None, n_fold=5, normalize=False, random_state=42):
    """
    Hitung mean score decrease (permutation importance) untuk setiap fitur.

    == Contoh ==
    df_imp = mean_score_decrease(X_train, y_train, model, plot=True, topk=10)

    == Parameter ==
    X_train     : pandas DataFrame — fitur training
    y_train     : pandas Series — label training
    model       : sklearn pipeline atau estimator yang sudah di-fit
    plot        : tampilkan barplot jika True
    topk        : tampilkan k fitur terpenting saja
    n_fold      : jumlah permutasi (default 5)
    normalize   : normalisasi importance agar totalnya 1
    random_state: random seed
    """
    imp = permutation_importance(model, X_train, y_train, n_repeats=n_fold, n_jobs=1, random_state=random_state)

    df_imp = pd.DataFrame({
        "feature"   : X_train.columns,
        "importance": imp["importances_mean"],
        "stdev"     : imp["importances_std"]
    }).sort_values("importance", ascending=False)

    if normalize:
        df_imp[["importance", "stdev"]] = df_imp[["importance", "stdev"]] / df_imp.importance.sum()

    if topk:
        df_imp = df_imp.head(topk)

    if plot:
        plt.figure(figsize=(15, 5))
        plt.bar(range(len(df_imp)), df_imp.importance, yerr=df_imp.stdev, color='c', error_kw={"capsize": 5})
        plt.xticks(range(len(df_imp)), df_imp.feature, rotation=45, horizontalalignment='right')
        plt.ylabel('importance')
        plt.title("Mean Score Decrease", fontsize=14)

    return df_imp
