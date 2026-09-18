import os
from warnings import warn

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (PolynomialFeatures, PowerTransformer,
                                   StandardScaler, MinMaxScaler, RobustScaler,
                                   OneHotEncoder, OrdinalEncoder, Normalizer, MaxAbsScaler)
from sklearn.pipeline import Pipeline


def num_pipe(impute='median', poly=None, transform=None, scaling=None, memory=None, n_neighbors=5, weights="uniform"):
    """
    Pipeline numerik otomatis untuk dipakai di ColumnTransformer.

    == Contoh ==
    from salasml.pipeline import num_pipe
    preprocessor = ColumnTransformer([
        ('numeric', num_pipe(scaling='standard'), num_cols)
    ])

    == Parameter ==
    impute   : {'knn', 'mean', 'median', None}
    scaling  : {'standard', 'minmax', 'robust', 'maxabs', 'normalize', None}
    transform: {'yeo-johnson', 'box-cox', None}
    poly     : int atau None — degree polynomial features
    """
    if impute not in ['knn', 'mean', 'median', None]:
        raise Exception("impute hanya mendukung {'knn', 'mean', 'median', None}")
    if scaling not in ['standard', 'minmax', 'robust', 'maxabs', 'normalize', None]:
        raise Exception("scaling hanya mendukung {'standard', 'minmax', 'robust', 'maxabs', 'normalize'}")
    if transform not in ['yeo-johnson', 'box-cox', None]:
        raise Exception("transform hanya mendukung {'yeo-johnson', 'box-cox'}")
    if (type(poly) is not int) and (poly is not None):
        raise Exception("poly harus int atau None")

    if impute is None:
        steps = []
    elif impute == "knn":
        steps = [('imputer', KNNImputer(n_neighbors=n_neighbors, weights=weights))]
    else:
        steps = [('imputer', SimpleImputer(strategy=impute))]

    if poly is not None:
        steps.append(('poly', PolynomialFeatures(poly)))

    if transform is not None and scaling is not None:
        warn("PowerTransformer sudah include standardisasi, argument scaling diabaikan")

    if transform is not None:
        steps.append(('transformer', PowerTransformer(transform)))
    elif scaling == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif scaling == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
    elif scaling == 'robust':
        steps.append(('scaler', RobustScaler()))
    elif scaling == 'maxabs':
        steps.append(('scaler', MaxAbsScaler()))
    elif scaling == 'normalize':
        steps.append(('scaler', Normalizer()))

    if memory is not None:
        os.makedirs('search_cache', exist_ok=True)
        return Pipeline(steps, memory='search_cache')
    else:
        return Pipeline(steps)


def cat_pipe(impute='most_frequent', encoder='onehot', memory=False):
    """
    Pipeline kategorikal otomatis untuk dipakai di ColumnTransformer.

    == Contoh ==
    from salasml.pipeline import cat_pipe
    preprocessor = ColumnTransformer([
        ('categoric', cat_pipe(encoder='onehot'), cat_cols)
    ])

    == Parameter ==
    impute : {'most_frequent', None}
    encoder: {'onehot', 'ordinal', None}
    """
    if impute not in ['most_frequent', None]:
        raise Exception("impute hanya mendukung {'most_frequent', None}")
    if encoder not in ['onehot', 'ordinal', None]:
        raise Exception("encoder hanya mendukung {'onehot', 'ordinal', None}")

    if impute is None:
        steps = []
    else:
        steps = [('imputer', SimpleImputer(strategy=impute))]

    if encoder == 'onehot':
        steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
    elif encoder == 'ordinal':
        steps.append(('ordinal', OrdinalEncoder()))

    if memory:
        os.makedirs('search_cache', exist_ok=True)
        return Pipeline(steps, memory='search_cache')
    else:
        return Pipeline(steps)
