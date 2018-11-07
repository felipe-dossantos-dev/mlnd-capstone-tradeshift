import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

BOOL_FEATURES_MAP = {
    'NO': 0,
    'YES': 1,
    np.nan: np.nan
}


def tranform_boolean_features(df, columns):
    bool_df = df[columns].applymap(
        lambda x: BOOL_FEATURES_MAP[x]).astype('bool')
    df.drop(labels=columns, axis="columns", inplace=True)
    df[columns] = bool_df[columns]
    return df


def tranform_int_features(df, columns):
    df[columns] = df[columns].apply(pd.to_numeric,
                                    downcast='integer', errors='coerce')
    return df


def tranform_float_features(df, columns):
    df[columns] = df[columns].apply(pd.to_numeric,
                                    downcast='float', errors='coerce')
    return df


def extract_uniques_words(df, columns):
    uniques = set()
    for c in columns:
        uniques.update(df[c].unique().tolist())
    return dict(zip(uniques, range(len(uniques))))


def tranform_content_features(df, columns, uniques):
    uniques[np.nan] = -1
    uniques[float('nan')] = -1
    cont_df = df[columns].applymap(
        lambda x: uniques[x] if pd.notnull(x) else -1).astype('int32')
    df.drop(labels=columns, axis="columns", inplace=True)
    df[columns] = cont_df
    return df


def transform_sparse_content_features(df, columns):
    return pd.get_dummies(df, columns=columns, sparse=True, dtype=bool)


def scale_features(df, columns):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[columns])
    df[columns] = df_scaled
    return (df, scaler)
