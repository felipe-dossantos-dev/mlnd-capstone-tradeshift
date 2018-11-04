import pandas as pd
import numpy as np

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
