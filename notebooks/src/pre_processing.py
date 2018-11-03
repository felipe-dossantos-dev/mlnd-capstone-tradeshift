import pandas as pd
import numpy as np

BOOL_FEATURES_MAP = {'NO': False, 'YES': True, np.nan: np.nan}


def tranform_boolean_features(df, columns):
    bool_df = df[columns].applymap(lambda x: BOOL_FEATURES_MAP[x])
    df.drop(labels=columns, axis="columns", inplace=True)
    df[columns] = bool_df[columns]
    return df
