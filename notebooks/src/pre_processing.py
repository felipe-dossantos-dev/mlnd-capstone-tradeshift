import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


BOOL_FEATURES_MAP = {
    'NO': 0,
    'YES': 1,
    np.nan: np.nan
}


def tranform_boolean_features(df, columns):
    bool_df = df[columns].applymap(
        lambda x: BOOL_FEATURES_MAP[x]).fillna(0).astype('bool')
    df.drop(labels=columns, axis="columns", inplace=True)
    df[columns] = bool_df[columns]
    return df


def transform_sparse_content_features(df, columns):
    return pd.get_dummies(df, columns=columns, sparse=True, dtype=bool)


def scale_features(df, columns):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[columns])
    df[columns] = df_scaled
    return (df, scaler)


def pca_features(df, columns, n_components, prefix='PCA_'):
    pca = PCA(n_components=n_components, svd_solver='randomized')
    data_pca = pca.fit_transform(df[columns])

    df.drop(labels=columns, axis="columns", inplace=True)

    cols_pca_name = [str(prefix) + str(i) for i in range(n_components)]
    df_pca = pd.DataFrame(data=data_pca, columns=cols_pca_name)

    df = pd.concat([df, df_pca], axis=1)
    return (df, pca)
