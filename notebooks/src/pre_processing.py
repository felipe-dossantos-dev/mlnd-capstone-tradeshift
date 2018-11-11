import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer


def tranform_boolean_features(df, columns):
    df.replace('YES', 1, inplace = True)
    df.replace('NO', 0, inplace = True)
    df[columns] = df[columns].astype(np.float32)
    df[columns] = df[columns].fillna(0)
    return df


def transform_sparse_content_features(df, columns):
    vec = DictVectorizer()
    df_vec = vec.fit_transform(df[columns].T.to_dict().values())
    return df_vec, vec


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
