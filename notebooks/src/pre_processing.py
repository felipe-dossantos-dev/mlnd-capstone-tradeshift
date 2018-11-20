import pandas as pd
import numpy as np
import os
import gc
import pickle
import describe as d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer

TRAIN_BOOL_TRANSFORM_PATH = "../working/1_train_bool_transform.pkl"
TRAIN_CONTENT_PATH = "../working/2_train_content.pkl"
TRAIN_NUMERICAL_PATH = "../working/3_train_numerical.pkl"
TRAIN_ONLY_CONTENT_ENCODING = "../working/4_train_only_content_encoding.pkl"
TRAIN_FLOAT_SCALING_PATH = "../working/5_train_float_scaling.pkl"
TRAIN_FLOAT_SCALER_PATH = "../working/6_train_float_scaler.pkl"
TRAIN_INT_SCALING_PATH = "../working/7_train_int_scaling.pkl"
TRAIN_INT_SCALER_PATH = "../working/8_train_int_scaler.pkl"
TRAIN_WH_PCA_PATH = "../working/9_train_wh_pca.pkl"
WH_PCA_PATH = "../working/10_wh_pca.pkl"

CONTENT_DICVECTOR = "../working/content_dicvector.pkl"


def tranform_boolean_features(df, columns):
    df.replace('YES', 1, inplace=True)
    df.replace('NO', 0, inplace=True)
    df[columns] = df[columns].fillna(0)
    df[columns] = df[columns].astype(np.float32)
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
    df_pca = pd.DataFrame(data=data_pca, columns=cols_pca_name, index=df.index)

    df = pd.concat([df, df_pca], axis=1)
    return (df, pca)


def tranform_bool_df(df, bool_vars):
    if not os.path.isfile(TRAIN_BOOL_TRANSFORM_PATH):
        df = tranform_boolean_features(df, bool_vars)
        df.to_pickle(TRAIN_BOOL_TRANSFORM_PATH)
        return df
    return pd.read_pickle(TRAIN_BOOL_TRANSFORM_PATH)


def transform_content_dummy(df, content_features):
    if not os.path.isfile(TRAIN_CONTENT_PATH):
        content_features_filter = [col for col in content_features]

        only_content_df = df.filter(
            content_features_filter, axis=1)
        only_content_df.to_pickle(TRAIN_CONTENT_PATH)
        print(only_content_df.columns)

        del only_content_df
        del df
        gc.collect()

        df = pd.read_pickle(TRAIN_BOOL_TRANSFORM_PATH)

        content_features_filter.append('id')
        not_content_df = df.filter(
            df.columns.difference(content_features_filter), axis=1)
        not_content_df.to_pickle(TRAIN_NUMERICAL_PATH)
        print(not_content_df.columns)

        del not_content_df
        del df
        gc.collect()
    return pd.read_pickle(TRAIN_CONTENT_PATH)


def transform_sparse(df, content_features):
    if not os.path.isfile(TRAIN_ONLY_CONTENT_ENCODING):
        df, vec = transform_sparse_content_features(
            df, content_features)
        pickle.dump(df, open(TRAIN_ONLY_CONTENT_ENCODING, "wb"))
        pickle.dump(vec, open(CONTENT_DICVECTOR, "wb"))
    return pickle.load(open(TRAIN_ONLY_CONTENT_ENCODING, 'rb'))


def transform_scale(df, cols, df_path, scaler_path):
    if not os.path.isfile(df_path):
        df, scaler = scale_features(df, cols)
        pickle.dump(scaler, open(scaler_path, "wb"))
        df.to_pickle(df_path)
    return pd.read_pickle(df_path)


def transform_scale_float(df, cols):
    return transform_scale(df, cols, TRAIN_FLOAT_SCALING_PATH,
                           TRAIN_FLOAT_SCALER_PATH)


def transform_scale_int(df, cols):
    return transform_scale(df, cols, TRAIN_INT_SCALING_PATH,
                           TRAIN_INT_SCALER_PATH)


def transform_pca(df, cols):
    if not os.path.isfile(TRAIN_WH_PCA_PATH):
        df, pca = pca_features(df, cols, 5, 'WH_PCA_')
        df.to_pickle(TRAIN_WH_PCA_PATH)
        pickle.dump(pca, open(WH_PCA_PATH, "wb"))
    return pd.read_pickle(TRAIN_WH_PCA_PATH)


def save_dataset(df, path):
    if not os.path.isfile(path):
        pickle.dump(df, open(path, "wb"))


def transform_all():
    meta = pickle.load(open(meta_path, "rb"))

    int_features = meta[(meta.dtype == 'int64') & (meta.role != 'id')].index
    float_features = meta[(meta.category == 'numerical')
                          & (meta.dtype == 'float64')].index
    bool_vars = meta[(meta.category == 'boolean')].index
    content_features = meta[(meta.category == 'content')].index
    wh_features = meta[(meta.category == 'height') |
                       (meta.category == 'width')].index
    # pre processing
    train_features = d.read_train_features()
    train_features = pre.tranform_boolean_features(train_features, bool_vars)

    train_features, int_scaler = pre.scale_features(
        train_features, int_features)
    pickle.dump(int_scaler, open('../working/16_int_scaler_train.pkl', "wb"))

    train_features, float_scaler = pre.scale_features(
        train_features, float_features)
    pickle.dump(float_scaler, open(
        '../working/17_float_scaler_train.pkl', "wb"))

    wh_features = meta[(meta.category == 'height') |
                       (meta.category == 'width')].index
    train_features, pca_wh = pre.pca_features(
        train_features, wh_features, 5, 'WH_PCA_')
    pickle.dump(pca_wh, open('../working/18_wh_pca_train.pkl', "wb"))

    content_df, vec = pre.transform_sparse_content_features(
        train_features, content_features)
    pickle.dump(pca_wh, open('../working/19_word_vec.pkl', "wb"))
    content_df = content_df.astype(np.float32)
    content_df.data = np.nan_to_num(content_df.data)

    train_features.drop(labels=['id'], axis="columns", inplace=True)
    train_features.drop(labels=content_features, axis="columns", inplace=True)
    return train_features


def save_all(X_train, X_test, X_content_train, X_content_test,
             y_train, y_test):
    pickle.dump(X_train, open('../working/19_x_train.pkl', 'wb'))
    pickle.dump(X_test, open('../working/20_x_test.pkl', 'wb'))
    pickle.dump(X_content_train, open(
        '../working/21_x_content_train.pkl', 'wb'))
    pickle.dump(X_content_test, open('../working/22_x_content_test.pkl', 'wb'))
    pickle.dump(y_train, open('../working/23_y_train.pkl', 'wb'))
    pickle.dump(y_test, open('../working/24_y_test.pkl', 'wb'))


def load_all():
    X_train = pickle.load(open('../working/19_x_train.pkl', 'rb'))
    X_test = pickle.load(open('../working/20_x_test.pkl', 'rb'))
    X_content_train = pickle.load(
        open('../working/21_x_content_train.pkl', 'rb'))
    X_content_test = pickle.load(
        open('../working/22_x_content_test.pkl', 'rb'))
    y_train = pickle.load(open('../working/23_y_train.pkl', 'rb'))
    y_test = pickle.load(open('../working/24_y_test.pkl', 'rb'))
    return X_train, X_test, X_content_train, X_content_test, y_train, y_test
