import pandas as pd


def read_train_features():
    return pd.read_csv('../data/raw/train.csv')


def read_test_features():
    return pd.read_csv('../data/raw/test.csv')


def create_features_meta(dataframe):
    data = []
    for f in dataframe.columns:

        # Defining the category
        if dataframe[f].dtype == int or dataframe[f].dtype == float:
            category = 'numerical'
        elif dataframe[f].dtype == object or dataframe[f].dtype == str:
            uniques = dataframe[f].unique().tolist()
            if 'YES' in uniques or 'NO' in uniques:
                category = 'boolean'
            else:
                category = 'content'

        # Defining the data type
        dtype = dataframe[f].dtype

        # Creating a Dict that contains all the metadata for the variable
        f_dict = {
            'varname': f,
            'category': category,
            'dtype': dtype
        }
        data.append(f_dict)

    meta = pd.DataFrame(
        data, columns=['varname', 'category', 'dtype'])
    meta.set_index('varname', inplace=True)
    return meta
